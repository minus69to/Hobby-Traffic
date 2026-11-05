import { CLASS_NAMES, type ClassName, type Replay, type ReplayHeader, type Track } from './types';

interface HeaderTrack {
  id: string;
  cls: number;
  first: number;
  n: number;
  x0: number;
  y0: number;
}

const MAGIC = 'DRSR';

/** Decode an uncompressed DRSR v1 blob (see src/drs/replay.py for the format). */
export function decodeReplay(buf: ArrayBuffer): Replay {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== MAGIC) throw new Error(`bad replay magic ${magic}`);
  const version = dv.getUint32(4, true);
  if (version !== 1) throw new Error(`unsupported replay version ${version}`);
  const hlen = dv.getUint32(8, true);
  const headerJson = new TextDecoder().decode(new Uint8Array(buf, 12, hlen));
  const header = JSON.parse(headerJson) as ReplayHeader & { tracks: HeaderTrack[] };
  let off = 12 + hlen;
  const nPos = header.n_pos;
  const pos = new Int16Array(buf.slice(off, off + nPos * 4));
  off += nPos * 4;
  const ang = new Uint8Array(buf, off, nPos);
  off += nPos;
  const spd = new Uint8Array(buf, off, nPos);
  off += nPos;
  const nj = header.junctions.length;
  const tls = new Uint8Array(buf.slice(off, off + header.n_frames * nj));
  const tracks: Track[] = [];
  const lastFrame = new Int32Array(header.tracks.length);
  let cursor = 0;
  const q = header.quant;
  const sq = header.speed_quant;
  const dt = header.dt;
  header.tracks.forEach((ht, k) => {
    const n = ht.n;
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const angle = new Float32Array(n);
    const speed = new Float32Array(n);
    const wait = new Float32Array(n);
    let cx = ht.x0;
    let cy = ht.y0;
    let w = 0;
    for (let i = 0; i < n; i++) {
      if (i > 0) {
        cx += pos[(cursor + i) * 2] ?? 0;
        cy += pos[(cursor + i) * 2 + 1] ?? 0;
      }
      x[i] = cx * q;
      y[i] = cy * q;
      angle[i] = ((ang[cursor + i] ?? 0) * 360) / 256;
      const s = (spd[cursor + i] ?? 0) * sq;
      speed[i] = s;
      w = s < 0.1 ? w + dt : 0;
      wait[i] = w;
    }
    const cls: ClassName = CLASS_NAMES[ht.cls] ?? 'car';
    tracks.push({ id: ht.id, cls, first: ht.first, n, x, y, angle, speed, wait });
    lastFrame[k] = ht.first + n - 1;
    cursor += n;
  });
  const rest: ReplayHeader & { tracks?: HeaderTrack[] } = { ...header };
  delete rest.tracks;
  return { header: rest, tracks, tls, lastFrame };
}

/**
 * Fetch a replay and decode it.  The file on disk is gzip-compressed; some static servers send it
 * with `Content-Encoding: gzip` (the browser then decompresses it transparently) while others (for
 * example GitHub Pages) send the raw bytes, so the magic bytes decide whether to inflate here.
 */
export async function fetchReplay(url: string, onProgress?: (loaded: number, total: number) => void): Promise<Replay> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`failed to fetch ${url}: ${res.status}`);
  const total = Number(res.headers.get('content-length') ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  const raw = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    raw.set(c, off);
    off += c.byteLength;
  }
  let buf: ArrayBuffer = raw.buffer;
  if (raw[0] === 0x1f && raw[1] === 0x8b) {
    const ds = new DecompressionStream('gzip') as unknown as ReadableWritablePair<Uint8Array, Uint8Array>;
    buf = await new Response(new Blob([raw]).stream().pipeThrough(ds)).arrayBuffer();
  }
  return decodeReplay(buf);
}

/** State of one track at a continuous time (seconds), linearly interpolated between samples. */
export interface TrackPose {
  x: number;
  y: number;
  angle: number;
  speed: number;
  wait: number;
  accel: number;
}

export function samplePose(tr: Track, dt: number, t: number, out: TrackPose): boolean {
  const f = t / dt - tr.first;
  if (f < 0 || f > tr.n - 1) return false;
  const i = Math.floor(f);
  const j = Math.min(i + 1, tr.n - 1);
  const a = f - i;
  const xi = tr.x[i] ?? 0;
  const yi = tr.y[i] ?? 0;
  out.x = xi + ((tr.x[j] ?? xi) - xi) * a;
  out.y = yi + ((tr.y[j] ?? yi) - yi) * a;
  const ai = tr.angle[i] ?? 0;
  const aj = tr.angle[j] ?? ai;
  let d = aj - ai;
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  out.angle = ai + d * a;
  const si = tr.speed[i] ?? 0;
  const sj = tr.speed[j] ?? si;
  out.speed = si + (sj - si) * a;
  out.accel = (sj - si) / dt;
  out.wait = tr.wait[i] ?? 0;
  return true;
}
