import * as THREE from 'three';

/** Deterministic PRNG (mulberry32) so the procedural city is identical on every load. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function pick<T>(r: () => number, arr: readonly T[]): T {
  return arr[Math.floor(r() * arr.length)] as T;
}

/** Ribbon (triangle strip) along a polyline in SUMO coordinates, lying flat at height y. */
export function ribbonGeometry(points: [number, number][], width: number, y = 0): THREE.BufferGeometry {
  const n = points.length;
  const pos: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  let dist = 0;
  for (let i = 0; i < n; i++) {
    const p = points[i]!;
    const prev = points[Math.max(i - 1, 0)]!;
    const next = points[Math.min(i + 1, n - 1)]!;
    let dx = next[0] - prev[0];
    let dy = next[1] - prev[1];
    const len = Math.hypot(dx, dy) || 1;
    dx /= len;
    dy /= len;
    // normal in SUMO plane (left of direction)
    const nx = -dy;
    const ny = dx;
    const hw = width / 2;
    // SUMO (x, y) -> world (x, -y)
    pos.push(p[0] + nx * hw, y, -(p[1] + ny * hw));
    pos.push(p[0] - nx * hw, y, -(p[1] - ny * hw));
    if (i > 0) dist += Math.hypot(p[0] - prev[0], p[1] - prev[1]);
    uv.push(0, dist, 1, dist);
    if (i < n - 1) {
      const a = i * 2;
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

/** Flat polygon (SUMO coordinates) at height y. */
export function polygonGeometry(points: [number, number][], y = 0): THREE.BufferGeometry {
  // ShapeGeometry lies in the XY plane; rotateX(-90deg) maps (x, Y, 0) -> (x, 0, -Y), so passing the
  // SUMO y as Y yields world z = -y, consistent with ribbonGeometry.
  const shape = new THREE.Shape(points.map(([x, y2]) => new THREE.Vector2(x, y2)));
  const g = new THREE.ShapeGeometry(shape);
  g.rotateX(-Math.PI / 2);
  g.translate(0, y, 0);
  return g;
}

export function polylineLength(points: [number, number][]): number {
  let d = 0;
  for (let i = 1; i < points.length; i++) d += Math.hypot(points[i]![0] - points[i - 1]![0], points[i]![1] - points[i - 1]![1]);
  return d;
}

/** Point and tangent at distance s along a polyline. */
export function pointAt(points: [number, number][], s: number): { x: number; y: number; tx: number; ty: number } {
  let acc = 0;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1]!;
    const b = points[i]!;
    const seg = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (acc + seg >= s || i === points.length - 1) {
      const f = seg > 0 ? Math.min(Math.max((s - acc) / seg, 0), 1) : 0;
      return { x: a[0] + (b[0] - a[0]) * f, y: a[1] + (b[1] - a[1]) * f, tx: (b[0] - a[0]) / (seg || 1), ty: (b[1] - a[1]) / (seg || 1) };
    }
    acc += seg;
  }
  const p = points[0]!;
  return { x: p[0], y: p[1], tx: 1, ty: 0 };
}

/** Canvas texture with a pastel facade and emissive-looking windows. */
export function facadeTexture(r: () => number, cols: number, rows: number, base: string, night: boolean): THREE.CanvasTexture {
  const w = 128;
  const h = 128;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, w, h);
  const cw = w / cols;
  const rh = h / rows;
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      const lit = r() < (night ? 0.55 : 0.0);
      ctx.fillStyle = night ? (lit ? `hsl(${40 + r() * 20} 90% ${55 + r() * 20}%)` : '#17131f') : `hsl(210 20% ${22 + r() * 18}%)`;
      ctx.fillRect(i * cw + cw * 0.25, j * rh + rh * 0.2, cw * 0.5, rh * 0.55);
      if (!night) {
        ctx.fillStyle = 'rgba(255,255,255,0.25)';
        ctx.fillRect(i * cw + cw * 0.25, j * rh + rh * 0.2, cw * 0.5, rh * 0.1);
      }
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.NearestFilter;
  return tex;
}

/** Sign texture with (fictional) Bengali shop text. */
export function signTexture(text: string, bg: string, fg: string, w = 256, h = 96): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(0,0,0,0.35)';
  ctx.lineWidth = 6;
  ctx.strokeRect(3, 3, w - 6, h - 6);
  ctx.fillStyle = fg;
  ctx.font = `bold ${Math.floor(h * 0.5)}px "Noto Sans Bengali", sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, w / 2, h / 2 + 4);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export const PASTELS = ['#e8c9a8', '#d9b8c4', '#c9d9c2', '#f0e3b2', '#bcd2e6', '#e6b8a2', '#d6c7e8', '#f2d0b0', '#c8dcd0', '#e4d4c0'];
export const SIGN_TEXTS = ['চা স্টল', 'ফার্মেসি', 'মুদি দোকান', 'টেইলার্স', 'হোটেল', 'ইলেকট্রনিক্স', 'বেকারি', 'সেলুন', 'মোবাইল', 'বইঘর'];
