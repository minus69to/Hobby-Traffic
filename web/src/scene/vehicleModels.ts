import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { ClassName } from '../replay/types';

/**
 * Low-poly vehicle models built from primitives.  Each class has two merged geometries:
 * `body` (white vertex colours, tinted per instance) and `detail` (fixed vertex colours).
 * Model space: origin at the front bumper on the ground, forward = -Z, up = +Y, +Z = towards the rear.
 */
export interface ClassModel {
  body: THREE.BufferGeometry;
  detail: THREE.BufferGeometry;
  brake: THREE.BufferGeometry; // rear lights
  head: THREE.BufferGeometry; // headlights
  length: number;
  palette: string[];
}

function coloured(g: THREE.BufferGeometry, colour: string): THREE.BufferGeometry {
  const c = new THREE.Color(colour);
  const n = g.getAttribute('position').count;
  const arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    arr[i * 3] = c.r;
    arr[i * 3 + 1] = c.g;
    arr[i * 3 + 2] = c.b;
  }
  g.setAttribute('color', new THREE.BufferAttribute(arr, 3));
  return g;
}

function box(w: number, h: number, d: number, x: number, y: number, z: number, colour: string): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(w, h, d);
  g.translate(x, y, z);
  return coloured(g, colour);
}

function wheel(r: number, w: number, x: number, y: number, z: number, colour = '#15151a'): THREE.BufferGeometry {
  const g = new THREE.CylinderGeometry(r, r, w, 10);
  g.rotateZ(Math.PI / 2);
  g.translate(x, y, z);
  return coloured(g, colour);
}

function sphere(r: number, x: number, y: number, z: number, colour: string): THREE.BufferGeometry {
  const g = new THREE.SphereGeometry(r, 8, 6);
  g.translate(x, y, z);
  return coloured(g, colour);
}

function merge(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const list = parts.map((p) => (p.index ? p.toNonIndexed() : p));
  const m = mergeGeometries(list, false);
  if (!m) throw new Error('merge failed');
  return m;
}

function lights(w: number, h: number, zRear: number, zFront: number, y: number, hw: number): { brake: THREE.BufferGeometry; head: THREE.BufferGeometry } {
  return {
    brake: merge([box(w, h, 0.06, -hw, y, zRear, '#ff2020'), box(w, h, 0.06, hw, y, zRear, '#ff2020')]),
    head: merge([box(w, h, 0.06, -hw, y, zFront, '#fff4d0'), box(w, h, 0.06, hw, y, zFront, '#fff4d0')]),
  };
}

export function buildModels(): Record<ClassName, ClassModel> {
  const W = '#ffffff';
  const car = (() => {
    const L = 4.3;
    const body = merge([box(1.75, 0.55, L - 0.2, 0, 0.5, L / 2, W), box(1.55, 0.5, 2.0, 0, 1.0, L / 2 + 0.2, W)]);
    const detail = merge([
      box(1.5, 0.35, 1.9, 0, 1.05, L / 2 + 0.2, '#223040'),
      wheel(0.32, 0.22, -0.8, 0.32, 0.8),
      wheel(0.32, 0.22, 0.8, 0.32, 0.8),
      wheel(0.32, 0.22, -0.8, 0.32, L - 0.8),
      wheel(0.32, 0.22, 0.8, 0.32, L - 0.8),
    ]);
    const l = lights(0.3, 0.12, L - 0.08, 0.02, 0.62, 0.62);
    return { body, detail, ...l, length: L, palette: ['#e9e7e1', '#c9ccd1', '#2d2f33', '#8a1c1c', '#1f3f7a', '#5d6b7a', '#b9a46a', '#3b5f3a'] };
  })();
  const makeBus = (dd: boolean) => {
    const L = 11;
    const H = dd ? 4.4 : 2.9;
    const body = merge([box(2.45, H - 0.6, L - 0.3, 0, 0.6 + (H - 0.6) / 2, L / 2, W)]);
    const windows: THREE.BufferGeometry[] = [box(2.5, 0.8, L - 1.4, 0, 1.9, L / 2 + 0.2, '#1a2630')];
    if (dd) windows.push(box(2.5, 0.8, L - 1.4, 0, 3.5, L / 2 + 0.2, '#1a2630'));
    const detail = merge([
      ...windows,
      box(2.3, 0.9, 0.1, 0, 1.8, 0.03, '#1a2630'),
      wheel(0.5, 0.3, -1.1, 0.5, 1.8),
      wheel(0.5, 0.3, 1.1, 0.5, 1.8),
      wheel(0.5, 0.3, -1.1, 0.5, L - 2.6),
      wheel(0.5, 0.3, 1.1, 0.5, L - 2.6),
      box(2.4, 0.1, L - 0.4, 0, H + 0.02, L / 2, '#cfcfcf'),
    ]);
    const l = lights(0.35, 0.18, L - 0.08, 0.02, 0.9, 0.9);
    return { body, detail, ...l, length: L, palette: dd ? ['#c8201a', '#b81d18', '#d6281f'] : ['#2a6fb8', '#2f8f5a', '#d9a21b', '#3b3f8a', '#e0e0e0'] };
  };
  const truck = (() => {
    const L = 8.5;
    const body = merge([box(2.3, 2.0, 2.4, 0, 1.6, 1.3, W)]);
    const detail = merge([
      box(2.4, 2.4, 5.6, 0, 1.7, L - 2.9, '#6a6f62'),
      box(2.1, 0.8, 0.1, 0, 2.1, 0.08, '#1a2630'),
      wheel(0.5, 0.3, -1.0, 0.5, 1.2),
      wheel(0.5, 0.3, 1.0, 0.5, 1.2),
      wheel(0.5, 0.3, -1.0, 0.5, L - 1.6),
      wheel(0.5, 0.3, 1.0, 0.5, L - 1.6),
    ]);
    const l = lights(0.35, 0.18, L - 0.08, 0.02, 0.9, 0.85);
    return { body, detail, ...l, length: L, palette: ['#b8562a', '#4d6b9a', '#8c8f3a', '#9a3a3a'] };
  })();
  const motorcycle = (() => {
    const L = 2.0;
    const body = merge([box(0.45, 0.35, 1.3, 0, 0.6, L / 2, W)]);
    const detail = merge([
      wheel(0.3, 0.1, 0, 0.3, 0.3),
      wheel(0.3, 0.1, 0, 0.3, L - 0.3),
      box(0.5, 0.5, 0.35, 0, 1.0, 1.1, '#3a3f66'), // rider torso
      sphere(0.16, 0, 1.4, 1.1, '#f2c8a0'),
      box(0.6, 0.05, 0.1, 0, 0.95, 0.5, '#222'),
    ]);
    const l = lights(0.12, 0.08, L - 0.05, 0.05, 0.6, 0);
    return { body, detail, ...l, length: L, palette: ['#c8102e', '#1a1a1a', '#2b5fd9', '#d9d9d9'] };
  })();
  const cng = (() => {
    const L = 2.7;
    const body = merge([box(1.35, 1.1, L - 0.2, 0, 0.75, L / 2, W), box(1.35, 0.1, L - 0.4, 0, 1.6, L / 2, W)]);
    const detail = merge([
      box(1.4, 0.45, L - 0.9, 0, 1.3, L / 2 + 0.1, '#1b2b1f'), // caged window strip
      wheel(0.28, 0.15, 0, 0.28, 0.3),
      wheel(0.28, 0.15, -0.55, 0.28, L - 0.35),
      wheel(0.28, 0.15, 0.55, 0.28, L - 0.35),
    ]);
    const l = lights(0.2, 0.1, L - 0.06, 0.03, 0.7, 0.45);
    return { body, detail, ...l, length: L, palette: ['#1f8f3a', '#1a7f33', '#239944'] };
  })();
  const rickshaw = (() => {
    const L = 2.4;
    const body = merge([box(1.05, 0.45, 1.1, 0, 0.75, L - 0.6, W)]); // carriage (tinted)
    const hood = new THREE.CylinderGeometry(0.62, 0.62, 1.1, 8, 1, false, Math.PI, Math.PI);
    hood.rotateZ(Math.PI / 2);
    hood.rotateY(Math.PI / 2);
    hood.translate(0, 1.25, L - 0.7);
    const detail = merge([
      coloured(hood, '#f2b632'),
      wheel(0.33, 0.06, 0, 0.33, 0.35, '#333'),
      wheel(0.33, 0.06, -0.55, 0.33, L - 0.45, '#333'),
      wheel(0.33, 0.06, 0.55, 0.33, L - 0.45, '#333'),
      box(0.4, 0.5, 0.3, 0, 1.0, 0.9, '#7a4f8a'), // driver
      sphere(0.14, 0, 1.4, 0.9, '#e8b890'),
      box(0.1, 0.5, 1.0, 0, 0.45, 1.1, '#222'),
    ]);
    const l = lights(0.08, 0.06, L - 0.06, 0.1, 0.4, 0.3);
    return { body, detail, ...l, length: L, palette: ['#d6288a', '#2a8fd6', '#e5c02a', '#2fb36a', '#d63a2a', '#7a3ad6'] };
  })();
  const pedestrian = (() => {
    const body = merge([box(0.38, 0.55, 0.22, 0, 1.05, 0, W)]);
    const detail = merge([box(0.3, 0.75, 0.2, 0, 0.4, 0, '#3a3a4a'), sphere(0.13, 0, 1.5, 0, '#e2b48e')]);
    const empty = new THREE.BufferGeometry();
    empty.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
    return { body, detail, brake: empty, head: empty, length: 0.4, palette: ['#d94a3a', '#3a7ad9', '#e0d040', '#40a060', '#f0f0f0', '#8a3ab0', '#f08a30'] };
  })();
  return { car, bus: makeBus(false), bus_dd: makeBus(true), truck, motorcycle, cng, rickshaw, pedestrian };
}
