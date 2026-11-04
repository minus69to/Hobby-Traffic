import { useEffect, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import * as THREE from 'three';
import type { CameraPreset } from '../store/playback';

/** Camera state shared by both split-view panes (mutable, not React state). */
export const camShared = {
  pos: new THREE.Vector3(),
  target: new THREE.Vector3(),
  goalPos: new THREE.Vector3(),
  goalTarget: new THREE.Vector3(),
  animating: false,
  initialised: false,
};

export function presetPose(preset: CameraPreset, centre: [number, number]): { pos: THREE.Vector3; target: THREE.Vector3 } {
  const [cx, cz] = centre;
  switch (preset) {
    case 'top':
      return { pos: new THREE.Vector3(cx, 150, cz + 0.01), target: new THREE.Vector3(cx, 0, cz) };
    case 'street':
      return { pos: new THREE.Vector3(cx + 13.5, 2.4, cz + 70), target: new THREE.Vector3(cx + 2, 2.0, cz + 5) };
    case 'follow':
    case 'aerial':
    default:
      return { pos: new THREE.Vector3(cx + 95, 95, cz + 95), target: new THREE.Vector3(cx, 0, cz) };
  }
}

export function CameraRig({
  primary,
  preset,
  centre,
  followPos,
  followHeading,
  reducedMotion,
  domElement,
}: {
  primary: boolean;
  preset: CameraPreset;
  centre: [number, number];
  followPos: (() => THREE.Vector3 | null) | null;
  followHeading: (() => number) | null;
  reducedMotion: boolean;
  domElement: HTMLElement | null;
}) {
  const camRef = useRef<THREE.PerspectiveCamera>(null);
  const controls = useRef<OrbitControlsImpl>(null);
  const { invalidate } = useThree();
  const followSmooth = useRef(new THREE.Vector3());

  useEffect(() => {
    if (!primary) return;
    const p = presetPose(preset, centre);
    if (!camShared.initialised) {
      camShared.pos.copy(p.pos);
      camShared.target.copy(p.target);
      camShared.initialised = true;
    }
    if (preset !== 'follow') {
      camShared.goalPos.copy(p.pos);
      camShared.goalTarget.copy(p.target);
      camShared.animating = true;
      if (reducedMotion) {
        camShared.pos.copy(p.pos);
        camShared.target.copy(p.target);
        camShared.animating = false;
      }
    }
    invalidate();
  }, [preset, centre, primary, reducedMotion, invalidate]);

  useFrame((_, delta) => {
    const cam = camRef.current;
    if (!cam) return;
    if (primary) {
      if (preset === 'follow' && followPos) {
        const fp = followPos();
        if (fp) {
          const h = followHeading ? followHeading() : 0;
          const yaw = (-h * Math.PI) / 180;
          const back = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw)).multiplyScalar(14);
          const goal = fp.clone().add(back).add(new THREE.Vector3(0, 6, 0));
          followSmooth.current.lerp(goal, Math.min(1, delta * 4));
          camShared.pos.copy(followSmooth.current);
          camShared.target.copy(fp).add(new THREE.Vector3(0, 1.2, 0));
        }
      } else if (camShared.animating) {
        const k = Math.min(1, delta * 3.2);
        camShared.pos.lerp(camShared.goalPos, k);
        camShared.target.lerp(camShared.goalTarget, k);
        if (camShared.pos.distanceTo(camShared.goalPos) < 0.3) camShared.animating = false;
      } else if (controls.current) {
        camShared.pos.copy(cam.position);
        camShared.target.copy(controls.current.target);
      }
      cam.position.copy(camShared.pos);
      if (controls.current) {
        controls.current.target.copy(camShared.target);
        controls.current.update();
      } else cam.lookAt(camShared.target);
    } else {
      cam.position.copy(camShared.pos);
      cam.lookAt(camShared.target);
    }
  });

  return (
    <>
      <PerspectiveCamera ref={camRef} makeDefault fov={45} near={0.5} far={2500} position={[centre[0] + 95, 95, centre[1] + 95]} />
      {primary && (
        <OrbitControls
          ref={controls}
          makeDefault
          enabled={preset !== 'follow'}
          enableDamping
          dampingFactor={0.12}
          maxPolarAngle={Math.PI / 2 - 0.03}
          minDistance={6}
          maxDistance={520}
          domElement={domElement ?? undefined}
          onStart={() => {
            camShared.animating = false;
          }}
        />
      )}
    </>
  );
}
