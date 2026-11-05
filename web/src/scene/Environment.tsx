import { useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Sky, Stars } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';

export function Lighting({ night, centre }: { night: boolean; centre: [number, number] }) {
  const light = useRef<THREE.DirectionalLight>(null);
  const { scene } = useThree();
  useMemo(() => {
    scene.fog = new THREE.Fog(night ? '#141220' : '#f3c99a', 260, 900);
    scene.background = new THREE.Color(night ? '#0b0a14' : '#f6d2a4');
  }, [scene, night]);
  const sun: [number, number, number] = night ? [-200, 60, -100] : [300, 120, -160];
  return (
    <group>
      {night ? <Stars radius={600} depth={50} count={1800} factor={4} fade /> : <Sky sunPosition={[sun[0], 22, sun[2]]} turbidity={6} rayleigh={2.2} mieCoefficient={0.02} mieDirectionalG={0.9} distance={3000} />}
      <hemisphereLight args={[night ? '#2a2c5a' : '#ffe2c0', night ? '#0d0c12' : '#4e5a3a', night ? 0.35 : 0.55]} />
      <directionalLight
        ref={light}
        position={[centre[0] + sun[0], sun[1], centre[1] + sun[2]]}
        target-position={[centre[0], 0, centre[1]]}
        intensity={night ? 0.25 : 2.6}
        color={night ? '#8fa0ff' : '#ffb570'}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-220}
        shadow-camera-right={220}
        shadow-camera-top={220}
        shadow-camera-bottom={-220}
        shadow-camera-near={10}
        shadow-camera-far={900}
        shadow-bias={-0.0004}
      />
      <ambientLight intensity={night ? 0.12 : 0.25} color={night ? '#4a4a8a' : '#ffd0a0'} />
    </group>
  );
}

export function Rain({ centre, reducedMotion }: { centre: [number, number]; reducedMotion: boolean }) {
  const n = 2500;
  const ref = useRef<THREE.Points>(null);
  const positions = useMemo(() => {
    const a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      a[i * 3] = centre[0] + (Math.random() - 0.5) * 320;
      a[i * 3 + 1] = Math.random() * 80;
      a[i * 3 + 2] = centre[1] + (Math.random() - 0.5) * 320;
    }
    return a;
  }, [centre]);
  useFrame((_, delta) => {
    const p = ref.current;
    if (!p || reducedMotion) return;
    const arr = p.geometry.getAttribute('position') as THREE.BufferAttribute;
    const a = arr.array as Float32Array;
    for (let i = 0; i < n; i++) {
      let y = (a[i * 3 + 1] ?? 0) - 28 * delta;
      if (y < 0) y = 80;
      a[i * 3 + 1] = y;
    }
    arr.needsUpdate = true;
  });
  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial color="#aab8d0" size={0.25} transparent opacity={0.6} sizeAttenuation />
    </points>
  );
}

export function NightBloom({ enabled }: { enabled: boolean }) {
  if (!enabled) return null;
  return (
    <EffectComposer>
      <Bloom luminanceThreshold={1.0} luminanceSmoothing={0.2} intensity={0.9} mipmapBlur />
    </EffectComposer>
  );
}
