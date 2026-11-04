import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

// Project-site base path for GitHub Pages (https://<owner>.github.io/<repo>/). Override with VITE_BASE.
const base = process.env.VITE_BASE ?? '/dhaka-rl-signals/';

export default defineConfig({
  base,
  plugins: [react()],
  resolve: {
    alias: {
      '@results': fileURLToPath(new URL('../results', import.meta.url)),
      '@models': fileURLToPath(new URL('../models', import.meta.url)),
    },
  },
  server: { fs: { allow: ['..'] } },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: {
        manualChunks: { three: ['three'], r3f: ['@react-three/fiber', '@react-three/drei', '@react-three/postprocessing', 'postprocessing'], charts: ['recharts'] },
      },
    },
  },
});
