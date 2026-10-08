import basicSsl from '@vitejs/plugin-basic-ssl';
import { defineConfig } from 'vite';

export default defineConfig(({ mode }) => ({
  base: '/face-reception/',
  // `npm run dev:phone` serves over HTTPS on the LAN so an iPhone may use the camera.
  plugins: mode === 'phone' ? [basicSsl()] : [],
  // Pre-bundling breaks the relative URL onnxruntime-web uses to find its .wasm file.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
}));
