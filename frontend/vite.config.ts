import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Mesh2Motion is vendored source, built by this Vite app as a same-origin
  // workspace rather than being deployed as a separate application.
  publicDir: resolve(__dirname, '../vendor/mesh2motion-app/static'),
  define: { PROCESS_ENV: JSON.stringify({}) },
  resolve: {
    // Mesh2Motion lives outside frontend/, so resolve its packages from the
    // studio dependency tree instead of looking for a nested node_modules.
    alias: {
      three: resolve(__dirname, 'node_modules/three'),
      'tippy.js': resolve(__dirname, 'node_modules/tippy.js'),
      jszip: resolve(__dirname, 'node_modules/jszip'),
    },
  },
  server: {
    port: 5173,
    host: '127.0.0.1',
    fs: { allow: [resolve(__dirname, '..')] },
  },
  build: {
    rollupOptions: {
      input: {
        studio: resolve(__dirname, 'index.html'),
        mesh2motion: resolve(__dirname, 'mesh2motion.html'),
      },
    },
  },
})
