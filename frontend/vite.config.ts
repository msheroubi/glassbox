import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend (python -m backend.main) serves /api and /ws; Vite proxies both in dev.
const backend = process.env.GLASSBOX_BACKEND ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 900 }, // recharts dominates the single bundle; fine for a local console
  server: {
    port: 5173,
    host: true,
    proxy: {
      "/api": { target: backend, changeOrigin: true },
      "/ws": { target: backend.replace(/^http/, "ws"), ws: true, changeOrigin: true },
    },
  },
});
