import { defineConfig } from "vite";

export default defineConfig({
  root: ".",
  publicDir: "public",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8092",
      "/emergency": "http://127.0.0.1:8092",
      "/health": "http://127.0.0.1:8092",
    },
  },
});
