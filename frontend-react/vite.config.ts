import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: trình duyệt chỉ gọi cùng origin (/api/*), Vite chuyển sang uvicorn và cắt tiền tố /api.
// Docker: nginx làm đúng việc này nên mã frontend không đổi.
const target = process.env.API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
