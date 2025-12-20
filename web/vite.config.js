import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

export default defineConfig(({ command }) => ({
  // dev: serve at /
  // build (GitHub Pages): serve under /ada-2025-project-radatouille/
  base: command === "serve" ? "/" : "/ada-2025-project-radatouille/",
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        intro: resolve(__dirname, "index.html"),
        community: resolve(__dirname, "/community/index.html"),
      },
    },
  },
}));