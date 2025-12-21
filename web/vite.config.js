import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

export default defineConfig(({ command }) => ({
  base: command === "serve" ? "/" : "/ada-2025-project-radatouille/",
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        intro: resolve(__dirname, "index.html"),
        conclusion: resolve(__dirname, "conclusion.html"), // ✅ add this
        community: resolve(__dirname, "community/index.html"), // ✅ remove leading /
      },
    },
  },
}));