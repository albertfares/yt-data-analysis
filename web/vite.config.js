import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/ada-2025-project-radatouille/", // <-- repo name, with slashes
  plugins: [react()],
});