import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import viteTsconfigPaths from "vite-tsconfig-paths";
import svgrPlugin from "vite-plugin-svgr";
import eslint from "vite-plugin-eslint";
import browserslistToEsbuild from "browserslist-to-esbuild";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), eslint(), viteTsconfigPaths(), svgrPlugin()],
  server: {
    port: 3000,
    hmr: { port: 1025 },
  },
  worker: {
    format: "es",
  },
  build: {
    outDir: "build",
    target: browserslistToEsbuild(),
  },
});
