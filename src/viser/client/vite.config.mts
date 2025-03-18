import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { vanillaExtractPlugin } from "@vanilla-extract/vite-plugin";
import viteTsconfigPaths from "vite-tsconfig-paths";
import svgrPlugin from "vite-plugin-svgr";
import eslint from "vite-plugin-eslint";
import browserslistToEsbuild from "browserslist-to-esbuild";
import basicSsl from "@vitejs/plugin-basic-ssl"; // Import the plugin

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    eslint({ failOnError: false, failOnWarning: false }),
    viteTsconfigPaths(),
    svgrPlugin(),
    vanillaExtractPlugin(),
    basicSsl(), // Add the plugin here
  ],
  server: {
    port: 3000,
    hmr: { port: 1025 },
    https: true, // Enable HTTPS
  },
  worker: {
    format: "es",
  },
  build: {
    outDir: "build",
    target: browserslistToEsbuild(),
  },
});
