import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  base: "/Secundo-v-1.0/", // 👈 Important: your GitHub repo name here
  build: {
    outDir: "dist", // 👈 the folder GitHub Pages will publish
  },
  server: {
    host: "0.0.0.0",
    port: 8080,
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
