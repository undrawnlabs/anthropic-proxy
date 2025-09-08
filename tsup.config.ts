import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/server.ts"],
  outDir: "dist",
  format: ["esm"],          // keep ESM (matches "type":"module")
  target: "node18",
  platform: "node",
  sourcemap: true,
  minify: true,
  clean: true,
  splitting: false,
  dts: false,
  banner: { js: "#!/usr/bin/env node" } // makes the output executable if you use it as a CLI
});
