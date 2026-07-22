import path from "path";
import type { NextConfig } from "next";

const isDockerBuild = process.env.ORRERY_DOCKER_BUILD === "1";
const isDemoMode = process.env.NEXT_PUBLIC_DEMO_MODE === "1";

const nextConfig: NextConfig = {
  /* config options here */
  output: "standalone",
  typescript: {
    ignoreBuildErrors: isDockerBuild,
  },

  // Promoted from `experimental.reactCompiler` to a top-level option in Next 16
  // (React Compiler is stable as of the 1.0 release). Disabled for Docker
  // builds: its Babel pass over the forked-plotly module graph OOM-kills
  // memory-capped build VMs (HF Space builder).
  reactCompiler: !isDockerBuild,

  // Old route names — query strings are preserved automatically.
  async redirects() {
    return [
      { source: "/features", destination: "/sae", permanent: true },
      { source: "/test-embed", destination: "/collections", permanent: true },
      // Read-only demo builds expose the Explore page only.
      ...(isDemoMode
        ? [
            { source: "/collections/:path*", destination: "/", permanent: false },
            { source: "/collections", destination: "/", permanent: false },
            { source: "/sae/:path*", destination: "/", permanent: false },
            { source: "/sae", destination: "/", permanent: false },
          ]
        : []),
    ];
  },

  // Next 16 makes Turbopack the default bundler for `next dev` and `next build`,
  // and Turbopack ignores the `webpack` function below, so the forked Plotly's
  // problematic imports must be re-declared here:
  //   - glslify: MUST resolve to the runtime stub (a tagged-template join), NOT
  //     an empty module. The regl-* packages call glslify() at runtime around
  //     their pre-compiled shaders; an empty module makes createScatter() throw
  //     (silently swallowed by react-plotly) and 2D scattergl renders nothing.
  //     Reuses the same glslify-runtime-stub.js the webpack alias uses.
  //   - maplibre-gl.css: required (for its style side-effect) by
  //     forked/plotly.js/src/registry.js for map traces this app never uses.
  //     Turbopack cannot instantiate a CSS module required from JS ("module
  //     factory is not available"), which aborts the entire Plotly load. Alias
  //     the CSS request to an empty JS module (CSS->CSS does not work).
  turbopack: {
    resolveAlias: {
      glslify: "./glslify-runtime-stub.js",
      "maplibre-gl/dist/maplibre-gl.css": "./maplibre-css-stub.js",
    },
  },

  webpack: (config) => {
    // The regl-* packages wrap their pre-compiled shaders in runtime
    // glslify() calls, so glslify cannot be aliased to `false` (that makes
    // createScatter() throw and scattergl render nothing). The stub keeps
    // webpack's "Critical dependency" warnings silenced while behaving as
    // the tagged-template join glslify performs at runtime.
    // Only applies to the Docker build (`next build --webpack`); Turbopack
    // ignores this and uses turbopack.resolveAlias above.
    config.resolve.alias['glslify'] = path.resolve(__dirname, 'glslify-runtime-stub.js');
    return config;
  },
};

export default nextConfig;
