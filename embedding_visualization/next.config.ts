import path from "path";
import type { NextConfig } from "next";

const isDockerBuild = process.env.ORRERY_DOCKER_BUILD === "1";

const nextConfig: NextConfig = {
  /* config options here */
  output: "standalone",
  eslint: {
    ignoreDuringBuilds: isDockerBuild,
  },
  typescript: {
    ignoreBuildErrors: isDockerBuild,
  },

  experimental: {
    reactCompiler: true,
  },

  // Old route names — query strings are preserved automatically.
  async redirects() {
    return [
      { source: "/features", destination: "/sae", permanent: true },
      { source: "/test-embed", destination: "/collections", permanent: true },
    ];
  },

  webpack: (config) => {
    // The regl-* packages wrap their pre-compiled shaders in runtime
    // glslify() calls, so glslify cannot be aliased to `false` (that makes
    // createScatter() throw and scattergl render nothing). The stub keeps
    // webpack's "Critical dependency" warnings silenced while behaving as
    // the tagged-template join glslify performs at runtime.
    config.resolve.alias['glslify'] = path.resolve(__dirname, 'glslify-runtime-stub.js');
    return config;
  },
};

export default nextConfig;
