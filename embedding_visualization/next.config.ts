import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */

  experimental: {
    reactCompiler: true,
  },

  webpack: (config) => {
    // glslify is a browserify shader compiler that regl-scatter2d lists as a
    // dependency but never calls at runtime (shaders are pre-compiled).
    // Ignoring it silences "Critical dependency" warnings from webpack.
    config.resolve.alias['glslify'] = false;
    return config;
  },
};

export default nextConfig;
