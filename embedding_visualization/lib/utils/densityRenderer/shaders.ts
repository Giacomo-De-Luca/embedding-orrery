/**
 * GLSL ES 3.0 shader sources for the 2D density overlay.
 *
 * Ported (near-verbatim) from Apple's embedding-atlas WebGL2 renderer:
 *   references/embedding-atlas/packages/component/src/lib/webgl2_renderer/
 *   (fill_count_buffer.ts, gaussian_blur_2.ts, paint_density_map.ts,
 *    paint_contours.ts, gamma_correction.ts)
 * Copyright (c) 2025 Apple Inc. Licensed under MIT License.
 *
 * The one intentional change: the gamma-correction fragment is made
 * alpha-preserving (unpremultiply → linear→sRGB → re-premultiply) because our
 * output canvas overlays the Plotly plot instead of owning an opaque view.
 */

/** Fullscreen-quad vertex shader (shared by blur and paint passes). */
export const QUAD_VERTEX = `#version 300 es
precision highp float;
layout(location=0) in vec2 xy;
out vec2 uv;
void main() {
  gl_Position = vec4(xy, 0, 1);
  uv = (xy + 1.0) / 2.0;
}
`;

/** Point-splat vertex shader, category variant: one-hot into an RGBA channel. */
export const FILL_COUNT_VERTEX_CATEGORY = `#version 300 es
precision highp float;
uniform mat3 matrix;
layout(location=0) in float x;
layout(location=1) in float y;
layout(location=2) in int category;
out vec4 color;
void main() {
  gl_Position = vec4(matrix * vec3(x, y, 1), 1);
  if (category == 0) {
    color = vec4(1, 0, 0, 0);
  } else if (category == 1) {
    color = vec4(0, 1, 0, 0);
  } else if (category == 2) {
    color = vec4(0, 0, 1, 0);
  } else if (category == 3) {
    color = vec4(0, 0, 0, 1);
  }
  gl_PointSize = 1.0;
}
`;

/** Point-splat vertex shader, single-channel variant (muted mode). */
export const FILL_COUNT_VERTEX_PLAIN = `#version 300 es
precision highp float;
uniform mat3 matrix;
layout(location=0) in float x;
layout(location=1) in float y;
out vec4 color;
void main() {
  gl_Position = vec4(matrix * vec3(x, y, 1), 1);
  color = vec4(1, 0, 0, 0);
  gl_PointSize = 1.0;
}
`;

/**
 * Point-splat vertex shader, mean-color variant: each point contributes its
 * own (linearized) RGB into the color channels and 1 into alpha, so the
 * blurred buffer holds density-weighted color sums (RGB) and density (A).
 * Used when there are more categories than the 4 one-hot channels can carry.
 */
export const FILL_COUNT_VERTEX_COLOR = `#version 300 es
precision highp float;
uniform mat3 matrix;
layout(location=0) in float x;
layout(location=1) in float y;
layout(location=2) in vec3 pointColor;
out vec4 color;
void main() {
  gl_Position = vec4(matrix * vec3(x, y, 1), 1);
  color = vec4(pointColor, 1);
  gl_PointSize = 1.0;
}
`;

export const FILL_COUNT_FRAGMENT = `#version 300 es
precision highp float;
in vec4 color;
out vec4 outColor;
void main() {
  outColor = color;
}
`;

/**
 * Separable gaussian blur, σ ≈ 20 device px, applied as 4 pre-baked filter
 * passes per direction (8 draws total).
 */
export const BLUR_FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D image;
uniform vec2 resolution;
uniform vec2 direction;
in vec2 uv;
out vec4 outColor;

uniform float weight0;
uniform vec3 distances;
uniform vec3 weights;

void main() {
  vec4 color = texture(image, uv) * weight0;
  if (weights.x != 0.0) {
    color += texture(image, uv + direction * vec2(distances.x) / resolution) * weights.x;
    color += texture(image, uv - direction * vec2(distances.x) / resolution) * weights.x;
  }
  if (weights.y != 0.0) {
    color += texture(image, uv + direction * vec2(distances.y) / resolution) * weights.y;
    color += texture(image, uv - direction * vec2(distances.y) / resolution) * weights.y;
  }
  if (weights.z != 0.0) {
    color += texture(image, uv + direction * vec2(distances.z) / resolution) * weights.z;
    color += texture(image, uv - direction * vec2(distances.z) / resolution) * weights.z;
  }
  outColor = color;
}
`;

/** Pre-baked R20 blur filter passes: [distances(vec3), weight0, weights(vec3)]. */
export const BLUR_FILTERS_R20: [number[], number[], number[]][] = [
  [[1, 2, 3], [0.2288468365182578], [0.18230006506971572, 0.1356122230111784, 0.06766429365997693]],
  [[2, 6, 10], [0.09116254014100238], [0.23317759354726447, 0.18385867277788717, 0.03738246360434722]],
  [[3, 10, 20], [0.2950645715317288], [0.010918865853671198, 0.23773695670296047, 0.10381189167750389]],
  [[4, 16, 30], [0.20085957073474772], [0.14463019087130788, 0.17934533765938643, 0.07559468610193185]],
];

/** Pixel radius of the R20 blur (bandwidth 20, 3σ cutoff). */
export function gaussianBlurR20PixelRadius(radius: number): number {
  return Math.ceil(radius * 3);
}

/** Quantized density-band paint, up to 4 category channels mixed per pixel. */
export const PAINT_DENSITY_MAP_FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D source;
uniform vec2 resolution;
uniform float densityScaler;
uniform float quantizationStep;

uniform mat4 colorMatrix;
uniform int isDarkMode;
uniform float globalAlpha;

in vec2 uv;
out vec4 outColor;

/* Combine alphas with symmetric blending equation f(a, b) = a + b - ab. */
float combine_alphas(vec4 alphas) {
  float r = alphas.x + alphas.y - alphas.x * alphas.y;
  r = r + alphas.z - r * alphas.z;
  r = r + alphas.w - r * alphas.w;
  return r;
}

void main() {
  vec4 density = texture(source, uv) * densityScaler;

  if (density.x > 1.0 || density.y > 1.0 || density.z > 1.0 || density.w > 1.0) {
    density = density / max(max(max(density.x, density.y), density.z), density.w);
  } else {
    density = floor(density / quantizationStep) * quantizationStep;
  }

  if (density.x + density.y + density.z + density.w == 0.0) {
    discard;
  }

  float alpha = combine_alphas(density);

  density *= alpha / (density.x + density.y + density.z + density.w);

  vec3 c1 = colorMatrix[0].rgb * density.x;
  vec3 c2 = colorMatrix[1].rgb * density.y;
  vec3 c3 = colorMatrix[2].rgb * density.z;
  vec3 c4 = colorMatrix[3].rgb * density.w;
  vec3 c;

  if (isDarkMode == 0) {
    c = vec3(1.0) - alpha + c1 + c2 + c3 + c4;
  } else {
    c = c1 + c2 + c3 + c4;
  }

  outColor = vec4(c, 1.0) * alpha * globalAlpha;
}
`;

/**
 * Mean-color variants of the paint passes: density lives in the alpha
 * channel, RGB/A recovers the local density-weighted mean category color.
 * Same quantization/alpha structure as the channel variants.
 */
export const PAINT_DENSITY_MEAN_FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D source;
uniform vec2 resolution;
uniform float densityScaler;
uniform float quantizationStep;

uniform int isDarkMode;
uniform float globalAlpha;

in vec2 uv;
out vec4 outColor;

void main() {
  vec4 acc = texture(source, uv);
  float density = acc.a * densityScaler;

  if (density > 1.0) {
    density = 1.0;
  } else {
    density = floor(density / quantizationStep) * quantizationStep;
  }

  if (density == 0.0 || acc.a == 0.0) {
    discard;
  }

  vec3 meanColor = acc.rgb / acc.a;
  float alpha = density;

  vec3 c;
  if (isDarkMode == 0) {
    c = vec3(1.0) - alpha + meanColor * alpha;
  } else {
    c = meanColor * alpha;
  }

  outColor = vec4(c, 1.0) * alpha * globalAlpha;
}
`;

export const PAINT_CONTOURS_MEAN_FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D source;
uniform vec2 resolution;
uniform float densityScaler;
uniform float quantizationStep;
uniform float globalAlpha;

in vec2 uv;
out vec4 outColor;

float sample_density(vec2 uv) {
  float d = texture(source, uv).a * densityScaler;
  d = min(1.0, max(0.0, d));
  d = floor(d / quantizationStep);
  return d;
}

void main() {
  // Run the Sobel operator on the density (alpha) channel.
  float v11 = sample_density(uv + vec2(-1, -1) / resolution);
  float v12 = sample_density(uv + vec2(-1,  0) / resolution);
  float v13 = sample_density(uv + vec2(-1, +1) / resolution);
  float v21 = sample_density(uv + vec2( 0, -1) / resolution);
  float v23 = sample_density(uv + vec2( 0, +1) / resolution);
  float v31 = sample_density(uv + vec2(+1, -1) / resolution);
  float v32 = sample_density(uv + vec2(+1,  0) / resolution);
  float v33 = sample_density(uv + vec2(+1, +1) / resolution);
  float gx = v11 + v12 * 2.0 + v13 - v31 - v32 * 2.0 - v33;
  float gy = v11 + v21 * 2.0 + v31 - v13 - v23 * 2.0 - v33;
  float alpha = length(vec2(gx, gy)) * 0.2;
  alpha = min(1.0, max(0.0, alpha));

  // Color the line by the local mean category color.
  vec4 acc = texture(source, uv);
  vec3 meanColor = acc.a > 0.0 ? acc.rgb / acc.a : vec3(0.5);

  outColor = vec4(meanColor, 1.0) * alpha * globalAlpha;
}
`;

/** Sobel iso-contour paint over the quantized density, one channel per pass. */
export const PAINT_CONTOURS_FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D source;
uniform vec2 resolution;
uniform float densityScaler;
uniform float quantizationStep;
uniform vec4 channelMask;
uniform vec4 color;
uniform float globalAlpha;

in vec2 uv;
out vec4 outColor;

float sample_density(vec2 uv) {
  float d = dot(texture(source, uv), channelMask) * densityScaler;
  d = min(1.0, max(0.0, d));
  d = floor(d / quantizationStep);
  return d;
}

void main() {
  // Run the Sobel operator.
  float v11 = sample_density(uv + vec2(-1, -1) / resolution);
  float v12 = sample_density(uv + vec2(-1,  0) / resolution);
  float v13 = sample_density(uv + vec2(-1, +1) / resolution);
  float v21 = sample_density(uv + vec2( 0, -1) / resolution);
  float v23 = sample_density(uv + vec2( 0, +1) / resolution);
  float v31 = sample_density(uv + vec2(+1, -1) / resolution);
  float v32 = sample_density(uv + vec2(+1,  0) / resolution);
  float v33 = sample_density(uv + vec2(+1, +1) / resolution);
  float gx = v11 + v12 * 2.0 + v13 - v31 - v32 * 2.0 - v33;
  float gy = v11 + v21 * 2.0 + v31 - v13 - v23 * 2.0 - v33;
  // Derive alpha value from the result.
  float alpha = length(vec2(gx, gy)) * 0.2;
  alpha = min(1.0, max(0.0, alpha));
  outColor = color * alpha * globalAlpha;
}
`;

/**
 * Final linear→sRGB pass onto the visible canvas. The xyScaler crops the
 * safe-margin overscan back to the visible viewport. Alpha-preserving on
 * premultiplied input (divergence from Apple — see module header).
 */
export const GAMMA_VERTEX = `#version 300 es
precision highp float;
uniform vec2 xyScaler;
layout(location=0) in vec2 xy;
out vec2 uv;
void main() {
  gl_Position = vec4(xy * xyScaler, 0, 1);
  uv = (xy + 1.0) / 2.0;
}
`;

export const GAMMA_FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D source;
uniform float gamma;
in vec2 uv;
out vec4 outColor;
void main() {
  vec4 color = texture(source, uv);
  if (color.a > 0.0) {
    color.rgb = pow(color.rgb / color.a, vec3(1.0 / gamma)) * color.a;
  }
  outColor = color;
}
`;
