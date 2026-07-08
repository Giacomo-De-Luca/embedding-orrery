/**
 * Minimal raw-WebGL2 resource helpers for the density overlay — a
 * de-dataflowed take on embedding-atlas's webgl2_renderer/utils.ts
 * (Apple MIT). No caching harness: resources are created once by
 * DensityRenderer and explicitly disposed.
 */

export interface GlProgram {
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation>;
}

export interface GlFramebuffer {
  framebuffer: WebGLFramebuffer;
  texture: WebGLTexture;
  width: number;
  height: number;
  channels: 1 | 4;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Failed to create shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile failed: ${log}`);
  }
  return shader;
}

export function createProgram(
  gl: WebGL2RenderingContext,
  vertexSource: string,
  fragmentSource: string,
): GlProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  if (!program) throw new Error('Failed to create program');
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  // Shaders can be flagged for deletion once linked.
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link failed: ${log}`);
  }

  const uniforms: Record<string, WebGLUniformLocation> = {};
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
  for (let i = 0; i < count; i++) {
    const info = gl.getActiveUniform(program, i);
    if (!info) continue;
    const name = info.name.replace(/\[0\]$/, '');
    const location = gl.getUniformLocation(program, info.name);
    if (location) uniforms[name] = location;
  }
  return { program, uniforms };
}

export function createStaticBuffer(
  gl: WebGL2RenderingContext,
  data: Float32Array | Uint8Array | Int8Array,
): WebGLBuffer {
  const buffer = gl.createBuffer();
  if (!buffer) throw new Error('Failed to create buffer');
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return buffer;
}

/**
 * Float32 framebuffer (R32F or RGBA32F) with LINEAR filtering and
 * CLAMP_TO_EDGE, matching the embedding-atlas density pipeline. Requires
 * EXT_color_buffer_float (+ OES_texture_float_linear for the LINEAR filter).
 */
export function createFramebuffer(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  channels: 1 | 4,
): GlFramebuffer {
  const texture = gl.createTexture();
  const framebuffer = gl.createFramebuffer();
  if (!texture || !framebuffer) throw new Error('Failed to create framebuffer');

  gl.bindTexture(gl.TEXTURE_2D, texture);
  const internalFormat = channels === 1 ? gl.R32F : gl.RGBA32F;
  const format = channels === 1 ? gl.RED : gl.RGBA;
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);

  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  return { framebuffer, texture, width, height, channels };
}

export function disposeFramebuffer(gl: WebGL2RenderingContext, fb: GlFramebuffer): void {
  gl.deleteFramebuffer(fb.framebuffer);
  gl.deleteTexture(fb.texture);
}
