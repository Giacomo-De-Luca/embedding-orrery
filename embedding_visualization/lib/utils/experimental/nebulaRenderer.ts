/**
 * WebGL nebula renderer for Plan B (Glow Particles).
 * Renders billboard sprites with Gaussian falloff and additive blending.
 */

const VERTEX_SHADER = `
  attribute vec3 aPosition;
  attribute float aOpacity;
  attribute float aSize;

  uniform mat4 uProjection;
  uniform mat4 uView;
  uniform mat4 uModel;
  uniform float uPixelRatio;

  varying float vOpacity;

  void main() {
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    vec4 viewPos = uView * worldPos;
    gl_Position = uProjection * viewPos;

    // Size attenuation based on distance from camera
    float dist = length(viewPos.xyz);
    gl_PointSize = aSize * uPixelRatio * (200.0 / max(dist, 0.1));

    vOpacity = aOpacity;
  }
`;

const FRAGMENT_SHADER = `
  precision mediump float;

  uniform vec3 uColor;
  varying float vOpacity;

  void main() {
    // Radial gradient: Gaussian falloff from center
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r = dot(uv, uv);
    float alpha = smoothstep(1.0, 0.0, r) * vOpacity;

    if (alpha < 0.001) discard;

    gl_FragColor = vec4(uColor * alpha, alpha);
  }
`;

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Nebula shader compile error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

export class NebulaRenderer {
  private gl: WebGLRenderingContext;
  private program: WebGLProgram | null = null;
  private positionBuffer: WebGLBuffer | null = null;
  private opacityBuffer: WebGLBuffer | null = null;
  private sizeBuffer: WebGLBuffer | null = null;
  private particleCount = 0;

  // Attribute locations
  private aPosition = -1;
  private aOpacity = -1;
  private aSize = -1;

  // Uniform locations
  private uProjection: WebGLUniformLocation | null = null;
  private uView: WebGLUniformLocation | null = null;
  private uModel: WebGLUniformLocation | null = null;
  private uColor: WebGLUniformLocation | null = null;
  private uPixelRatio: WebGLUniformLocation | null = null;

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
    this.initShaders();
  }

  private initShaders(): void {
    const { gl } = this;
    const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    if (!vs || !fs) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Nebula program link error:', gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      return;
    }

    this.program = program;

    // Cache locations
    this.aPosition = gl.getAttribLocation(program, 'aPosition');
    this.aOpacity = gl.getAttribLocation(program, 'aOpacity');
    this.aSize = gl.getAttribLocation(program, 'aSize');

    this.uProjection = gl.getUniformLocation(program, 'uProjection');
    this.uView = gl.getUniformLocation(program, 'uView');
    this.uModel = gl.getUniformLocation(program, 'uModel');
    this.uColor = gl.getUniformLocation(program, 'uColor');
    this.uPixelRatio = gl.getUniformLocation(program, 'uPixelRatio');

    // Create buffers
    this.positionBuffer = gl.createBuffer();
    this.opacityBuffer = gl.createBuffer();
    this.sizeBuffer = gl.createBuffer();

    // Clean up shaders (linked to program, no longer needed standalone)
    gl.deleteShader(vs);
    gl.deleteShader(fs);
  }

  updateParticles(positions: Float32Array, opacities: Float32Array, sizes: Float32Array): void {
    const { gl } = this;
    this.particleCount = positions.length / 3;

    if (this.positionBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    }
    if (this.opacityBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.opacityBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, opacities, gl.STATIC_DRAW);
    }
    if (this.sizeBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.sizeBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, sizes, gl.STATIC_DRAW);
    }
  }

  draw(
    projection: Float32Array,
    view: Float32Array,
    model: Float32Array,
    color: [number, number, number],
  ): void {
    const { gl, program } = this;
    if (!program || this.particleCount === 0) return;

    gl.useProgram(program);

    // Set uniforms
    gl.uniformMatrix4fv(this.uProjection, false, projection);
    gl.uniformMatrix4fv(this.uView, false, view);
    gl.uniformMatrix4fv(this.uModel, false, model);
    gl.uniform3fv(this.uColor, color);
    gl.uniform1f(this.uPixelRatio, window.devicePixelRatio || 1);

    // Bind position
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(this.aPosition);
    gl.vertexAttribPointer(this.aPosition, 3, gl.FLOAT, false, 0, 0);

    // Bind opacity
    gl.bindBuffer(gl.ARRAY_BUFFER, this.opacityBuffer);
    gl.enableVertexAttribArray(this.aOpacity);
    gl.vertexAttribPointer(this.aOpacity, 1, gl.FLOAT, false, 0, 0);

    // Bind size
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sizeBuffer);
    gl.enableVertexAttribArray(this.aSize);
    gl.vertexAttribPointer(this.aSize, 1, gl.FLOAT, false, 0, 0);

    // Draw
    gl.drawArrays(gl.POINTS, 0, this.particleCount);

    // Disable attribs
    gl.disableVertexAttribArray(this.aPosition);
    gl.disableVertexAttribArray(this.aOpacity);
    gl.disableVertexAttribArray(this.aSize);
  }

  dispose(): void {
    const { gl } = this;
    if (this.positionBuffer) gl.deleteBuffer(this.positionBuffer);
    if (this.opacityBuffer) gl.deleteBuffer(this.opacityBuffer);
    if (this.sizeBuffer) gl.deleteBuffer(this.sizeBuffer);
    if (this.program) gl.deleteProgram(this.program);
    this.positionBuffer = null;
    this.opacityBuffer = null;
    this.sizeBuffer = null;
    this.program = null;
  }
}
