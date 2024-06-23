import { SpriteInfo } from "./common";

export class Context {
  canvas: HTMLCanvasElement;
  gl: WebGL2RenderingContext;
  viewport: Viewport;
  spriteSize: number = 1;
  atlas!: TextureAtlas;
  sprites!: SpriteBatch;
  shader!: Shader;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2")!;
    this.viewport = new Viewport(canvas.width, canvas.height);
  }

  setup(textures: [string, ImageBitmap][], sprites: SpriteInfo[], spriteSize: number) {
    this.spriteSize = spriteSize;
    this.atlas = new TextureAtlas(this.gl, textures);
    this.sprites = new SpriteBatch(this.gl, sprites, spriteSize, this.atlas);
    this.shader = new Shader(this.gl, vertex, fragment);

    console.log("setup", this);
  }

  update() {
    // update positions
    for (let i = 0; i < this.sprites.size; i++) {
      let x = this.sprites.x.get(i) + 1;
      if (x > this.viewport.width) {
        x = -this.spriteSize;
      }
      this.sprites.x.set(i, x);
    }
    this.sprites.x.flush(); // upload data to GPU
  }

  render() {
    this.canvas.width = this.canvas.clientWidth;
    this.canvas.height = this.canvas.clientHeight;

    this.viewport.width = this.canvas.width;
    this.viewport.height = this.canvas.height;

    this.gl.viewport(0, 0, this.viewport.width, this.viewport.height);
    this.gl.clearColor(0, 0, 0, 1);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);

    this.shader.bind();
    this.shader.uniforms.projection.set(this.viewport.projection());
    this.gl.bindVertexArray(this.sprites.vao);
    this.gl.bindTexture(this.gl.TEXTURE_2D_ARRAY, this.atlas.texture);
    this.gl.drawArraysInstanced(this.gl.TRIANGLES, 0, 6, this.sprites.size);
  }
}

// quad attributes:
// - position: vec2
// - texcoord: vec2
// per-instance attributes:
// - tid: uint8
// - x: float, world space
// - y: float, world space
// uniforms:
// - projection: mat4
// - atlas: sampler2DArray

// no index buffer, use 6 vertices. anchor is top-left
// prettier-ignore
const QUAD = new Float32Array([
  // X, Y, U, V
  -0.5, -0.5, 0, 0,
   0.5, -0.5, 1, 0,
   0.5,  0.5, 1, 1,
  -0.5, -0.5, 0, 0,
   0.5,  0.5, 1, 1,
  -0.5,  0.5, 0, 1,
]);

const vertex = `#version 300 es
precision highp float;

uniform mat4 projection;

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

layout(location = 2) in uint tid;
layout(location = 3) in float x; // world space
layout(location = 4) in float y; // world space

flat out uint v_tid;
out vec2 v_texcoord;

const float SPRITE_SIZE = 64.0;

void main() {
  v_tid = tid;
  v_texcoord = texcoord;

  mat4 model = mat4(
    SPRITE_SIZE, 0, 0, 0,
    0, SPRITE_SIZE, 0, 0,
    0, 0, 1, 0,
    x, y, 0, 1
  );

  gl_Position = projection * model * vec4(position, 0, 1);
}
`;

const fragment = `#version 300 es
precision highp float;

uniform highp sampler2DArray atlas;

flat in uint v_tid;
in vec2 v_texcoord;

out vec4 fragColor;

void main() {
  fragColor = texture(atlas, vec3(v_texcoord, v_tid));
}
`;

// keeps track of canvas size and produces a projection matrix
class Viewport {
  width: number;
  height: number;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
  }

  projection() {
    return ortho(0, this.width, this.height, 0, -1, 1);
  }
}

function ortho(
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number
) {
  // prettier-ignore
  return [
    2 / (right - left), 0, 0, 0,
    0, 2 / (top - bottom), 0, 0,
    0, 0, -2 / (far - near), 0,
    -(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1,
  ];
}

// NOTE: All textures are assumed to be the same size
class TextureAtlas {
  nameToId: Map<string, number>;
  texture: WebGLTexture;

  constructor(gl: WebGL2RenderingContext, images: [string, ImageBitmap][]) {
    // create an atlas which is a 3D texture, with each image on its own layer in the 3D texture
    this.nameToId = new Map();
    this.texture = gl.createTexture()!;

    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texture);

    // prettier-ignore
    gl.texStorage3D(
      gl.TEXTURE_2D_ARRAY, 1, gl.RGBA8,
      images[0][1].width, images[0][1].height, images.length
    );

    for (let i = 0; i < images.length; i++) {
      const [src, image] = images[i];
      // prettier-ignore
      gl.texSubImage3D(
        gl.TEXTURE_2D_ARRAY, 0,
        0, 0, i,
        image.width, image.height, 1,
        gl.RGBA, gl.UNSIGNED_BYTE,
        image
      );

      this.nameToId.set(src, i);
    }

    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
  }

  getId(name: string) {
    const id = this.nameToId.get(name);
    if (id === undefined) {
      throw new Error(`Cannot find texture ${name}`);
    }
    return id;
  }
}

class SpriteBatch {
  size: number;

  quad: Buffer<Float32Array>;
  tid: Buffer<Uint8Array>;
  x: Buffer<Float32Array>;
  y: Buffer<Float32Array>;

  vao: WebGLVertexArrayObject;

  constructor(
    gl: WebGL2RenderingContext,
    sprites: SpriteInfo[],
    spriteSize: number,
    atlas: TextureAtlas
  ) {
    const tid = new Uint8Array(sprites.map((s) => atlas.getId(s.texture)!));
    const x = new Float32Array(sprites.map((s) => s.x + spriteSize / 2));
    const y = new Float32Array(sprites.map((s) => s.y + spriteSize / 2));

    this.size = sprites.length;
    this.quad = new Buffer(gl, QUAD, "static");
    this.tid = new Buffer(gl, tid, "static");
    this.x = new Buffer(gl, x, "dynamic");
    this.y = new Buffer(gl, y, "dynamic");

    // TODO: use `VertexArray` for this
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);

    this.quad.bind();
    gl.enableVertexAttribArray(0); // vec2[0]
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 4 * 4, 0);
    gl.enableVertexAttribArray(1); // vec2[1]
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 4 * 4, 2 * 4);

    this.tid.bind();
    gl.enableVertexAttribArray(2);
    gl.vertexAttribIPointer(2, 1, gl.UNSIGNED_BYTE, 1, 0);
    gl.vertexAttribDivisor(2, 1);

    this.x.bind();
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 4, 0);
    gl.vertexAttribDivisor(3, 1);

    this.y.bind();
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 1, gl.FLOAT, false, 4, 0);
    gl.vertexAttribDivisor(4, 1);

    gl.bindVertexArray(null);
  }
}

class Buffer<Data extends TypedArray> {
  gl: WebGL2RenderingContext;
  buffer: WebGLBuffer;
  data: Data;

  constructor(gl: WebGL2RenderingContext, data: Data, mode: "static" | "dynamic") {
    this.gl = gl;
    this.buffer = gl.createBuffer()!;
    this.data = data;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, mode === "static" ? gl.STATIC_DRAW : gl.DYNAMIC_DRAW);
  }

  get(offset: number) {
    return this.data[offset];
  }

  set(offset: number, value: number) {
    this.data[offset] = value;
  }

  bind() {
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffer);
  }

  flush() {
    this.gl.bufferData(this.gl.ARRAY_BUFFER, this.data, this.gl.DYNAMIC_DRAW);
  }
}

type TypedArray =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

type UniformSetter = (data: number | number[]) => void;

type Uniform = {
  readonly name: string;
  readonly size: GLint;
  readonly type: string;
  readonly location: WebGLUniformLocation;
  readonly set: UniformSetter;
};

export class Shader {
  gl: WebGL2RenderingContext;
  program: WebGLProgram;
  uniforms: { [name: string]: Uniform };

  constructor(gl: WebGL2RenderingContext, vertex: string, fragment: string) {
    this.gl = gl;

    // compile the shader
    const program = linkProgram(
      gl,
      compileShader(gl, vertex, gl.VERTEX_SHADER),
      compileShader(gl, fragment, gl.FRAGMENT_SHADER)
    );
    this.program = program;

    // reflect uniforms
    this.uniforms = {};
    this.bind();
    for (let i = 0, len = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS); i < len; ++i) {
      const info = gl.getActiveUniform(this.program, i)!;
      const location = gl.getUniformLocation(program, info.name)!;
      this.uniforms[info.name] = {
        name: info.name,
        size: info.size,
        type: stringifyType(info.type),
        location,
        set: createSetter(gl, info.type, location),
      };
    }
    this.unbind();
  }

  bind() {
    this.gl.useProgram(this.program);
  }

  unbind() {
    this.gl.useProgram(null);
  }
}

function buildShaderErrorMessage(gl: WebGL2RenderingContext, shader: WebGLShader): string {
  const source = gl.getShaderSource(shader);
  const log = gl.getShaderInfoLog(shader);

  // if both sources are null, exit early
  if (source === null) {
    return `\n${log}\n`;
  }
  if (log === null) {
    return `Unknown error`;
  }
  // parse for line number and error
  const tokens = log
    .split("\n")
    .filter((it) => it.length > 1)
    .map((it) => it.replace(/(ERROR:\s)/g, ""))
    .map((it) => it.split(":"))
    .flat()
    .map((it) => it.trim());
  const [line, token, error] = [parseInt(tokens[1]), tokens[2], tokens[3]];
  const lines = source.split(/\n|\r/g);
  // pad first line - this always works
  // because the first line in a webgl shader MUST be a #version directive
  // and no whitespace characters may precede it
  lines[0] = `    ${lines[0]}`;

  let padding = `${lines.length}`.length;

  for (let i = 0; i < lines.length; ++i) {
    if (i === line - 1) {
      const whitespaces = lines[i].match(/\s+/);
      if (whitespaces !== null) {
        lines[i] = `${"-".repeat(whitespaces[0].length - 1)}> ${lines[i].trimStart()}`;
      }
      lines[i] = `${" ".repeat(padding - `${i + 1}`.length)}${i + 1} +${lines[i]}`;
    } else {
      lines[i] = `${" ".repeat(padding - `${i + 1}`.length)}${i + 1} |  ${lines[i]}`;
    }
  }
  lines.push(`${" ".repeat(padding)} |`);
  lines.push(`${" ".repeat(padding)} +-------> ${error}: ${token}`);
  lines.push(``);

  return lines.join("\n");
}

function compileShader(gl: WebGL2RenderingContext, source: string, type: GLenum): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (gl.getShaderParameter(shader, gl.COMPILE_STATUS) === false) {
    throw new Error("\n" + buildShaderErrorMessage(gl, shader));
  }
  return shader;
}

function linkProgram(
  gl: WebGL2RenderingContext,
  vertex: WebGLShader,
  fragment: WebGLShader
): WebGLProgram {
  const program = gl.createProgram()!;
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);

  if (gl.getProgramParameter(program, /* LINK_STATUS */ 0x8b82) === false) {
    const log = gl.getProgramInfoLog(program);
    throw new Error(`Failed to link program: ${log}`);
  }
  return program;
}

function createSetter(
  gl: WebGL2RenderingContext,
  type: number,
  location: WebGLUniformLocation
): UniformSetter {
  let typeInfo: [desc: "scalar" | "array" | "matrix", size: number, name: string];
  switch (type) {
    case 0x1400:
    case 0x1402:
    case 0x1404:
    case 0x8b56:
    case 0x8b5e:
    case 0x8b5f:
    case 0x8b60:
    case 0x8dc1:
    case 0x8dd2:
      typeInfo = ["scalar", 1, "uniform1i"];
      break;
    case 0x1401:
    case 0x1403:
    case 0x1405:
      typeInfo = ["scalar", 1, "uniform1ui"];
      break;
    case 0x8b53:
    case 0x8b57:
      typeInfo = ["array", 2, "uniform2iv"];
      break;
    case 0x8b54:
    case 0x8b58:
      typeInfo = ["array", 3, "uniform3iv"];
      break;
    case 0x8b55:
    case 0x8b59:
      typeInfo = ["array", 4, "uniform4iv"];
      break;
    case 0x1406:
      typeInfo = ["scalar", 1, "uniform1f"];
      break;
    case 0x8b50:
      typeInfo = ["array", 2, "uniform2fv"];
      break;
    case 0x8b51:
      typeInfo = ["array", 3, "uniform3fv"];
      break;
    case 0x8b52:
      typeInfo = ["array", 4, "uniform4fv"];
      break;
    case 0x8dc6:
      typeInfo = ["array", 2, "uniform2uiv"];
      break;
    case 0x8dc7:
      typeInfo = ["array", 3, "uniform3uiv"];
      break;
    case 0x8dc8:
      typeInfo = ["array", 4, "uniform4uiv"];
      break;
    case 0x8b5a:
      typeInfo = ["matrix", 2 * 2, "uniformMatrix2fv"];
      break;
    case 0x8b65:
      typeInfo = ["matrix", 2 * 3, "uniformMatrix2x3fv"];
      break;
    case 0x8b66:
      typeInfo = ["matrix", 2 * 4, "uniformMatrix2x4fv"];
      break;
    case 0x8b67:
      typeInfo = ["matrix", 3 * 2, "uniformMatrix3x2fv"];
      break;
    case 0x8b5b:
      typeInfo = ["matrix", 3 * 3, "uniformMatrix3fv"];
      break;
    case 0x8b68:
      typeInfo = ["matrix", 3 * 4, "uniformMatrix3x4fv"];
      break;
    case 0x8b69:
      typeInfo = ["matrix", 4 * 2, "uniformMatrix4x2fv"];
      break;
    case 0x8b6a:
      typeInfo = ["matrix", 4 * 3, "uniformMatrix4x3fv"];
      break;
    case 0x8b5c:
      typeInfo = ["matrix", 4 * 4, "uniformMatrix4fv"];
      break;
    default:
      throw new Error(`Unknown uniform type: ${type}`);
  }

  const setter = typeInfo[2];
  switch (typeInfo[0]) {
    case "scalar": {
      // @ts-ignore
      const setterFn = gl[setter].bind(gl);
      return function (data) {
        if (import.meta.env.DEV && typeof data !== "number") {
          throw new Error(
            `Invalid uniform data: expected ${stringifyType(type)}, got ${typeof data}`
          );
        }
        setterFn(location, data);
      };
    }
    case "array": {
      // @ts-ignore
      const setterFn = gl[setter].bind(gl);
      return function (data) {
        if (import.meta.env.DEV && (!Array.isArray(data) || data.length !== typeInfo[1])) {
          throw new Error(
            `Invalid uniform data: expected ${stringifyType(type)}, got ${typeof data}`
          );
        }
        // @ts-ignore
        setterFn(location, data);
      };
    }
    case "matrix": {
      // @ts-ignore
      const setterFn = gl[setter].bind(gl);
      return function (data) {
        if (import.meta.env.DEV && (!Array.isArray(data) || data.length !== typeInfo[1])) {
          throw new Error(
            `Invalid uniform data: expected ${stringifyType(type)}, got ${typeof data}`
          );
        }
        // @ts-ignore
        setterFn(location, false, data);
      };
    }
  }
}

function stringifyType(type: number): string {
  switch (type) {
    case 0x1400:
      return "byte";
    case 0x1402:
      return "short";
    case 0x1404:
      return "int";
    case 0x8b56:
      return "bool";
    case 0x8b5e:
      return "2d float sampler";
    case 0x8b5f:
      return "3d float sampler";
    case 0x8dc1:
      return "2d float sampler array";
    case 0x8dd2:
      return "2d unsigned int sampler";
    case 0x8b60:
      return "cube sampler";
    case 0x1401:
      return "unsigned byte";
    case 0x1403:
      return "unsigned short";
    case 0x1405:
      return "unsigned int";
    case 0x8b53:
      return "int 2-component vector";
    case 0x8b54:
      return "int 3-component vector";
    case 0x8b55:
      return "int 4-component vector";
    case 0x8b57:
      return "bool 2-component vector";
    case 0x8b58:
      return "bool 3-component vector";
    case 0x8b59:
      return "bool 4-component vector";
    case 0x1406:
      return "float";
    case 0x8b50:
      return "float 2-component vector";
    case 0x8b51:
      return "float 3-component vector";
    case 0x8b52:
      return "float 4-component vector";
    case 0x8dc6:
      return "unsigned int 2-component vector";
    case 0x8dc7:
      return "unsigned int 3-component vector";
    case 0x8dc8:
      return "unsigned int 4-component vector";
    case 0x8b5a:
      return "float 2x2 matrix";
    case 0x8b65:
      return "float 2x3 matrix";
    case 0x8b66:
      return "float 2x4 matrix";
    case 0x8b5b:
      return "float 3x3 matrix";
    case 0x8b67:
      return "float 3x2 matrix";
    case 0x8b68:
      return "float 3x4 matrix";
    case 0x8b5c:
      return "float 4x4 matrix";
    case 0x8b69:
      return "float 4x2 matrix";
    case 0x8b6a:
      return "float 4x3 matrix";
    default:
      throw new Error(`Unknown uniform type: ${type}`);
  }
}

export interface BufferDescriptor {
  /**
   * Attribute index
   *
   * e.g. for attribute `layout(location = 0) in vec2 POSITION` it would be `0`.
   */
  location: number;
  /**
   * Number of `baseType` in the compound type.
   *
   * e.g. for attribute `layout(location = 0) in vec2 POSITION`, it would be `2`, because it's a `vec2`.
   */
  arraySize: number;
  /**
   * Base type of the attribute
   *
   * e.g. for attribute `layout(location = 0) in vec2 POSITION`, it would be `GL.FLOAT`, because it's a `vec2`, which is comprised of two floats.
   */
  baseType: GLenum;
  /**
   * Whether the value should be normalized to the (0, 1) range.
   */
  normalized: boolean;
}

function sizeof(type: GLenum) {
  switch (type) {
    case 0x1400:
      return /* byte */ 1;
    case 0x1401:
      return /* unsigned byte */ 1;
    case 0x8b56:
      return /* bool */ 1;
    case 0x1402:
      return /* short */ 2;
    case 0x1403:
      return /* unsigned short */ 2;
    case 0x1404:
      return /* int */ 4;
    case 0x1405:
      return /* unsigned int */ 4;
    case 0x1406:
      return /* float */ 4;
    default:
      throw new Error(`Unknown type: ${type}`);
  }
}

// TODO: make this work with:
//       - multiple buffers
//       - interleaved and separate buffers
//       - instanced rendering
export class VertexArray {
  gl: WebGL2RenderingContext;
  vao: WebGLVertexArrayObject;

  constructor(
    gl: WebGL2RenderingContext,
    buffers: { buffer: Buffer<TypedArray>; descriptors: BufferDescriptor[] }[]
  ) {
    this.gl = gl;
    this.vao = gl.createVertexArray()!;

    if (import.meta.env.DEV && buffers.length === 0) {
      throw new Error("No buffers provided");
    }

    gl.bindVertexArray(this.vao);

    gl.bindVertexArray(null);
  }

  bind() {
    this.gl.bindVertexArray(this.vao);
  }

  unbind() {
    this.gl.bindVertexArray(null);
  }
}

