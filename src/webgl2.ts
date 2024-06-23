import { SpriteInfo } from "./common";

export class Context {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  private viewport: Viewport;
  private spriteSize: number = 1;
  private atlas!: TextureAtlas;
  private sprites!: SpriteBatch;
  private shader!: Shader;

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
      // note: have to account for sprites being renderered from the center
      // in canvas2d version they are rendered from top left corner
      if (x - this.spriteSize / 2 > this.viewport.width) {
        x = -this.spriteSize / 2;
      }
      this.sprites.x.set(i, x);
    }
  }

  render() {
    this.canvas.width = this.canvas.clientWidth;
    this.canvas.height = this.canvas.clientHeight;

    this.viewport.width = this.canvas.width;
    this.viewport.height = this.canvas.height;

    this.gl.viewport(0, 0, this.viewport.width, this.viewport.height);
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);

    this.shader.bind();
    this.shader.uniform("projection").set(this.viewport.projection());
    this.atlas.bind(0);
    this.sprites.draw();
  }

  destroy() {
    this.atlas.destroy();
    this.shader.destroy();
    this.sprites.destroy();
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
  private gl: WebGL2RenderingContext;
  private nameToId: Map<string, number>;
  private texture: WebGLTexture;

  // create an atlas which is a 3D texture, with each image on its own layer in the 3D texture
  constructor(gl: WebGL2RenderingContext, images: [string, ImageBitmap][]) {
    this.gl = gl;
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

  private index: number = -1;
  bind(index: number) {
    this.index = index;
    this.gl.activeTexture(this.gl.TEXTURE0 + index);
    this.gl.bindTexture(this.gl.TEXTURE_2D_ARRAY, this.texture);
  }

  unbind() {
    this.gl.activeTexture(this.gl.TEXTURE0 + this.index);
    this.gl.bindTexture(this.gl.TEXTURE_2D_ARRAY, null);
  }

  destroy() {
    this.gl.deleteTexture(this.texture);
    this.nameToId.clear();
    this.index = -1;
  }
}

class SpriteBatch {
  gl: WebGL2RenderingContext;

  size: number;

  quad: Buffer<Float32>;
  tid: Buffer<Uint8>;
  x: Buffer<Float32>;
  y: Buffer<Float32>;
  attribs: AttributeSet;

  constructor(
    gl: WebGL2RenderingContext,
    sprites: SpriteInfo[],
    spriteSize: number,
    atlas: TextureAtlas
  ) {
    this.gl = gl;

    const tid = new Uint8Array(sprites.map((s) => atlas.getId(s.texture)!));
    const x = new Float32Array(sprites.map((s) => s.x + spriteSize / 2));
    const y = new Float32Array(sprites.map((s) => s.y + spriteSize / 2));

    this.size = sprites.length;
    this.quad = new Buffer(gl, QUAD, "static");
    this.tid = new Buffer(gl, tid, "static");
    this.x = new Buffer(gl, x, "dynamic");
    this.y = new Buffer(gl, y, "dynamic");

    this.attribs = new AttributeSet(gl, [
      {
        buffer: this.quad,
        attributes: [vec2Attrib(0), vec2Attrib(1)],
      },
      {
        buffer: this.tid,
        attributes: [ubyteAttrib(2, 1)],
      },
      {
        buffer: this.x,
        attributes: [floatAttrib(3, 1)],
      },
      {
        buffer: this.y,
        attributes: [floatAttrib(4, 1)],
      },
    ]);
  }

  draw() {
    this.x.flush();
    this.y.flush();

    this.attribs.bind();
    this.gl.drawArraysInstanced(this.gl.TRIANGLES, 0, 6, this.size);
  }

  destroy() {
    this.quad.destroy();
    this.tid.destroy();
    this.x.destroy();
    this.y.destroy();
    this.attribs.destroy();
  }
}

class Buffer<Data extends BufferType> {
  private gl: WebGL2RenderingContext;
  private buffer: WebGLBuffer;
  private data: Data;

  constructor(gl: WebGL2RenderingContext, data: Data, mode: "static" | "dynamic") {
    this.gl = gl;
    this.buffer = gl.createBuffer()!;
    this.data = data;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, mode === "static" ? gl.STATIC_DRAW : gl.DYNAMIC_DRAW);
  }

  private dirty = false;

  get(offset: number) {
    return this.data[offset];
  }

  set(offset: number, value: number) {
    this.data[offset] = value;
    this.dirty = true;
  }

  bind() {
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffer);
  }

  unbind() {
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
  }

  flush() {
    if (!this.dirty) {
      return;
    }

    this.bind();
    this.gl.bufferData(this.gl.ARRAY_BUFFER, this.data, this.gl.DYNAMIC_DRAW);
  }

  destroy() {
    this.gl.deleteBuffer(this.buffer);
    this.data = null!;
  }
}

type Int8 = Int8Array;
type Uint8 = Uint8Array;
type Int16 = Int16Array;
type Uint16 = Uint16Array;
type Int32 = Int32Array;
type Uint32 = Uint32Array;
type Float32 = Float32Array;
type Float64 = Float64Array;

type BufferType = Int8 | Uint8 | Int16 | Uint16 | Int32 | Uint32 | Float32 | Float64;

type UniformSetter = (data: number | number[]) => void;

type Uniform = {
  readonly name: string;
  readonly size: GLint;
  readonly type: string;
  readonly location: WebGLUniformLocation;
  readonly set: UniformSetter;
};

export class Shader {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private uniforms: { [name: string]: Uniform };

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

  uniform(name: string): Readonly<Uniform> {
    return this.uniforms[name];
  }

  bind() {
    this.gl.useProgram(this.program);
  }

  unbind() {
    this.gl.useProgram(null);
  }

  destroy() {
    this.gl.deleteProgram(this.program);
    this.program = null!;
    this.uniforms = null!;
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

interface AttributeArrayDescriptor {
  /** Buffer to bind the given attributes to */
  buffer: Buffer<BufferType>;
  attributes: AttributeDescriptor[];
}

interface AttributeDescriptor {
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
   *
   * Ignored for integer types.
   */
  normalized: boolean;

  divisor: number;
}

function floatAttrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 1, baseType: FloatT, normalized: false, divisor };
}

function vec2Attrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 2, baseType: FloatT, normalized: false, divisor };
}

/* function vec3Attrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 3, baseType: FloatT, normalized: false, divisor };
}

function vec4Attrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 4, baseType: FloatT, normalized: false, divisor };
}

function intAttrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 1, baseType: IntT, normalized: false, divisor };
}

function ivec2Attrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 2, baseType: IntT, normalized: false, divisor };
}

function ivec3Attrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 3, baseType: IntT, normalized: false, divisor };
}

function ivec4Attrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 4, baseType: IntT, normalized: false, divisor };
} */

function ubyteAttrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 1, baseType: UnsignedByteT, normalized: false, divisor };
}

/* function uintAttrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 1, baseType: UnsignedIntT, normalized: false, divisor };
}

function uvec2Attrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 2, baseType: UnsignedIntT, normalized: false, divisor };
}

function uvec3Attrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 3, baseType: UnsignedIntT, normalized: false, divisor };
}

function uvec4Attrib(location: number, divisor: number = 0): AttributeDescriptor {
  return { location, arraySize: 4, baseType: UnsignedIntT, normalized: false, divisor };
} */

const ByteT = 0x1400;
const UnsignedByteT = 0x1401;
const ShortT = 0x1402;
const UnsignedShortT = 0x1403;
const IntT = 0x1404;
const UnsignedIntT = 0x1405;
const FloatT = 0x1406;

function attribSizeOf(type: GLenum) {
  switch (type) {
    case ByteT:
    case UnsignedByteT:
      return 1;
    case ShortT:
    case UnsignedShortT:
      return 2;
    case IntT:
    case UnsignedIntT:
    case FloatT:
      return 4;
    default:
      throw new Error(`Unknown base type: ${type}`);
  }
}

function attribIsInt(type: GLenum) {
  switch (type) {
    case ByteT:
    case UnsignedByteT:
    case ShortT:
    case UnsignedShortT:
    case IntT:
    case UnsignedIntT:
      return true;
    default:
      return false;
  }
}

export class AttributeSet {
  gl: WebGL2RenderingContext;
  vao: WebGLVertexArrayObject;

  constructor(gl: WebGL2RenderingContext, descriptors: AttributeArrayDescriptor[]) {
    this.gl = gl;
    this.vao = gl.createVertexArray()!;

    if (descriptors.length === 0) {
      throw new Error("No descriptors provided");
    }

    gl.bindVertexArray(this.vao);

    for (const descriptor of descriptors) {
      let stride = 0;
      for (const attribute of descriptor.attributes) {
        stride += attribSizeOf(attribute.baseType) * attribute.arraySize;
      }

      descriptor.buffer.bind();

      let offset = 0;
      for (const attribute of descriptor.attributes) {
        gl.enableVertexAttribArray(attribute.location);
        if (attribIsInt(attribute.baseType)) {
          gl.vertexAttribIPointer(
            attribute.location,
            attribute.arraySize,
            attribute.baseType,
            stride,
            offset
          );
        } else {
          gl.vertexAttribPointer(
            attribute.location,
            attribute.arraySize,
            attribute.baseType,
            attribute.normalized,
            stride,
            offset
          );
        }

        if (attribute.divisor !== 0) {
          gl.vertexAttribDivisor(attribute.location, attribute.divisor);
        }

        offset += attribSizeOf(attribute.baseType) * attribute.arraySize;
      }

      descriptor.buffer.unbind();
    }

    gl.bindVertexArray(null);
  }

  bind() {
    this.gl.bindVertexArray(this.vao);
  }

  unbind() {
    this.gl.bindVertexArray(null);
  }

  destroy() {
    this.gl.deleteVertexArray(this.vao);
  }
}

