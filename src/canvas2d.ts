// Rendering sprites using Canvas2D

import { SpriteInfo } from "./common";

export class Context {
  private ctx: CanvasRenderingContext2D;
  private canvas: HTMLCanvasElement;
  private textures: Map<string, ImageBitmap> = new Map();
  private sprites: Sprite[] = [];
  private spriteSize: number = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  setup(textures: [string, ImageBitmap][], sprites: SpriteInfo[], spriteSize: number) {
    for (const [src, image] of textures) {
      this.textures.set(src, image);
    }

    for (const sprite of sprites) {
      const spriteTexture = this.textures.get(sprite.texture);
      if (!spriteTexture) {
        throw new Error(`Cannot find texture ${sprite.texture}`);
      }
      this.sprites.push(new Sprite(spriteTexture, sprite.x, sprite.y));
    }

    this.spriteSize = spriteSize;
  }

  update() {
    for (let i = 0; i < this.sprites.length; i++) {
      const sprite = this.sprites[i];
      sprite.x += 1;
      if (sprite.x > this.canvas.width) {
        sprite.x = -this.spriteSize;
      }
    }
  }

  render() {
    this.canvas.width = this.canvas.clientWidth;
    this.canvas.height = this.canvas.clientHeight;

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    for (const sprite of this.sprites) {
      sprite.draw(this.ctx, this.spriteSize);
    }
  }

  destroy() {
    this.textures.clear();
    this.sprites.length = 0;
  }
}

class Sprite {
  constructor(public texture: ImageBitmap, public x: number, public y: number) {}

  draw(ctx: CanvasRenderingContext2D, size: number) {
    ctx.drawImage(this.texture, this.x, this.y, size, size);
  }
}

