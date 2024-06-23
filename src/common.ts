export declare class Context {
  constructor(canvas: HTMLCanvasElement);

  setup(textures: [string, ImageBitmap][], sprites: SpriteInfo[], spriteSize: number): void;
  update(): void;
  render(): void;
}

export type SpriteInfo = {
  texture: string;
  x: number;
  y: number;
};

export function ref<T>(value: T) {
  const callbacks = new Set<(value: T) => void>();
  return {
    current: value,

    set(value: T) {
      this.current = value;
      for (const callback of callbacks) {
        callback(value);
      }
    },

    subscribe(callback: (value: T) => void): () => void {
      callbacks.add(callback);
      return () => callbacks.delete(callback);
    },
  };
}

export function debounce(callback: () => void, delay: number) {
  let timeout: number | undefined;
  return () => {
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(() => {
      callback();
      timeout = undefined;
    }, delay);
  };
}

export function debug<T>(value: T) {
  console.log(value);
  return value;
}

export async function loadImage(src: string): Promise<[string, ImageBitmap]> {
  const image = new Image();
  image.src = src;
  await image.decode();
  return [src, await createImageBitmap(image)];
}

