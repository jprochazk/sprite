import "./style.css";

import { SpriteInfo, ref, debounce, debug, loadImage, type Context } from "./common";
import { Context as Context_Canvas2D } from "./canvas2d";
import { Context as Context_WebGL2 } from "./webgl2";

const SPRITE_SIZE = 64;
const textures: [string, ImageBitmap][] = await Promise.all([
  loadImage("assets/dankeG.png"),
  loadImage("assets/copeG.png"),
]);

function spriteAt(rows: number, cols: number, i: number): SpriteInfo {
  // wrap around back to start if `i` exceeds rows * cols
  i = i % (rows * cols);
  const x = i % cols;
  const y = Math.floor(i / cols);
  const texture = textures[i % textures.length][0];

  return { texture, x: x * SPRITE_SIZE, y: y * SPRITE_SIZE };
}

function spriteGrid(rows: number, cols: number) {
  const sprites: SpriteInfo[] = [];
  for (let i = 0; i < rows * cols; i++) {
    sprites.push(spriteAt(rows, cols, i));
  }
  return debug(sprites);
}

const dropdown = document.getElementById("renderer") as HTMLSelectElement;
let dropdownValue = ref(dropdown.value);
dropdown.addEventListener("change", () => {
  dropdownValue.set(dropdown.value);
});

const slider = document.getElementById("count") as HTMLInputElement;
let sliderValue = ref(slider.valueAsNumber);
slider.addEventListener("input", () => {
  sliderValue.set(slider.valueAsNumber);
  const label = slider.labels?.[0];
  if (label) {
    label.textContent = slider.value;
  }
});

const canvasContainer = document.getElementById("canvas-container") as HTMLDivElement;

let stop = () => {};

function start(renderer: typeof Context, count: number) {
  stop();

  const canvas = document.createElement("canvas");
  canvas.width = canvasContainer.clientWidth;
  canvas.height = canvasContainer.clientHeight;
  canvasContainer.innerHTML = "";
  canvasContainer.appendChild(canvas);

  // split count by aspect ratio of canvas
  const aspectRatio = canvas.width / canvas.height;
  const cols = Math.round(Math.sqrt(count * aspectRatio));
  const rows = Math.ceil(count / cols);

  const ctx = new renderer(canvas);
  ctx.setup(textures, spriteGrid(rows, cols), SPRITE_SIZE);

  stop = () => {
    cancelAnimationFrame(raf);
    canvas.remove();
    ctx.destroy();
  };

  let raf = 0;
  function loop() {
    ctx.update();
    ctx.render();
    raf = requestAnimationFrame(loop);
  }

  loop();
}

function run() {
  if (dropdownValue.current === "canvas2d") {
    start(Context_Canvas2D, sliderValue.current);
  } else if (dropdownValue.current === "webgl2") {
    start(Context_WebGL2, sliderValue.current);
  }
}

const runDebounced = debounce(() => run(), 100);

sliderValue.subscribe(runDebounced);
dropdownValue.subscribe(runDebounced);
runDebounced();

