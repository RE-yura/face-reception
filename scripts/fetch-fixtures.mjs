// Downloads public-domain NASA portraits from Wikimedia Commons and crops them to 960×720 (4:3)
// so they can be used both as unit-test inputs and as Chrome's fake camera feed.
import jpeg from 'jpeg-js';
import { writeFileSync } from 'node:fs';

const UA = 'face-reception-tests/1.0 (https://github.com/RE-yura/face-reception)';
const BASE = 'https://upload.wikimedia.org/wikipedia/commons/thumb';
const SOURCES = {
  'meir-a': `${BASE}/c/c7/Official_portrait_of_NASA_astronaut_Jessica_Meir_wearing_a_spacesuit_%28jsc2025e078605_alt%29.jpg/960px-Official_portrait_of_NASA_astronaut_Jessica_Meir_wearing_a_spacesuit_%28jsc2025e078605_alt%29.jpg`,
  'meir-b': `${BASE}/1/1d/Jessica_Meir_official_portrait_in_an_EMU_%28B%26W%29.jpg/960px-Jessica_Meir_official_portrait_in_an_EMU_%28B%26W%29.jpg`,
  kim: `${BASE}/e/e9/Jsc2024e052605_alt_%28Aug._6%2C_2024%29_---_Official_portrait_of_NASA_astronaut_Jonny_Kim.jpg/960px-Jsc2024e052605_alt_%28Aug._6%2C_2024%29_---_Official_portrait_of_NASA_astronaut_Jonny_Kim.jpg`,
};
const WIDTH = 960;
const HEIGHT = 720;

for (const [name, url] of Object.entries(SOURCES)) {
  const res = await fetch(url, { headers: { 'User-Agent': UA } });
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
  const src = jpeg.decode(new Uint8Array(await res.arrayBuffer()), { useTArray: true, formatAsRGBA: true });
  if (src.width !== WIDTH || src.height < HEIGHT) throw new Error(`${name}: unexpected size ${src.width}x${src.height}`);
  const data = src.data.subarray(0, WIDTH * HEIGHT * 4); // top 720 rows: the face is in the upper part of each portrait
  const out = jpeg.encode({ width: WIDTH, height: HEIGHT, data }, 92);
  writeFileSync(new URL(`../tests/fixtures/faces/${name}.jpg`, import.meta.url), out.data);
  console.log(`${name}.jpg ${WIDTH}x${HEIGHT}`);
}
