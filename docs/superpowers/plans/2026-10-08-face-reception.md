# face-reception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** attendance_manager（PyQt5 + PyTorch）を、ブラウザ内の ONNX Runtime Web だけで顔検出・照合する静的 Web アプリ「face-reception」として作り直し、GitHub Pages で公開する。

**Architecture:** Vite + TypeScript の静的サイト。カメラのコマを Web Worker に渡し、YuNet（顔検出）→ 5点相似変換で 112×112 に揃える → SFace（128次元）を onnxruntime-web の wasm 版で実行する。メインスレッドは照合（コサイン類似度 ≥ 0.363）、IndexedDB への保存、画面の描画だけを担う。推論・照合・状態遷移は DOM に依存しない純粋なモジュールに分け、Vitest で検証する。ブラウザでの結合は、Chrome の仮想カメラに顔写真を流す Playwright の E2E で検証する。

**Tech Stack:** Vite 8.3.4, TypeScript 6.0.3, onnxruntime-web 1.30.0（`onnxruntime-web/wasm`）, idb 8.0.4, Vitest 5.0.3, fake-indexeddb 6.2.5, jpeg-js 0.4.4, Playwright 1.64.0（system Chrome）, GitHub Actions + GitHub Pages

**Spec:** `docs/superpowers/specs/2026-10-08-face-reception-design.md`

**この計画のコードについて:** 全ファイルを、計画を書く段階で scratchpad 上に実際に組み、`tsc`、Vitest（59件）、`vite build`、Playwright E2E（7件）、iOS シミュレーター（iOS 26.5 Safari）でのモデル読み込みまで通してある。各ステップのコードはそのまま写せば動く。違う結果が出たら、コードを推測で直さずに原因を調べること。

## Global Constraints

- リポジトリ: `~/ghq/github.com/RE-yura/face-reception`（作成済み、`main` に spec のコミットが1つある）。Task 1〜9 はブランチ `feature/initial-app` で作業し、Task 10 で `main` に取り込む。
- Node 24、npm。依存はすべて `--save-exact` で固定する。
- ORT は `onnxruntime-web@1.30.0` を `onnxruntime-web/wasm` からだけ import する。既定のバンドルや WebGPU 版は使わない（iOS Safari 26 でメモリが膨らんでタブが落ちる: onnxruntime #26827 / WebKit 304810）。
- ORT の設定は `ort.env.wasm.numThreads = 1`。`.wasm` は `?url` import で自分のビルドから配信し、CDN は使わない。セッションには `logSeverityLevel: 3` を付ける。
- モデルは `public/models/` に普通のファイルとして置く（Git LFS 不可。GitHub Pages が LFS を配信できないため）:
  - `face_detection_yunet_2026may.onnx`: 229,738 bytes、sha256 `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`
  - `face_recognition_sface_2021dec.onnx`: 38,696,353 bytes、sha256 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`
- Vite の `base` は `/face-reception/`。
- しきい値などの定数は `src/config.ts` にだけ置く（`MATCH_THRESHOLD = 0.363` ほか）。
- 画面の文言はすべて日本語で、各タスクに書いた文字列をそのまま使う（E2E が文字列で照合する）。
- 顔の画像・特徴量は端末の外に出さない。アプリが行う通信は、同じオリジンのアプリ本体とモデルの取得だけ。
- TypeScript は `tsconfig.json` の `erasableSyntaxOnly`（パラメータプロパティ・enum・namespace 禁止）と `verbatimModuleSyntax`（型は `import type`）に従い、相対 import には `.ts` 拡張子を付ける。
- 対応ブラウザ: iOS 17 以降の Safari、最新のデスクトップ Chrome / Safari / Firefox。
- コミットメッセージの末尾には必ず次の2行を付ける:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01PADPRtudUhYxccwepxBWdj
  ```

## Review Focus

spec では触れていないが、使う人が出会いやすい入力や状況。各項目について、受け持つタスクにそれを押さえるテストを入れてある。

1. **iPhone のインカメラは縦長のコマ（例: 480×640）を返す。** 縦長の画面でも、左右反転したプレビューに顔の枠がずれずに重なるべき。→ Task 6 `geometry.test.ts`「handles the portrait frames an iPhone front camera delivers」
2. **カメラに複数人が映る。** 受付ではカメラに一番近い（枠が一番大きい）人だけを照合し、登録ではそのコマを数えないべき。→ Task 3 `largestFace` のテスト、Task 6 `EnrollSession` の「several faces」テスト
3. **名前に `<b>` のような HTML に見える文字列を入れる。** 一覧にも受付のメッセージにも、そのまま文字として出るべき。→ Task 9 E2E「shows names as plain text, never as HTML」
4. **画面ロックやアプリ切り替えでページが裏に回り、また戻ってくる。** 裏ではカメラを手放し、戻ったら自動で検出が再開するべき。→ Task 9 E2E「releases the camera while the page is in the background and resumes when it returns」
5. **同じ名前でもう一度登録する。** 別人として増えず、同じ人に写真が追加されるべき（一覧の枚数が 5 → 10）。→ Task 4 `store.test.ts`「appends embeddings when the same name…」、Task 9 E2E「adds photos to an existing name and deletes people」

## File Structure

| パス | 役割 | タスク |
|---|---|---|
| `package.json`, `tsconfig.json`, `vite.config.ts`, `vitest.config.ts`, `.gitignore` | ビルドとテストの設定 | 1 |
| `scripts/fetch-models.sh`, `public/models/*` | モデルとライセンスの取得・配置 | 1 |
| `src/config.ts` | 定数（しきい値、間隔、モデルのファイル名とサイズ） | 1 |
| `src/vision/types.ts` | `Point`, `RgbaImage`, `Face` | 2 |
| `src/vision/align.ts` | 相似変換の推定、アフィン変換での切り出し、`alignFace` | 2 |
| `src/vision/detector.ts` | YuNet の入力作り、出力の復元、NMS、`largestFace` | 3 |
| `src/match.ts` | コサイン類似度での照合 | 4 |
| `src/store.ts` | IndexedDB（とメモリ版）の登録者ストア | 4 |
| `src/vision/embedder.ts` | SFace の入力作り、L2 正規化 | 5 |
| `src/vision/pipeline.ts` | ORT セッションを持ち、検出と特徴量抽出を実行（Worker と Node テストで共用） | 5 |
| `scripts/fetch-fixtures.mjs`, `scripts/opencv_reference.py`, `tests/fixtures/*` | テスト用の顔写真と OpenCV の基準値 | 5 |
| `src/worker-protocol.ts` | Worker とのメッセージ型 | 6 |
| `src/reception-state.ts` | 受付タブの表示状態 | 6 |
| `src/enroll-session.ts` | 登録時の撮影の進行 | 6 |
| `src/errors.ts` | カメラ・モデル読み込みのエラー分類と文言 | 6 |
| `src/ui/geometry.ts` | フレーム座標 → 画面座標（cover 表示＋左右反転） | 6 |
| `index.html`, `src/layout.css`, `src/theme.css` | 画面の構造、レイアウト、見た目 | 7 |
| `src/worker.ts`, `src/vision-client.ts` | 推論 Worker と、メイン側の窓口 | 8 |
| `src/camera.ts`, `src/ui/dom.ts`, `src/ui/stage.ts`, `src/ui/start-screen.ts` | カメラ、プレビューと枠の描画、起動画面 | 8 |
| `playwright.config.ts`, `tests/e2e/helpers.ts`, `tests/e2e/startup.spec.ts` | E2E の土台と起動まわりのテスト | 8 |
| `src/ui/reception-panel.ts`, `src/ui/enroll-panel.ts`, `src/ui/thumbnail.ts` | 受付と登録の画面 | 9 |
| `src/main.ts` | 起動、タブ切り替え、裏に回ったときの処理（Task 7 で仮置き → 8 → 9 で完成） | 7, 8, 9 |
| `tests/e2e/reception.spec.ts` | 登録・受付・削除の E2E | 9 |
| `README.md`, `.github/workflows/deploy.yml`, `scripts/ios-sim-check.mjs` | 説明、配信、iOS シミュレーターでの確認 | 10 |

---

### Task 1: プロジェクトの土台とモデル

**Files:**
- Create: `package.json`, `tsconfig.json`, `vite.config.ts`, `vitest.config.ts`, `.gitignore`
- Create: `scripts/fetch-models.sh`, `src/config.ts`, `tests/helpers/models.ts`
- Create（スクリプトで取得）: `public/models/face_detection_yunet_2026may.onnx`, `public/models/face_recognition_sface_2021dec.onnx`, `public/models/LICENSE-yunet.txt`, `public/models/LICENSE-sface.txt`
- Test: `tests/unit/models.test.ts`

**Interfaces:**
- Consumes: なし
- Produces:
  - `src/config.ts`: `DETECT_INPUT_LONG_SIDE`, `DETECT_SCORE_THRESHOLD`, `NMS_IOU_THRESHOLD`, `MATCH_THRESHOLD`, `RECOGNIZE_INTERVAL_MS`, `ENROLL_SHOTS`, `ENROLL_INTERVAL_MS`, `ENROLL_TIMEOUT_MS`（すべて `number`）、`MODELS: { yunet: { file: string; bytes: number }; sface: { file: string; bytes: number } }`
  - `tests/helpers/models.ts`: `readModel(file: string): Uint8Array`
  - モデルの入出力: YuNet は入力 `input`（1×3×H×W、BGR 0〜255）、出力 `cls_{8,16,32}`（1×N×1）, `obj_*`（1×N×1）, `bbox_*`（1×N×4）, `kps_*`（1×N×10）。SFace は入力 `data`（1×3×112×112、RGB 0〜255）、出力 `fc1`（1×128）

- [ ] **Step 1: ブランチを切る**

```bash
cd ~/ghq/github.com/RE-yura/face-reception
git switch -c feature/initial-app
```

- [ ] **Step 2: `package.json` を書いて依存を入れる**

`package.json`:

```json
{
  "name": "face-reception",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "dev:phone": "vite --host --mode phone",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:e2e": "playwright test"
  }
}
```

```bash
npm install --save-exact onnxruntime-web@1.30.0 idb@8.0.4
npm install --save-exact -D typescript@6.0.3 vite@8.3.4 vitest@5.0.3 @types/node@24.19.1 @vitejs/plugin-basic-ssl@2.3.0 @playwright/test@1.64.0 fake-indexeddb@6.2.5 jpeg-js@0.4.4
```

Expected: `package.json` に `dependencies` と `devDependencies` が追加され、`package-lock.json` ができる。

- [ ] **Step 3: 設定ファイルを書く**

`tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "es2023",
    "module": "esnext",
    "lib": ["ES2023", "DOM"],
    "types": ["vite/client", "node"],
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "verbatimModuleSyntax": true,
    "moduleDetection": "force",
    "noEmit": true,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "erasableSyntaxOnly": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src", "tests", "vite.config.ts", "vitest.config.ts", "playwright.config.ts"]
}
```

`vite.config.ts`:

```ts
import basicSsl from '@vitejs/plugin-basic-ssl';
import { defineConfig } from 'vite';

export default defineConfig(({ mode }) => ({
  base: '/face-reception/',
  // `npm run dev:phone` serves over HTTPS on the LAN so an iPhone may use the camera.
  plugins: mode === 'phone' ? [basicSsl()] : [],
  // Pre-bundling breaks the relative URL onnxruntime-web uses to find its .wasm file.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
}));
```

`vitest.config.ts`:

```ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/unit/**/*.test.ts'],
    testTimeout: 60_000,
    hookTimeout: 60_000,
  },
});
```

`.gitignore`:

```gitignore
node_modules/
dist/
test-results/
playwright-report/
.superpowers/
```

- [ ] **Step 4: 失敗するテストを書く**

`tests/helpers/models.ts`:

```ts
import { readFileSync } from 'node:fs';

export function readModel(file: string): Uint8Array {
  return readFileSync(new URL(`../../public/models/${file}`, import.meta.url));
}
```

`tests/unit/models.test.ts`:

```ts
import * as ort from 'onnxruntime-web/wasm';
import { describe, expect, it } from 'vitest';
import { MODELS } from '../../src/config.ts';
import { readModel } from '../helpers/models.ts';

ort.env.wasm.numThreads = 1;
const options: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'], logSeverityLevel: 3 };

describe('model files', () => {
  it('have the sizes the loading progress bar expects', () => {
    expect(readModel(MODELS.yunet.file).byteLength).toBe(MODELS.yunet.bytes);
    expect(readModel(MODELS.sface.file).byteLength).toBe(MODELS.sface.bytes);
  });

  it('YuNet takes a 320×256 BGR image and returns cls/obj/bbox/kps per stride', async () => {
    const session = await ort.InferenceSession.create(readModel(MODELS.yunet.file), options);
    expect(session.inputNames).toEqual(['input']);
    const out = await session.run({ input: new ort.Tensor('float32', new Float32Array(3 * 256 * 320), [1, 3, 256, 320]) });
    for (const [stride, cells] of [[8, 1280], [16, 320], [32, 80]] as const) {
      expect(out[`cls_${stride}`].dims).toEqual([1, cells, 1]);
      expect(out[`obj_${stride}`].dims).toEqual([1, cells, 1]);
      expect(out[`bbox_${stride}`].dims).toEqual([1, cells, 4]);
      expect(out[`kps_${stride}`].dims).toEqual([1, cells, 10]);
    }
  });

  it('SFace takes a 112×112 RGB face and returns a 128-d feature', async () => {
    const session = await ort.InferenceSession.create(readModel(MODELS.sface.file), options);
    expect(session.inputNames).toEqual(['data']);
    const out = await session.run({ data: new ort.Tensor('float32', new Float32Array(3 * 112 * 112).fill(128), [1, 3, 112, 112]) });
    expect(out.fc1.dims).toEqual([1, 128]);
  });
});
```

- [ ] **Step 5: テストが失敗することを確かめる**

Run: `npm test`
Expected: FAIL。`src/config.ts` が見つからないエラー（`Failed to load url ../../src/config.ts` など）。

- [ ] **Step 6: 定数とモデル取得スクリプトを書き、モデルを取る**

`src/config.ts`:

```ts
export const DETECT_INPUT_LONG_SIDE = 320;
export const DETECT_SCORE_THRESHOLD = 0.8;
export const NMS_IOU_THRESHOLD = 0.3;
export const MATCH_THRESHOLD = 0.363;
export const RECOGNIZE_INTERVAL_MS = 500;
export const ENROLL_SHOTS = 5;
export const ENROLL_INTERVAL_MS = 400;
export const ENROLL_TIMEOUT_MS = 10_000;

export const MODELS = {
  yunet: { file: 'face_detection_yunet_2026may.onnx', bytes: 229_738 },
  sface: { file: 'face_recognition_sface_2021dec.onnx', bytes: 38_696_353 },
} as const;
```

`scripts/fetch-models.sh`:

```bash
#!/usr/bin/env bash
# Downloads the ONNX models and their licenses from OpenCV Zoo into public/models/.
# OpenCV Zoo stores the models in Git LFS, which GitHub Pages cannot serve, so we commit plain copies.
set -euo pipefail
cd "$(dirname "$0")/.."

ZOO="https://github.com/opencv/opencv_zoo/raw/26cc381e4d2594bb9f47a26eb8fd96c94a13660d/models"
OUT="public/models"
mkdir -p "$OUT"

fetch() { # <path under models/> <output file> [sha256]
  curl -fsSL -o "$OUT/$2" "$ZOO/$1"
  if [ -n "${3:-}" ]; then echo "$3  $OUT/$2" | shasum -a 256 -c -; fi
}

fetch face_detection_yunet/face_detection_yunet_2026may.onnx face_detection_yunet_2026may.onnx ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0
fetch face_recognition_sface/face_recognition_sface_2021dec.onnx face_recognition_sface_2021dec.onnx 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
fetch face_detection_yunet/LICENSE LICENSE-yunet.txt
fetch face_recognition_sface/LICENSE LICENSE-sface.txt
```

```bash
chmod +x scripts/fetch-models.sh
scripts/fetch-models.sh
```

Expected: `public/models/face_detection_yunet_2026may.onnx: OK` と `public/models/face_recognition_sface_2021dec.onnx: OK` が出て、`public/models/` に4ファイルができる。

- [ ] **Step 7: テストと型チェックが通ることを確かめる**

Run: `npm test && npx tsc`
Expected: `Tests  3 passed (3)`。`tsc` は何も出さずに終わる。

- [ ] **Step 8: コミットする**

モデルが LFS ではなく普通のファイルとして入ることを確かめてからコミットする。

```bash
git check-attr filter -- public/models/*.onnx   # 2行とも "filter: unspecified" であること
git add package.json package-lock.json tsconfig.json vite.config.ts vitest.config.ts .gitignore scripts/fetch-models.sh src/config.ts tests public/models
git commit -F - <<'EOF'
Set up Vite + TypeScript project with ONNX models

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PADPRtudUhYxccwepxBWdj
EOF
```

---

### Task 2: 顔を揃える（相似変換と切り出し）

**Files:**
- Create: `src/vision/types.ts`, `src/vision/align.ts`
- Test: `tests/unit/align.test.ts`

**Interfaces:**
- Consumes: なし
- Produces:
  - `src/vision/types.ts`: `type Point = [number, number]`、`interface RgbaImage { width: number; height: number; data: Uint8ClampedArray | Uint8Array }`（DOM の `ImageData` と構造的に互換）、`interface Face { box: { x: number; y: number; width: number; height: number }; landmarks: Point[]; score: number }`（landmarks は 右目, 左目, 鼻, 右口角, 左口角 の順、画像座標）
  - `src/vision/align.ts`: `ALIGNED_SIZE = 112`、`ARCFACE_TEMPLATE: Point[]`、`type Affine = [number, number, number, number, number, number]`（OpenCV の 2×3 行列の並び）、`estimateSimilarity(src: Point[], dst: Point[]): Affine`、`invertAffine(m: Affine): Affine`、`warpAffine(src: RgbaImage, m: Affine, width: number, height: number): RgbaImage`（m は src → dst、範囲外は不透明な黒、`data` は `Uint8ClampedArray`）、`alignFace(src: RgbaImage, landmarks: Point[]): RgbaImage`（112×112）

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/align.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { ALIGNED_SIZE, ARCFACE_TEMPLATE, alignFace, estimateSimilarity, invertAffine, warpAffine, type Affine } from '../../src/vision/align.ts';
import type { Point, RgbaImage } from '../../src/vision/types.ts';

const apply = (m: Affine, [x, y]: Point): Point => [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]];

function similarity(scale: number, angle: number, tx: number, ty: number): Affine {
  const a = scale * Math.cos(angle);
  const b = scale * Math.sin(angle);
  return [a, -b, tx, b, a, ty];
}

/** R = 10·x, G = 10·y, B = 0 */
function gradient(width: number, height: number): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      data[o] = x * 10;
      data[o + 1] = y * 10;
      data[o + 3] = 255;
    }
  }
  return { width, height, data };
}

const pixel = (img: RgbaImage, x: number, y: number) => Array.from(img.data.slice((y * img.width + x) * 4, (y * img.width + x) * 4 + 4));

describe('estimateSimilarity', () => {
  it('recovers a known rotation, scale and translation', () => {
    const truth = similarity(1.7, 0.4, 12, -30);
    const src: Point[] = ARCFACE_TEMPLATE.map(([x, y]) => [x * 3 + 5, y * 2 - 1]);
    const m = estimateSimilarity(src, src.map((p) => apply(truth, p)));
    m.forEach((v, i) => expect(v).toBeCloseTo(truth[i], 6));
  });

  it('returns the identity for identical point sets', () => {
    const m = estimateSimilarity(ARCFACE_TEMPLATE, ARCFACE_TEMPLATE);
    [1, 0, 0, 0, 1, 0].forEach((v, i) => expect(m[i]).toBeCloseTo(v, 9));
  });
});

describe('invertAffine', () => {
  it('undoes the transform', () => {
    const m = similarity(0.5, -1.1, 40, 7);
    const back = apply(invertAffine(m), apply(m, [13, 29]));
    expect(back[0]).toBeCloseTo(13, 9);
    expect(back[1]).toBeCloseTo(29, 9);
  });
});

describe('warpAffine', () => {
  it('moves pixels under a translation', () => {
    const out = warpAffine(gradient(10, 10), [1, 0, 2, 0, 1, 3], 10, 10);
    expect(pixel(out, 5, 5)).toEqual([30, 20, 0, 255]);
  });

  it('interpolates bilinearly between source pixels', () => {
    const out = warpAffine(gradient(10, 10), [1, 0, 0.5, 0, 1, 0], 10, 10);
    expect(pixel(out, 5, 5)).toEqual([45, 50, 0, 255]);
  });

  it('fills pixels that fall outside the source with opaque black', () => {
    const out = warpAffine(gradient(10, 10), [1, 0, 5, 0, 1, 5], 10, 10);
    expect(pixel(out, 0, 0)).toEqual([0, 0, 0, 255]);
  });
});

describe('alignFace', () => {
  it('moves the five landmarks onto the ArcFace template in a 112×112 crop', () => {
    const toImage = similarity(2.2, 0.3, 90, 40);
    const landmarks = ARCFACE_TEMPLATE.map((p) => apply(toImage, p));
    const width = 400;
    const height = 400;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let i = 3; i < data.length; i += 4) data[i] = 255;
    for (const [lx, ly] of landmarks) {
      for (let y = Math.round(ly) - 3; y <= Math.round(ly) + 3; y++) {
        for (let x = Math.round(lx) - 3; x <= Math.round(lx) + 3; x++) data.fill(255, (y * width + x) * 4, (y * width + x) * 4 + 3);
      }
    }
    const aligned = alignFace({ width, height, data }, landmarks);
    expect([aligned.width, aligned.height]).toEqual([ALIGNED_SIZE, ALIGNED_SIZE]);
    for (const [tx, ty] of ARCFACE_TEMPLATE) expect(pixel(aligned, Math.round(tx), Math.round(ty))[0]).toBeGreaterThan(200);
    expect(pixel(aligned, 56, 20)[0]).toBe(0);
  });
});
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `npx vitest run tests/unit/align.test.ts`
Expected: FAIL。`src/vision/align.ts` が見つからない。

- [ ] **Step 3: 実装する**

`src/vision/types.ts`:

```ts
export type Point = [number, number];

/** RGBA pixels, row-major. Structurally compatible with the DOM ImageData. */
export interface RgbaImage {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
}

export interface Face {
  box: { x: number; y: number; width: number; height: number };
  /** right eye, left eye, nose tip, right mouth corner, left mouth corner (image coordinates) */
  landmarks: Point[];
  score: number;
}
```

`src/vision/align.ts`:

```ts
import type { Point, RgbaImage } from './types.ts';

export const ALIGNED_SIZE = 112;

/** Five-point ArcFace template for a 112×112 crop (same values as OpenCV's FaceRecognizerSF). */
export const ARCFACE_TEMPLATE: Point[] = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

/** 2×3 affine matrix in OpenCV order: x' = m[0]x + m[1]y + m[2], y' = m[3]x + m[4]y + m[5]. */
export type Affine = [number, number, number, number, number, number];

/** Least-squares similarity transform (rotation + uniform scale + translation) mapping src onto dst. */
export function estimateSimilarity(src: Point[], dst: Point[]): Affine {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    sx += src[i][0]; sy += src[i][1];
    dx += dst[i][0]; dy += dst[i][1];
  }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let num = 0, numCross = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const ax = src[i][0] - sx, ay = src[i][1] - sy;
    const bx = dst[i][0] - dx, by = dst[i][1] - dy;
    num += ax * bx + ay * by;
    numCross += ax * by - ay * bx;
    den += ax * ax + ay * ay;
  }
  const a = num / den;
  const b = numCross / den;
  return [a, -b, dx - (a * sx - b * sy), b, a, dy - (b * sx + a * sy)];
}

export function invertAffine(m: Affine): Affine {
  const [a, b, c, d, e, f] = m;
  const det = a * e - b * d;
  const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
  return [ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)];
}

/** Warps `src` by `m` (src → dst coordinates) into a width×height image with bilinear sampling; outside pixels are black. */
export function warpAffine(src: RgbaImage, m: Affine, width: number, height: number): RgbaImage {
  const inv = invertAffine(m);
  const out = new Uint8ClampedArray(width * height * 4);
  const sw = src.width, sh = src.height, sd = src.data;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const fx = inv[0] * x + inv[1] * y + inv[2];
      const fy = inv[3] * x + inv[4] * y + inv[5];
      const x0 = Math.floor(fx), y0 = Math.floor(fy);
      const wx = fx - x0, wy = fy - y0;
      const o = (y * width + x) * 4;
      for (let ch = 0; ch < 3; ch++) {
        let v = 0;
        for (let j = 0; j < 2; j++) {
          const yy = y0 + j;
          if (yy < 0 || yy >= sh) continue;
          const wyj = j === 0 ? 1 - wy : wy;
          for (let i = 0; i < 2; i++) {
            const xx = x0 + i;
            if (xx < 0 || xx >= sw) continue;
            const wxi = i === 0 ? 1 - wx : wx;
            v += sd[(yy * sw + xx) * 4 + ch] * wxi * wyj;
          }
        }
        out[o + ch] = v;
      }
      out[o + 3] = 255;
    }
  }
  return { width, height, data: out };
}

/** Crops the face to 112×112 so that its five landmarks land on the ArcFace template. */
export function alignFace(src: RgbaImage, landmarks: Point[]): RgbaImage {
  return warpAffine(src, estimateSimilarity(landmarks, ARCFACE_TEMPLATE), ALIGNED_SIZE, ALIGNED_SIZE);
}
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `npx vitest run tests/unit/align.test.ts && npx tsc`
Expected: `Tests  7 passed (7)`。`tsc` はエラーなし。

- [ ] **Step 5: コミットする**

```bash
git add src/vision/types.ts src/vision/align.ts tests/unit/align.test.ts
git commit -F - <<'EOF'
Add five-point face alignment

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PADPRtudUhYxccwepxBWdj
EOF
```

---

### Task 3: YuNet の前処理と後処理

**Files:**
- Create: `src/vision/detector.ts`
- Test: `tests/unit/detector.test.ts`

**Interfaces:**
- Consumes: `Face`, `Point`, `RgbaImage`（Task 2 の `src/vision/types.ts`）
- Produces（`src/vision/detector.ts`）:
  - `YUNET_STRIDES = [8, 16, 32] as const`
  - `interface DetectorLayout { width: number; height: number; scale: number }`、`detectorLayout(frameWidth: number, frameHeight: number, longSide: number): DetectorLayout`（長辺を `longSide` にし、縦横を32の倍数まで右下に余白を足したサイズ。`scale` は入力ピクセル ÷ フレームピクセル）
  - `toYuNetInput(img: RgbaImage): Float32Array`（BGR、0〜255、NCHW）
  - `type YuNetOutputs = Record<string, { data: unknown }>`
  - `decodeYuNet(outputs: YuNetOutputs, inputWidth: number, inputHeight: number, scoreThreshold: number): Face[]`
  - `nms(faces: Face[], iouThreshold: number): Face[]`（スコア順）
  - `largestFace(faces: Face[]): Face | undefined`
  - `toFrameCoords(face: Face, layout: DetectorLayout): Face`
  - `postprocessYuNet(outputs: YuNetOutputs, layout: DetectorLayout, scoreThreshold: number, iouThreshold: number): Face[]`（復元 → NMS → フレーム座標）

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/detector.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { decodeYuNet, detectorLayout, largestFace, nms, toFrameCoords, toYuNetInput, YUNET_STRIDES } from '../../src/vision/detector.ts';
import type { Face } from '../../src/vision/types.ts';

function emptyOutputs(width: number, height: number) {
  const out: Record<string, { data: Float32Array }> = {};
  for (const s of YUNET_STRIDES) {
    const cells = (width / s) * (height / s);
    out[`cls_${s}`] = { data: new Float32Array(cells) };
    out[`obj_${s}`] = { data: new Float32Array(cells) };
    out[`bbox_${s}`] = { data: new Float32Array(cells * 4) };
    out[`kps_${s}`] = { data: new Float32Array(cells * 10) };
  }
  return out;
}

const face = (x: number, y: number, size: number, score: number): Face => ({
  box: { x, y, width: size, height: size },
  landmarks: [[x, y], [x, y], [x, y], [x, y], [x, y]],
  score,
});

describe('detectorLayout', () => {
  it('scales the long side to 320 and pads to multiples of 32', () => {
    expect(detectorLayout(640, 480, 320)).toEqual({ width: 320, height: 256, scale: 0.5 });
    expect(detectorLayout(480, 640, 320)).toEqual({ width: 256, height: 320, scale: 0.5 });
    expect(detectorLayout(1280, 720, 320)).toEqual({ width: 320, height: 192, scale: 0.25 });
  });
});

describe('toYuNetInput', () => {
  it('converts RGBA to planar BGR in 0–255', () => {
    const img = { width: 2, height: 1, data: new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]) };
    expect(Array.from(toYuNetInput(img))).toEqual([30, 60, 20, 50, 10, 40]);
  });
});

describe('decodeYuNet', () => {
  it('decodes box, landmarks and score of a cell', () => {
    const out = emptyOutputs(64, 32);
    // stride 16 grid is 4×2; cell (row 1, col 2) has index 6
    out.cls_16.data[6] = 0.81;
    out.obj_16.data[6] = 1;
    out.bbox_16.data.set([0.5, 0.25, Math.log(2), Math.log(1.5)], 6 * 4);
    out.kps_16.data.set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 6 * 10);
    const faces = decodeYuNet(out, 64, 32, 0.5);
    expect(faces).toHaveLength(1);
    const [f] = faces;
    expect(f.score).toBeCloseTo(0.9, 6);
    expect(f.box.x).toBeCloseTo(24, 4);
    expect(f.box.y).toBeCloseTo(8, 4);
    expect(f.box.width).toBeCloseTo(32, 4);
    expect(f.box.height).toBeCloseTo(24, 4);
    expect(f.landmarks[0][0]).toBeCloseTo(33.6, 4);
    expect(f.landmarks[0][1]).toBeCloseTo(19.2, 4);
    expect(f.landmarks[4][0]).toBeCloseTo(46.4, 4);
    expect(f.landmarks[4][1]).toBeCloseTo(32, 4);
  });

  it('drops cells below the score threshold and clamps scores above 1', () => {
    const out = emptyOutputs(64, 32);
    out.cls_8.data[0] = 0.25;
    out.obj_8.data[0] = 1; // score 0.5
    out.cls_32.data[1] = 1.4;
    out.obj_32.data[1] = 1; // clamped to 1
    const faces = decodeYuNet(out, 64, 32, 0.8);
    expect(faces.map((f) => f.score)).toEqual([1]);
  });
});

describe('nms', () => {
  it('keeps the best of overlapping faces and every separate face, best first', () => {
    const kept = nms([face(0, 0, 100, 0.85), face(10, 10, 100, 0.95), face(300, 0, 100, 0.9)], 0.3);
    expect(kept.map((f) => f.score)).toEqual([0.95, 0.9]);
  });

  it('keeps overlapping faces whose IoU is at or below the threshold', () => {
    const kept = nms([face(0, 0, 100, 0.9), face(80, 0, 100, 0.8)], 0.3);
    expect(kept).toHaveLength(2);
  });
});

describe('toFrameCoords', () => {
  it('undoes the detector scaling', () => {
    const f = toFrameCoords(face(10, 20, 30, 0.9), { width: 320, height: 256, scale: 0.5 });
    expect(f.box).toEqual({ x: 20, y: 40, width: 60, height: 60 });
    expect(f.landmarks[0]).toEqual([20, 40]);
    expect(f.score).toBe(0.9);
  });
});

describe('largestFace', () => {
  it('picks the face with the biggest box, so only the person nearest the camera is recognized', () => {
    expect(largestFace([face(0, 0, 50, 0.99), face(100, 0, 120, 0.85), face(300, 0, 80, 0.9)])?.score).toBe(0.85);
  });

  it('returns undefined when there is no face', () => {
    expect(largestFace([])).toBeUndefined();
  });
});
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `npx vitest run tests/unit/detector.test.ts`
Expected: FAIL。`src/vision/detector.ts` が見つからない。

- [ ] **Step 3: 実装する**

復元の式は OpenCV の `FaceDetectorYN`（`modules/objdetect/src/face_detect.cpp`）と同じ。

`src/vision/detector.ts`:

```ts
import type { Face, Point, RgbaImage } from './types.ts';

export const YUNET_STRIDES = [8, 16, 32] as const;

/** Size of the YuNet input: the frame scaled so its long side is `longSide`, then padded right/bottom to multiples of 32. */
export interface DetectorLayout {
  width: number;
  height: number;
  /** input pixels per frame pixel */
  scale: number;
}

export function detectorLayout(frameWidth: number, frameHeight: number, longSide: number): DetectorLayout {
  const scale = longSide / Math.max(frameWidth, frameHeight);
  return {
    width: Math.ceil((frameWidth * scale) / 32) * 32,
    height: Math.ceil((frameHeight * scale) / 32) * 32,
    scale,
  };
}

/** RGBA → YuNet input tensor data: BGR, 0–255, NCHW. */
export function toYuNetInput(img: RgbaImage): Float32Array {
  const plane = img.width * img.height;
  const out = new Float32Array(plane * 3);
  const d = img.data;
  for (let i = 0; i < plane; i++) {
    out[i] = d[i * 4 + 2];
    out[plane + i] = d[i * 4 + 1];
    out[plane * 2 + i] = d[i * 4];
  }
  return out;
}

/** YuNet outputs keyed by name: cls_8, obj_8, bbox_8, kps_8, … for strides 8/16/32. */
export type YuNetOutputs = Record<string, { data: unknown }>;

const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1);

/** Decodes raw YuNet outputs into faces in input-pixel coordinates (same formulas as OpenCV's FaceDetectorYN). */
export function decodeYuNet(outputs: YuNetOutputs, inputWidth: number, inputHeight: number, scoreThreshold: number): Face[] {
  const faces: Face[] = [];
  for (const stride of YUNET_STRIDES) {
    const cols = inputWidth / stride;
    const rows = inputHeight / stride;
    const cls = outputs[`cls_${stride}`].data as Float32Array;
    const obj = outputs[`obj_${stride}`].data as Float32Array;
    const bbox = outputs[`bbox_${stride}`].data as Float32Array;
    const kps = outputs[`kps_${stride}`].data as Float32Array;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const score = Math.sqrt(clamp01(cls[idx]) * clamp01(obj[idx]));
        if (score < scoreThreshold) continue;
        const cx = (c + bbox[idx * 4]) * stride;
        const cy = (r + bbox[idx * 4 + 1]) * stride;
        const w = Math.exp(bbox[idx * 4 + 2]) * stride;
        const h = Math.exp(bbox[idx * 4 + 3]) * stride;
        const landmarks: Point[] = [];
        for (let n = 0; n < 5; n++) {
          landmarks.push([(kps[idx * 10 + n * 2] + c) * stride, (kps[idx * 10 + n * 2 + 1] + r) * stride]);
        }
        faces.push({ box: { x: cx - w / 2, y: cy - h / 2, width: w, height: h }, landmarks, score });
      }
    }
  }
  return faces;
}

function iou(a: Face['box'], b: Face['box']): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.width * a.height + b.width * b.height - inter);
}

/** Greedy non-maximum suppression: keeps the highest-scoring face of each overlapping group, sorted by score. */
export function nms(faces: Face[], iouThreshold: number): Face[] {
  const sorted = [...faces].sort((p, q) => q.score - p.score);
  const kept: Face[] = [];
  for (const f of sorted) {
    if (kept.every((k) => iou(k.box, f.box) <= iouThreshold)) kept.push(f);
  }
  return kept;
}

/** The face with the biggest box (the first one on ties), or undefined for none. */
export function largestFace(faces: Face[]): Face | undefined {
  let best: Face | undefined;
  for (const f of faces) if (!best || f.box.width * f.box.height > best.box.width * best.box.height) best = f;
  return best;
}

/** Maps a face from input-pixel coordinates back to frame coordinates. */
export function toFrameCoords(face: Face, layout: DetectorLayout): Face {
  const s = 1 / layout.scale;
  return {
    box: { x: face.box.x * s, y: face.box.y * s, width: face.box.width * s, height: face.box.height * s },
    landmarks: face.landmarks.map(([x, y]) => [x * s, y * s] as Point),
    score: face.score,
  };
}

export function postprocessYuNet(
  outputs: YuNetOutputs,
  layout: DetectorLayout,
  scoreThreshold: number,
  iouThreshold: number,
): Face[] {
  return nms(decodeYuNet(outputs, layout.width, layout.height, scoreThreshold), iouThreshold).map((f) =>
    toFrameCoords(f, layout),
  );
}
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `npx vitest run tests/unit/detector.test.ts && npx tsc`
Expected: `Tests  9 passed (9)`。`tsc` はエラーなし。

- [ ] **Step 5: コミットする**

```bash
git add src/vision/detector.ts tests/unit/detector.test.ts
git commit -F - <<'EOF'
Add YuNet pre- and post-processing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PADPRtudUhYxccwepxBWdj
EOF
```

---

### Task 4: 照合と登録者の保存

**Files:**
- Create: `src/match.ts`, `src/store.ts`
- Test: `tests/unit/match.test.ts`, `tests/unit/store.test.ts`

**Interfaces:**
- Consumes: なし
- Produces:
  - `src/match.ts`: `interface EnrolledPerson { id: string; name: string; embeddings: ArrayLike<number>[] }`、`type Match = { kind: 'matched'; id: string; name: string; score: number } | { kind: 'unknown'; score: number }`、`dot(a: ArrayLike<number>, b: ArrayLike<number>): number`、`matchFace(query: ArrayLike<number>, people: EnrolledPerson[], threshold: number): Match | null`（特徴量を持つ人が1人もいなければ null）
  - `src/store.ts`: `interface Person { id: string; name: string; embeddings: number[][]; thumbnail: Blob; createdAt: number }`、`interface PeopleStore { listPeople(): Promise<Person[]>; addEnrollment(name: string, embeddings: ArrayLike<number>[], thumbnail: Blob): Promise<Person>; deletePerson(id: string): Promise<void>; deleteAll(): Promise<void> }`、`mergeEnrollment(...)`、`openPeopleStore(dbName = 'face-reception'): Promise<PeopleStore>`、`createMemoryPeopleStore(): PeopleStore`。`listPeople` は登録の古い順。`addEnrollment` は名前の前後の空白を除き、同じ名前があればその人に特徴量を追加し（id とサムネイルはそのまま）、空なら `Error('name is empty')` を投げる

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/match.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { dot, matchFace, type EnrolledPerson } from '../../src/match.ts';

const unit = (deg: number) => [Math.cos((deg * Math.PI) / 180), Math.sin((deg * Math.PI) / 180)];

const people: EnrolledPerson[] = [
  { id: 'a', name: 'あおやま', embeddings: [unit(0), unit(80)] },
  { id: 'b', name: 'たなか', embeddings: [unit(40)] },
];

describe('dot', () => {
  it('is the cosine similarity of unit vectors', () => {
    expect(dot(unit(0), unit(60))).toBeCloseTo(0.5, 9);
  });
});

describe('matchFace', () => {
  it('returns null when nobody is enrolled', () => {
    expect(matchFace(unit(0), [], 0.363)).toBeNull();
    expect(matchFace(unit(0), [{ id: 'x', name: 'x', embeddings: [] }], 0.363)).toBeNull();
  });

  it("scores each person by their closest embedding and picks the best person", () => {
    const m = matchFace(unit(75), people, 0.363);
    expect(m).toMatchObject({ kind: 'matched', id: 'a', name: 'あおやま' });
    expect(m?.score).toBeCloseTo(Math.cos((5 * Math.PI) / 180), 9);
  });

  it('accepts a score exactly at the threshold', () => {
    const threshold = dot(unit(30), unit(40));
    expect(matchFace(unit(30), [people[1]], threshold)?.kind).toBe('matched');
  });

  it('reports unknown with the best score when nobody reaches the threshold', () => {
    const m = matchFace(unit(-100), people, 0.363);
    expect(m?.kind).toBe('unknown');
    expect(m?.score).toBeLessThan(0.363);
  });
});
```

`tests/unit/store.test.ts`:

```ts
import 'fake-indexeddb/auto';
import { describe, expect, it } from 'vitest';
import { createMemoryPeopleStore, openPeopleStore, type PeopleStore } from '../../src/store.ts';

const thumb = (byte: number) => new Blob([new Uint8Array([byte])], { type: 'image/jpeg' });
const vec = (...v: number[]) => new Float32Array(v);

describe.each([
  ['IndexedDB', () => openPeopleStore(`test-${crypto.randomUUID()}`)],
  ['memory', async () => createMemoryPeopleStore()],
] as const)('%s store', (_label, open: () => Promise<PeopleStore>) => {
  it('adds a person with their embeddings and thumbnail', async () => {
    const store = await open();
    const added = await store.addEnrollment('あおやま', [vec(0.5, 0.25), vec(1, 0)], thumb(1));
    const [person, ...rest] = await store.listPeople();
    expect(rest).toEqual([]);
    expect(person.id).toBe(added.id);
    expect(person.name).toBe('あおやま');
    expect(person.embeddings).toEqual([[0.5, 0.25], [1, 0]]);
    expect(person.thumbnail).toBeInstanceOf(Blob);
    expect(person.thumbnail.size).toBe(1);
  });

  it('appends embeddings when the same name (ignoring surrounding spaces) enrolls again', async () => {
    const store = await open();
    const first = await store.addEnrollment('たなか', [vec(1, 0)], thumb(1));
    await store.addEnrollment('  たなか ', [vec(0, 1)], thumb(2));
    const people = await store.listPeople();
    expect(people).toHaveLength(1);
    expect(people[0].id).toBe(first.id);
    expect(people[0].embeddings).toEqual([[1, 0], [0, 1]]);
    expect(new Uint8Array(await people[0].thumbnail.arrayBuffer())).toEqual(new Uint8Array([1]));
  });

  it('rejects an empty name', async () => {
    const store = await open();
    await expect(store.addEnrollment('   ', [vec(1, 0)], thumb(1))).rejects.toThrow('name is empty');
  });

  it('deletes one person or everyone', async () => {
    const store = await open();
    const a = await store.addEnrollment('a', [vec(1, 0)], thumb(1));
    await store.addEnrollment('b', [vec(0, 1)], thumb(2));
    await store.addEnrollment('c', [vec(1, 1)], thumb(3));
    await store.deletePerson(a.id);
    expect((await store.listPeople()).map((p) => p.name).sort()).toEqual(['b', 'c']);
    await store.deleteAll();
    expect(await store.listPeople()).toEqual([]);
  });
});

describe('IndexedDB store persistence', () => {
  it('keeps people across reopening the database', async () => {
    const name = `test-${crypto.randomUUID()}`;
    await (await openPeopleStore(name)).addEnrollment('すずき', [vec(1, 0)], thumb(1));
    const reopened = await openPeopleStore(name);
    expect((await reopened.listPeople()).map((p) => p.name)).toEqual(['すずき']);
  });
});
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `npx vitest run tests/unit/match.test.ts tests/unit/store.test.ts`
Expected: FAIL。`src/match.ts` と `src/store.ts` が見つからない。

- [ ] **Step 3: 実装する**

`src/match.ts`:

```ts
export interface EnrolledPerson {
  id: string;
  name: string;
  embeddings: ArrayLike<number>[];
}

export type Match =
  | { kind: 'matched'; id: string; name: string; score: number }
  | { kind: 'unknown'; score: number };

/** Dot product. Embeddings are unit length, so this is their cosine similarity. */
export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/** Scores each person by their most similar embedding. Returns null when nobody has any embedding. */
export function matchFace(query: ArrayLike<number>, people: EnrolledPerson[], threshold: number): Match | null {
  let best: { person: EnrolledPerson; score: number } | null = null;
  for (const person of people) {
    for (const embedding of person.embeddings) {
      const score = dot(query, embedding);
      if (!best || score > best.score) best = { person, score };
    }
  }
  if (!best) return null;
  return best.score >= threshold
    ? { kind: 'matched', id: best.person.id, name: best.person.name, score: best.score }
    : { kind: 'unknown', score: best.score };
}
```

`src/store.ts`:

```ts
import { openDB, type IDBPDatabase } from 'idb';

export interface Person {
  id: string;
  name: string;
  embeddings: number[][];
  thumbnail: Blob;
  createdAt: number;
}

export interface PeopleStore {
  listPeople(): Promise<Person[]>;
  /** Adds a new person, or appends the embeddings to the person who already has this name. */
  addEnrollment(name: string, embeddings: ArrayLike<number>[], thumbnail: Blob): Promise<Person>;
  deletePerson(id: string): Promise<void>;
  deleteAll(): Promise<void>;
}

const STORE = 'people';

/** The person to write for an enrollment, given everyone currently stored. */
export function mergeEnrollment(
  people: Person[],
  name: string,
  embeddings: ArrayLike<number>[],
  thumbnail: Blob,
): Person {
  const trimmed = name.trim();
  if (!trimmed) throw new Error('name is empty');
  const vectors = embeddings.map((e) => Array.from(e));
  const existing = people.find((p) => p.name === trimmed);
  if (existing) return { ...existing, embeddings: [...existing.embeddings, ...vectors] };
  return { id: crypto.randomUUID(), name: trimmed, embeddings: vectors, thumbnail, createdAt: Date.now() };
}

const byCreatedAt = (a: Person, b: Person) => a.createdAt - b.createdAt;

class IdbPeopleStore implements PeopleStore {
  private readonly db: IDBPDatabase;

  constructor(db: IDBPDatabase) {
    this.db = db;
  }

  async listPeople(): Promise<Person[]> {
    return ((await this.db.getAll(STORE)) as Person[]).sort(byCreatedAt);
  }

  async addEnrollment(name: string, embeddings: ArrayLike<number>[], thumbnail: Blob): Promise<Person> {
    const tx = this.db.transaction(STORE, 'readwrite');
    const person = mergeEnrollment((await tx.store.getAll()) as Person[], name, embeddings, thumbnail);
    await tx.store.put(person);
    await tx.done;
    return person;
  }

  async deletePerson(id: string): Promise<void> {
    await this.db.delete(STORE, id);
  }

  async deleteAll(): Promise<void> {
    await this.db.clear(STORE);
  }
}

export async function openPeopleStore(dbName = 'face-reception'): Promise<PeopleStore> {
  const db = await openDB(dbName, 1, {
    upgrade(database) {
      database.createObjectStore(STORE, { keyPath: 'id' });
    },
  });
  return new IdbPeopleStore(db);
}

/** Fallback when IndexedDB is unavailable: same behavior, but nothing survives a reload. */
export function createMemoryPeopleStore(): PeopleStore {
  const people = new Map<string, Person>();
  return {
    async listPeople() {
      return [...people.values()].sort(byCreatedAt);
    },
    async addEnrollment(name, embeddings, thumbnail) {
      const person = mergeEnrollment([...people.values()], name, embeddings, thumbnail);
      people.set(person.id, person);
      return person;
    },
    async deletePerson(id) {
      people.delete(id);
    },
    async deleteAll() {
      people.clear();
    },
  };
}
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `npx vitest run tests/unit/match.test.ts tests/unit/store.test.ts && npx tsc`
Expected: `Tests  14 passed (14)`。`tsc` はエラーなし。

- [ ] **Step 5: コミットする**

```bash
git add src/match.ts src/store.ts tests/unit/match.test.ts tests/unit/store.test.ts
git commit -F - <<'EOF'
Add cosine matching and IndexedDB people store

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PADPRtudUhYxccwepxBWdj
EOF
```

---

### Task 5: SFace と推論パイプライン（OpenCV の結果と突き合わせる）

**Files:**
- Create: `scripts/fetch-fixtures.mjs`, `scripts/opencv_reference.py`, `tests/fixtures/faces/README.md`
- Create（スクリプトで生成）: `tests/fixtures/faces/meir-a.jpg`, `tests/fixtures/faces/meir-b.jpg`, `tests/fixtures/faces/kim.jpg`, `tests/fixtures/opencv-reference.json`
- Create: `tests/helpers/images.ts`, `src/vision/embedder.ts`, `src/vision/pipeline.ts`
- Test: `tests/unit/embedder.test.ts`, `tests/unit/pipeline.test.ts`

**Interfaces:**
- Consumes: Task 2（`alignFace`, `warpAffine`, 型）、Task 3（`detectorLayout`, `postprocessYuNet`, `toYuNetInput`, `DetectorLayout`）、Task 4（`dot`）、Task 1（`MODELS` ほか定数、`readModel`）
- Produces:
  - `src/vision/embedder.ts`: `toSFaceInput(aligned: RgbaImage): Float32Array`（RGB、0〜255、NCHW）、`l2normalize(v: Float32Array): Float32Array`
  - `src/vision/pipeline.ts`: `interface DetectOptions { scoreThreshold: number; iouThreshold: number }`、`class FacePipeline { static create(yunetModel: Uint8Array, sfaceModel: Uint8Array): Promise<FacePipeline>; detect(input: RgbaImage, layout: DetectorLayout, opts: DetectOptions): Promise<Face[]>; embed(frame: RgbaImage, face: Face): Promise<{ embedding: Float32Array; aligned: RgbaImage }> }`。`detect` の `input` は `layout.width × layout.height` に縮小・余白付けしたコマで、戻り値はフレーム座標。`embed` の `frame` は原寸のコマ
  - `tests/helpers/images.ts`: `FACE_NAMES`, `type FaceName`, `loadFace(name: FaceName): RgbaImage`, `prepareDetectorInput(frame: RgbaImage, longSide: number): { input: RgbaImage; layout: DetectorLayout }`
  - テスト用の顔写真 3 枚（960×720 JPEG）。Task 8 以降の E2E で Chrome の仮想カメラにも使う

- [ ] **Step 1: テスト用の顔写真を取得する**

NASA 宇宙飛行士の公式写真（パブリックドメイン）を Wikimedia Commons から取り、上側 960×720 を切り出す。同じ人の別写真 2 枚と、別人 1 枚。

`scripts/fetch-fixtures.mjs`:

```js
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
```

`tests/fixtures/faces/README.md`:

````markdown
# Face fixtures

Official NASA astronaut portraits from Wikimedia Commons. As works of NASA they are in the public domain.
`scripts/fetch-fixtures.mjs` downloads the 960px versions and keeps the top 960×720 (the face is in the upper part).

| File | Person | Source |
|---|---|---|
| `meir-a.jpg` | Jessica Meir (2025) | https://commons.wikimedia.org/wiki/File:Official_portrait_of_NASA_astronaut_Jessica_Meir_wearing_a_spacesuit_(jsc2025e078605_alt).jpg |
| `meir-b.jpg` | Jessica Meir (2018, black and white) | https://commons.wikimedia.org/wiki/File:Jessica_Meir_official_portrait_in_an_EMU_(B%26W).jpg |
| `kim.jpg` | Jonny Kim (2024) | https://commons.wikimedia.org/wiki/File:Jsc2024e052605_alt_(Aug._6,_2024)_---_Official_portrait_of_NASA_astronaut_Jonny_Kim.jpg |

`../opencv-reference.json` holds OpenCV's own YuNet + SFace results for these files (`scripts/opencv_reference.py`).
````

```bash
node scripts/fetch-fixtures.mjs
```

Expected: `meir-a.jpg 960x720`、`meir-b.jpg 960x720`、`kim.jpg 960x720` の3行。

- [ ] **Step 2: OpenCV の基準値を作る**

OpenCV 本家の `FaceDetectorYN` と `FaceRecognizerSF` で同じ写真を処理し、枠・5点・特徴量を JSON に書き出す。`uv` が必要（このマシンにはある）。

`scripts/opencv_reference.py`:

```python
"""Writes tests/fixtures/opencv-reference.json: OpenCV's own YuNet + SFace results for the face fixtures.

Run: uvx --with opencv-python-headless==5.0.0.93 --with numpy python scripts/opencv_reference.py
"""
import json
import pathlib

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "public" / "models"
FACES = ROOT / "tests" / "fixtures" / "faces"

detector = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2026may.onnx"), "", (320, 320), 0.8, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(str(MODELS / "face_recognition_sface_2021dec.onnx"), "")

out = {}
for name in ["meir-a", "meir-b", "kim"]:
    img = cv2.imread(str(FACES / f"{name}.jpg"))
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    assert faces is not None and len(faces) == 1, f"{name}: expected one face"
    face = faces[0]
    feature = recognizer.feature(recognizer.alignCrop(img, face)).flatten()
    feature = feature / np.linalg.norm(feature)
    out[name] = {
        "box": [float(v) for v in face[0:4]],
        "landmarks": [[float(face[4 + 2 * i]), float(face[5 + 2 * i])] for i in range(5)],
        "embedding": [float(v) for v in feature],
    }
    print(name, "box", np.round(face[0:4], 1), "score", round(float(face[14]), 3))

(ROOT / "tests" / "fixtures" / "opencv-reference.json").write_text(json.dumps(out))
```

```bash
uvx --with opencv-python-headless==5.0.0.93 --with numpy python scripts/opencv_reference.py
```

Expected: 3人それぞれについて `box [...] score 0.9...` の行が出て（OpenCV の WARN 行は無視してよい）、`tests/fixtures/opencv-reference.json` ができる。

- [ ] **Step 3: 失敗するテストを書く**

`tests/helpers/images.ts`:

```ts
import jpeg from 'jpeg-js';
import { readFileSync } from 'node:fs';
import { warpAffine } from '../../src/vision/align.ts';
import { detectorLayout, type DetectorLayout } from '../../src/vision/detector.ts';
import type { RgbaImage } from '../../src/vision/types.ts';

export const FACE_NAMES = ['meir-a', 'meir-b', 'kim'] as const;
export type FaceName = (typeof FACE_NAMES)[number];

export function loadFace(name: FaceName): RgbaImage {
  const file = new URL(`../fixtures/faces/${name}.jpg`, import.meta.url);
  return jpeg.decode(readFileSync(file), { useTArray: true, formatAsRGBA: true });
}

/** Resizes and pads a frame for YuNet the way the worker's canvas does (pixel-centre aligned bilinear). */
export function prepareDetectorInput(frame: RgbaImage, longSide: number): { input: RgbaImage; layout: DetectorLayout } {
  const layout = detectorLayout(frame.width, frame.height, longSide);
  const s = layout.scale;
  const input = warpAffine(frame, [s, 0, 0.5 * s - 0.5, 0, s, 0.5 * s - 0.5], layout.width, layout.height);
  return { input, layout };
}
```

`tests/unit/embedder.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { l2normalize, toSFaceInput } from '../../src/vision/embedder.ts';

describe('toSFaceInput', () => {
  it('converts RGBA to planar RGB in 0–255', () => {
    const img = { width: 2, height: 1, data: new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]) };
    expect(Array.from(toSFaceInput(img))).toEqual([10, 40, 20, 50, 30, 60]);
  });
});

describe('l2normalize', () => {
  it('scales the vector to unit length', () => {
    const v = l2normalize(new Float32Array([3, 4]));
    expect(v[0]).toBeCloseTo(0.6, 6);
    expect(v[1]).toBeCloseTo(0.8, 6);
  });
});
```

`tests/unit/pipeline.test.ts`:

```ts
import * as ort from 'onnxruntime-web/wasm';
import { readFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { DETECT_INPUT_LONG_SIDE, DETECT_SCORE_THRESHOLD, MATCH_THRESHOLD, MODELS, NMS_IOU_THRESHOLD } from '../../src/config.ts';
import { dot } from '../../src/match.ts';
import { FacePipeline } from '../../src/vision/pipeline.ts';
import { FACE_NAMES, loadFace, prepareDetectorInput, type FaceName } from '../helpers/images.ts';
import { readModel } from '../helpers/models.ts';

interface Reference {
  box: [number, number, number, number];
  landmarks: [number, number][];
  embedding: number[];
}
const reference: Record<FaceName, Reference> = JSON.parse(
  readFileSync(new URL('../fixtures/opencv-reference.json', import.meta.url), 'utf8'),
);
const opts = { scoreThreshold: DETECT_SCORE_THRESHOLD, iouThreshold: NMS_IOU_THRESHOLD };

ort.env.wasm.numThreads = 1;
let pipeline: FacePipeline;
beforeAll(async () => {
  pipeline = await FacePipeline.create(readModel(MODELS.yunet.file), readModel(MODELS.sface.file));
});

describe('FacePipeline against OpenCV (full resolution)', () => {
  it.each(FACE_NAMES)('%s: same face box, landmarks and embedding as OpenCV', async (name) => {
    const frame = loadFace(name);
    const { input, layout } = prepareDetectorInput(frame, Math.max(frame.width, frame.height));
    const faces = await pipeline.detect(input, layout, opts);
    expect(faces).toHaveLength(1);
    const ref = reference[name];
    const { box, landmarks } = faces[0];
    [box.x, box.y, box.width, box.height].forEach((v, i) => expect(Math.abs(v - ref.box[i])).toBeLessThan(1));
    landmarks.forEach(([x, y], i) => {
      expect(Math.abs(x - ref.landmarks[i][0])).toBeLessThan(1);
      expect(Math.abs(y - ref.landmarks[i][1])).toBeLessThan(1);
    });
    const { embedding, aligned } = await pipeline.embed(frame, faces[0]);
    expect([aligned.width, aligned.height]).toEqual([112, 112]);
    expect(dot(embedding, ref.embedding)).toBeGreaterThan(0.99);
  });
});

describe('FacePipeline at the app setting (long side 320)', () => {
  const embeddings = {} as Record<FaceName, Float32Array>;
  beforeAll(async () => {
    for (const name of FACE_NAMES) {
      const frame = loadFace(name);
      const { input, layout } = prepareDetectorInput(frame, DETECT_INPUT_LONG_SIDE);
      const faces = await pipeline.detect(input, layout, opts);
      expect(faces).toHaveLength(1);
      embeddings[name] = (await pipeline.embed(frame, faces[0])).embedding;
    }
  });

  it('matches two different photos of the same person', () => {
    expect(dot(embeddings['meir-a'], embeddings['meir-b'])).toBeGreaterThanOrEqual(MATCH_THRESHOLD);
  });

  it('does not match different people', () => {
    expect(dot(embeddings['meir-a'], embeddings.kim)).toBeLessThan(MATCH_THRESHOLD);
    expect(dot(embeddings['meir-b'], embeddings.kim)).toBeLessThan(MATCH_THRESHOLD);
  });

  it('finds no face in a blank frame', async () => {
    const blank = { width: 640, height: 480, data: new Uint8ClampedArray(640 * 480 * 4) };
    const { input, layout } = prepareDetectorInput(blank, DETECT_INPUT_LONG_SIDE);
    expect(await pipeline.detect(input, layout, opts)).toEqual([]);
  });
});
```

- [ ] **Step 4: テストが失敗することを確かめる**

Run: `npx vitest run tests/unit/embedder.test.ts tests/unit/pipeline.test.ts`
Expected: FAIL。`src/vision/embedder.ts` と `src/vision/pipeline.ts` が見つからない。

- [ ] **Step 5: 実装する**

`src/vision/embedder.ts`:

```ts
import type { RgbaImage } from './types.ts';

/** Aligned 112×112 RGBA → SFace input tensor data: RGB, 0–255, NCHW (SFace normalizes internally). */
export function toSFaceInput(aligned: RgbaImage): Float32Array {
  const plane = aligned.width * aligned.height;
  const out = new Float32Array(plane * 3);
  const d = aligned.data;
  for (let i = 0; i < plane; i++) {
    out[i] = d[i * 4];
    out[plane + i] = d[i * 4 + 1];
    out[plane * 2 + i] = d[i * 4 + 2];
  }
  return out;
}

export function l2normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (const x of v) sum += x * x;
  const norm = Math.sqrt(sum);
  return v.map((x) => x / norm);
}
```

`src/vision/pipeline.ts`:

```ts
import * as ort from 'onnxruntime-web/wasm';
import { alignFace } from './align.ts';
import { postprocessYuNet, toYuNetInput, type DetectorLayout } from './detector.ts';
import { l2normalize, toSFaceInput } from './embedder.ts';
import type { Face, RgbaImage } from './types.ts';

export interface DetectOptions {
  scoreThreshold: number;
  iouThreshold: number;
}

/** Runs YuNet and SFace. Environment-agnostic: used by the Web Worker and by Node tests. */
export class FacePipeline {
  private readonly yunet: ort.InferenceSession;
  private readonly sface: ort.InferenceSession;

  private constructor(yunet: ort.InferenceSession, sface: ort.InferenceSession) {
    this.yunet = yunet;
    this.sface = sface;
  }

  static async create(yunetModel: Uint8Array, sfaceModel: Uint8Array): Promise<FacePipeline> {
    const opts: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'], logSeverityLevel: 3 };
    const yunet = await ort.InferenceSession.create(yunetModel, opts);
    const sface = await ort.InferenceSession.create(sfaceModel, opts);
    return new FacePipeline(yunet, sface);
  }

  /** `input` is the frame already resized and padded to layout.width × layout.height. Returns faces in frame coordinates. */
  async detect(input: RgbaImage, layout: DetectorLayout, opts: DetectOptions): Promise<Face[]> {
    const tensor = new ort.Tensor('float32', toYuNetInput(input), [1, 3, layout.height, layout.width]);
    const outputs = await this.yunet.run({ input: tensor });
    return postprocessYuNet(outputs, layout, opts.scoreThreshold, opts.iouThreshold);
  }

  /** `frame` is the full-resolution frame the face was detected in. */
  async embed(frame: RgbaImage, face: Face): Promise<{ embedding: Float32Array; aligned: RgbaImage }> {
    const aligned = alignFace(frame, face.landmarks);
    const tensor = new ort.Tensor('float32', toSFaceInput(aligned), [1, 3, aligned.height, aligned.width]);
    const outputs = await this.sface.run({ data: tensor });
    return { embedding: l2normalize(outputs.fc1.data as Float32Array), aligned };
  }
}
```

- [ ] **Step 6: テストが通ることを確かめる**

Run: `npx vitest run tests/unit/embedder.test.ts tests/unit/pipeline.test.ts && npx tsc`
Expected: `Tests  8 passed (8)`。`tsc` はエラーなし。計画時の実測値は、原寸で OpenCV との差が 0.3px 以内・特徴量のコサイン類似度 0.997 以上、長辺 320 で同じ人どうし 0.64・別人 0.1 未満。

- [ ] **Step 7: 全テストを流してコミットする**

Run: `npm test`
Expected: `Tests  41 passed (41)`

```bash
git add scripts/fetch-fixtures.mjs scripts/opencv_reference.py tests/fixtures tests/helpers/images.ts src/vision/embedder.ts src/vision/pipeline.ts tests/unit/embedder.test.ts tests/unit/pipeline.test.ts
git commit -F - <<'EOF'
Add SFace embedding pipeline verified against OpenCV

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PADPRtudUhYxccwepxBWdj
EOF
```

---

### Task 6: 画面の状態遷移とエラー分類

**Files:**
- Create: `src/worker-protocol.ts`, `src/reception-state.ts`, `src/enroll-session.ts`, `src/errors.ts`, `src/ui/geometry.ts`
- Test: `tests/unit/reception-state.test.ts`, `tests/unit/enroll-session.test.ts`, `tests/unit/errors.test.ts`, `tests/unit/geometry.test.ts`

**Interfaces:**
- Consumes: `Face`, `RgbaImage`（Task 2）、`Match`（Task 4）
- Produces:
  - `src/worker-protocol.ts`: `interface LargestFace { embedding: Float32Array; aligned: RgbaImage }`、`interface Analysis { faces: Face[]; largest?: LargestFace; frameWidth: number; frameHeight: number }`、`type InitFailure = 'network' | 'unsupported'`、`type WorkerRequest`（`init` / `analyze`）、`type WorkerResponse`（`progress` / `ready` / `init-error` / `analysis` / `analyze-error`）
  - `src/reception-state.ts`: `type ReceptionView = { kind: 'no-people' } | { kind: 'no-face' } | { kind: 'checking' } | { kind: 'matched'; name: string; score: number } | { kind: 'unknown'; score: number }`、`interface ReceptionInput { faceCount: number; peopleCount: number; match?: Match | null }`、`class ReceptionState { update(input: ReceptionInput): ReceptionView; reset(): void }`。`checking` は spec の4状態に加えた「顔は映っているが、まだ照合結果がない」間（最大 0.5 秒）の表示
  - `src/enroll-session.ts`: `interface EnrollSample<T> { faceCount: number; embedding?: Float32Array; aligned?: T }`、`type EnrollStatus<T> = { kind: 'collecting'; count: number; total: number; hint: 'ok' | 'one-face' } | { kind: 'done'; embeddings: Float32Array[]; thumbnailSource: T } | { kind: 'timeout'; count: number; total: number }`、`class EnrollSession<T> { constructor(opts: { shots: number; timeoutMs: number; startedAt: number }); accept(sample: EnrollSample<T>, now: number): EnrollStatus<T> }`
  - `src/errors.ts`: `type CameraProblem = 'insecure' | 'denied' | 'not-found' | 'other'`、`classifyCameraError(error: unknown, secureContext: boolean): CameraProblem`、`cameraProblemMessage(problem: CameraProblem): string`、`initFailureMessage(reason: InitFailure): string`
  - `src/ui/geometry.ts`: `type Box = Face['box']`、`toViewBox(box: Box, frameWidth: number, frameHeight: number, viewWidth: number, viewHeight: number, mirrored: boolean): Box`

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/reception-state.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { ReceptionState } from '../../src/reception-state.ts';

const matched = { kind: 'matched', id: 'a', name: 'あおやま', score: 0.7 } as const;

describe('ReceptionState', () => {
  it('asks to enroll first when nobody is enrolled', () => {
    expect(new ReceptionState().update({ faceCount: 1, peopleCount: 0 })).toEqual({ kind: 'no-people' });
  });

  it('asks for a face when none is visible', () => {
    expect(new ReceptionState().update({ faceCount: 0, peopleCount: 2 })).toEqual({ kind: 'no-face' });
  });

  it('shows checking until the first embedding arrives', () => {
    expect(new ReceptionState().update({ faceCount: 1, peopleCount: 2 })).toEqual({ kind: 'checking' });
  });

  it('keeps the last result on frames without an embedding, and clears it when the face leaves', () => {
    const state = new ReceptionState();
    expect(state.update({ faceCount: 1, peopleCount: 2, match: matched })).toEqual({ kind: 'matched', name: 'あおやま', score: 0.7 });
    expect(state.update({ faceCount: 1, peopleCount: 2 })).toEqual({ kind: 'matched', name: 'あおやま', score: 0.7 });
    expect(state.update({ faceCount: 0, peopleCount: 2 })).toEqual({ kind: 'no-face' });
    expect(state.update({ faceCount: 1, peopleCount: 2 })).toEqual({ kind: 'checking' });
  });

  it('shows unknown with the best score', () => {
    const view = new ReceptionState().update({ faceCount: 1, peopleCount: 2, match: { kind: 'unknown', score: 0.12 } });
    expect(view).toEqual({ kind: 'unknown', score: 0.12 });
  });

  it('forgets the last result on reset', () => {
    const state = new ReceptionState();
    state.update({ faceCount: 1, peopleCount: 2, match: matched });
    state.reset();
    expect(state.update({ faceCount: 1, peopleCount: 2 })).toEqual({ kind: 'checking' });
  });
});
```

`tests/unit/enroll-session.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { EnrollSession } from '../../src/enroll-session.ts';

const emb = (v: number) => new Float32Array([v]);
const session = () => new EnrollSession<string>({ shots: 3, timeoutMs: 10_000, startedAt: 1_000 });

describe('EnrollSession', () => {
  it('collects the requested number of embeddings and keeps the first crop as the thumbnail', () => {
    const s = session();
    expect(s.accept({ faceCount: 1, embedding: emb(1), aligned: 'first' }, 1_100)).toEqual({ kind: 'collecting', count: 1, total: 3, hint: 'ok' });
    expect(s.accept({ faceCount: 1, embedding: emb(2), aligned: 'second' }, 1_500)).toMatchObject({ kind: 'collecting', count: 2 });
    const done = s.accept({ faceCount: 1, embedding: emb(3), aligned: 'third' }, 1_900);
    expect(done).toEqual({ kind: 'done', embeddings: [emb(1), emb(2), emb(3)], thumbnailSource: 'first' });
  });

  it('does not count frames without an embedding', () => {
    expect(session().accept({ faceCount: 1 }, 1_100)).toEqual({ kind: 'collecting', count: 0, total: 3, hint: 'ok' });
  });

  it('does not count frames with no face or several faces, and asks for one face', () => {
    const s = session();
    expect(s.accept({ faceCount: 0, embedding: emb(1), aligned: 'x' }, 1_100)).toEqual({ kind: 'collecting', count: 0, total: 3, hint: 'one-face' });
    expect(s.accept({ faceCount: 2, embedding: emb(1), aligned: 'x' }, 1_200)).toEqual({ kind: 'collecting', count: 0, total: 3, hint: 'one-face' });
  });

  it('times out after the deadline', () => {
    const s = session();
    s.accept({ faceCount: 1, embedding: emb(1), aligned: 'x' }, 1_100);
    expect(s.accept({ faceCount: 1, embedding: emb(2), aligned: 'y' }, 11_001)).toEqual({ kind: 'timeout', count: 1, total: 3 });
  });
});
```

`tests/unit/errors.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { cameraProblemMessage, classifyCameraError, initFailureMessage } from '../../src/errors.ts';

describe('classifyCameraError', () => {
  it('blames the page origin when the context is not secure', () => {
    expect(classifyCameraError(new DOMException('x', 'NotAllowedError'), false)).toBe('insecure');
  });

  it('maps getUserMedia errors', () => {
    expect(classifyCameraError(new DOMException('x', 'NotAllowedError'), true)).toBe('denied');
    expect(classifyCameraError(new DOMException('x', 'SecurityError'), true)).toBe('denied');
    expect(classifyCameraError(new DOMException('x', 'NotFoundError'), true)).toBe('not-found');
    expect(classifyCameraError(new DOMException('x', 'OverconstrainedError'), true)).toBe('not-found');
    expect(classifyCameraError(new DOMException('x', 'NotReadableError'), true)).toBe('other');
    expect(classifyCameraError('weird', true)).toBe('other');
  });
});

describe('messages', () => {
  it('explains how to allow the camera on iPhone when denied', () => {
    expect(cameraProblemMessage('denied')).toContain('設定');
  });

  it('tells network failures apart from unsupported browsers', () => {
    expect(initFailureMessage('network')).toContain('もう一度');
    expect(initFailureMessage('unsupported')).toContain('対応していません');
  });
});
```

`tests/unit/geometry.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { toViewBox } from '../../src/ui/geometry.ts';

const box = { x: 200, y: 100, width: 100, height: 50 };

describe('toViewBox', () => {
  it('crops the sides when a landscape frame fills a portrait view', () => {
    expect(toViewBox(box, 640, 480, 320, 480, false)).toEqual({ x: 40, y: 100, width: 100, height: 50 });
  });

  it('flips horizontally for a mirrored view', () => {
    expect(toViewBox(box, 640, 480, 320, 480, true)).toEqual({ x: 180, y: 100, width: 100, height: 50 });
  });

  it('handles the portrait frames an iPhone front camera delivers', () => {
    // 480×640 frame in a 390×300 view: scale 0.8125, frame shown 390×520, 110px cropped top and bottom
    expect(toViewBox(box, 480, 640, 390, 300, true)).toEqual({ x: 146.25, y: -28.75, width: 81.25, height: 40.625 });
  });

  it('scales when the view is smaller than the frame', () => {
    expect(toViewBox(box, 640, 480, 320, 240, false)).toEqual({ x: 100, y: 50, width: 50, height: 25 });
  });
});
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `npx vitest run tests/unit/reception-state.test.ts tests/unit/enroll-session.test.ts tests/unit/errors.test.ts tests/unit/geometry.test.ts`
Expected: FAIL。4つのモジュールが見つからない。

- [ ] **Step 3: 実装する**

`src/worker-protocol.ts`:

```ts
import type { Face, RgbaImage } from './vision/types.ts';

export interface LargestFace {
  embedding: Float32Array;
  /** 112×112 aligned crop, used as the enrollment thumbnail */
  aligned: RgbaImage;
}

export interface Analysis {
  faces: Face[];
  /** Present only when the request asked for an embedding and at least one face was found. */
  largest?: LargestFace;
  frameWidth: number;
  frameHeight: number;
}

export type InitFailure = 'network' | 'unsupported';

export type WorkerRequest =
  | { type: 'init'; modelBaseUrl: string }
  | { type: 'analyze'; id: number; frame: ImageBitmap; embed: boolean };

export type WorkerResponse =
  | { type: 'progress'; loaded: number; total: number }
  | { type: 'ready' }
  | { type: 'init-error'; reason: InitFailure; message: string }
  | { type: 'analysis'; id: number; analysis: Analysis }
  | { type: 'analyze-error'; id: number; message: string };
```

`src/reception-state.ts`:

```ts
import type { Match } from './match.ts';

export type ReceptionView =
  | { kind: 'no-people' }
  | { kind: 'no-face' }
  | { kind: 'checking' }
  | { kind: 'matched'; name: string; score: number }
  | { kind: 'unknown'; score: number };

export interface ReceptionInput {
  faceCount: number;
  peopleCount: number;
  /** Present only on frames where an embedding was computed. */
  match?: Match | null;
}

/** Turns per-frame analyses into what the reception tab shows. Keeps the last result while the face stays in view. */
export class ReceptionState {
  private last: Match | null = null;

  update(input: ReceptionInput): ReceptionView {
    if (input.peopleCount === 0) {
      this.last = null;
      return { kind: 'no-people' };
    }
    if (input.faceCount === 0) {
      this.last = null;
      return { kind: 'no-face' };
    }
    if (input.match) this.last = input.match;
    if (!this.last) return { kind: 'checking' };
    return this.last.kind === 'matched'
      ? { kind: 'matched', name: this.last.name, score: this.last.score }
      : { kind: 'unknown', score: this.last.score };
  }

  reset(): void {
    this.last = null;
  }
}
```

`src/enroll-session.ts`:

```ts
export interface EnrollSample<T> {
  faceCount: number;
  embedding?: Float32Array;
  aligned?: T;
}

export type EnrollStatus<T> =
  | { kind: 'collecting'; count: number; total: number; hint: 'ok' | 'one-face' }
  | { kind: 'done'; embeddings: Float32Array[]; thumbnailSource: T }
  | { kind: 'timeout'; count: number; total: number };

/** Collects `shots` embeddings from frames that show exactly one face, within `timeoutMs`. */
export class EnrollSession<T> {
  private readonly embeddings: Float32Array[] = [];
  private first: T | undefined;
  private readonly shots: number;
  private readonly deadline: number;

  constructor(opts: { shots: number; timeoutMs: number; startedAt: number }) {
    this.shots = opts.shots;
    this.deadline = opts.startedAt + opts.timeoutMs;
  }

  accept(sample: EnrollSample<T>, now: number): EnrollStatus<T> {
    const count = this.embeddings.length;
    if (now > this.deadline) return { kind: 'timeout', count, total: this.shots };
    if (sample.faceCount !== 1) return { kind: 'collecting', count, total: this.shots, hint: 'one-face' };
    if (sample.embedding && sample.aligned !== undefined) {
      this.embeddings.push(sample.embedding);
      this.first ??= sample.aligned;
      if (this.embeddings.length >= this.shots) {
        return { kind: 'done', embeddings: [...this.embeddings], thumbnailSource: this.first };
      }
    }
    return { kind: 'collecting', count: this.embeddings.length, total: this.shots, hint: 'ok' };
  }
}
```

`src/errors.ts`:

```ts
import type { InitFailure } from './worker-protocol.ts';

export type CameraProblem = 'insecure' | 'denied' | 'not-found' | 'other';

export function classifyCameraError(error: unknown, secureContext: boolean): CameraProblem {
  if (!secureContext) return 'insecure';
  const name = error instanceof DOMException || error instanceof Error ? error.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') return 'denied';
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'not-found';
  return 'other';
}

export function cameraProblemMessage(problem: CameraProblem): string {
  switch (problem) {
    case 'insecure':
      return 'カメラは https:// で始まるページでしか使えません。';
    case 'denied':
      return 'カメラの使用が許可されていません。iPhone では「設定」の Safari の項目でカメラを「確認」か「許可」にしてから、もう一度試してください。';
    case 'not-found':
      return 'カメラが見つかりませんでした。';
    case 'other':
      return 'カメラを起動できませんでした。ほかのアプリがカメラを使っていないか確認してください。';
  }
}

export function initFailureMessage(reason: InitFailure): string {
  return reason === 'network'
    ? 'モデルの読み込みに失敗しました。通信環境を確認して、もう一度試してください。'
    : 'このブラウザには対応していません。最新の Safari か Chrome で開いてください。';
}
```

`src/ui/geometry.ts`:

```ts
import type { Face } from '../vision/types.ts';

export type Box = Face['box'];

/**
 * Maps a box from camera-frame pixels to the on-screen video element, which shows the frame with
 * `object-fit: cover` and, when `mirrored`, flipped horizontally.
 */
export function toViewBox(box: Box, frameWidth: number, frameHeight: number, viewWidth: number, viewHeight: number, mirrored: boolean): Box {
  const scale = Math.max(viewWidth / frameWidth, viewHeight / frameHeight);
  const offsetX = (viewWidth - frameWidth * scale) / 2;
  const offsetY = (viewHeight - frameHeight * scale) / 2;
  const width = box.width * scale;
  const left = box.x * scale + offsetX;
  return {
    x: mirrored ? viewWidth - left - width : left,
    y: box.y * scale + offsetY,
    width,
    height: box.height * scale,
  };
}
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `npm test && npx tsc`
Expected: `Test Files  11 passed (11)`、`Tests  59 passed (59)`。`tsc` はエラーなし。

- [ ] **Step 5: コミットする**

```bash
git add src/worker-protocol.ts src/reception-state.ts src/enroll-session.ts src/errors.ts src/ui/geometry.ts tests/unit/reception-state.test.ts tests/unit/enroll-session.test.ts tests/unit/errors.test.ts tests/unit/geometry.test.ts
git commit -F - <<'EOF'
Add reception and enrollment state, error messages, view geometry

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PADPRtudUhYxccwepxBWdj
EOF
```

---

### Task 7: 画面の構造と見た目を決める（ユーザーの選択が必要）

このタスクは、サブエージェントではなくコントローラー（ユーザーと会話しているセッション）が行う。ユーザーが見た目を選ぶまで次に進まない。

**Files:**
- Create: `index.html`, `src/layout.css`, `src/theme.css`, `src/main.ts`（仮置き。Task 8 で置き換える）

**Interfaces:**
- Consumes: なし
- Produces:
  - `index.html` の DOM。以降のタスクと E2E はこの id・role・ラベルに依存する: `#start-screen`, `#start-button`, `#load-status`, `#load-progress`, `#load-message`, `#start-error`, `#start-error-message`, `#retry-button`, `#main-screen[data-tab]`, `#tab-reception`, `#tab-enroll`（role=tab）, `#stage[data-face-count]`, `#camera`, `#overlay`, `#panel-reception`, `#reception-message`, `#reception-score`, `#go-enroll`, `#panel-enroll`, `#enroll-form`, `#enroll-name`（label「名前」）, `#enroll-button`, `#enroll-status`, `#volatile-note`, `#people-list`, `#people-empty`, `#delete-all`
  - `src/theme.css` が `:root` に定義する変数（`src/ui/stage.ts` が canvas の描画に使う）: `--box`（枠の色）, `--box-accent`（照合対象の枠とラベルの背景）, `--label-fg`（ラベルの文字色）, `--label-font`（canvas の `font` 指定。例: `bold 16px sans-serif`）。`src/layout.css` が使う `--bg`, `--fg` も定義する

- [ ] **Step 1: 画面の構造とレイアウトを書く**

`index.html`:

```html
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
    <title>顔認証デモ</title>
    <link rel="icon" href="data:," />
  </head>
  <body>
    <main class="app">
      <section id="start-screen" class="start-screen">
        <h1 class="title">顔認証デモ</h1>
        <p class="lead">名前と顔を登録すると、カメラに映った人が誰かを当てます。顔のデータはこの端末の中にだけ保存され、外部には送られません。</p>
        <button id="start-button" class="button button-primary" type="button">はじめる</button>
        <div id="load-status" class="load-status" hidden>
          <progress id="load-progress" class="load-progress" max="1" value="0"></progress>
          <p id="load-message" class="load-message">準備しています…</p>
        </div>
        <div id="start-error" class="start-error" role="alert" hidden>
          <p id="start-error-message" class="start-error-message"></p>
          <button id="retry-button" class="button" type="button">もう一度試す</button>
        </div>
      </section>

      <section id="main-screen" class="main-screen" data-tab="reception" hidden>
        <nav class="tabs" role="tablist" aria-label="モード">
          <button id="tab-reception" class="tab" role="tab" type="button" aria-selected="true" aria-controls="panel-reception">受付</button>
          <button id="tab-enroll" class="tab" role="tab" type="button" aria-selected="false" aria-controls="panel-enroll">登録</button>
        </nav>
        <div id="stage" class="stage" data-face-count="0">
          <video id="camera" class="camera" playsinline muted autoplay></video>
          <canvas id="overlay" class="overlay"></canvas>
        </div>
        <section id="panel-reception" class="panel panel-reception" role="tabpanel" aria-labelledby="tab-reception">
          <p id="reception-message" class="reception-message" aria-live="polite">カメラに顔を映してください</p>
          <p id="reception-score" class="reception-score"></p>
          <button id="go-enroll" class="button button-primary" type="button" hidden>登録する</button>
        </section>
        <section id="panel-enroll" class="panel panel-enroll" role="tabpanel" aria-labelledby="tab-enroll" hidden>
          <form id="enroll-form" class="enroll-form">
            <label class="enroll-label" for="enroll-name">名前</label>
            <input id="enroll-name" class="enroll-name" name="name" type="text" maxlength="20" autocomplete="off" enterkeyhint="done" placeholder="例: あおやま" />
            <button id="enroll-button" class="button button-primary" type="submit" disabled>撮影</button>
          </form>
          <p id="enroll-status" class="enroll-status" aria-live="polite"></p>
          <p id="volatile-note" class="volatile-note" hidden>この端末には保存できないため、ページを閉じると登録は消えます。</p>
          <h2 class="people-heading">登録した人</h2>
          <ul id="people-list" class="people-list"></ul>
          <p id="people-empty" class="people-empty">まだ誰も登録されていません。</p>
          <button id="delete-all" class="button button-danger" type="button" hidden>全員削除</button>
        </section>
      </section>
    </main>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

`src/layout.css`（配置と大きさだけ。色や書体は持たない）:

```css
/* Layout only. Colors, fonts and component looks live in theme.css. */
*,
*::before,
*::after {
  box-sizing: border-box;
}

[hidden] {
  display: none !important;
}

html,
body {
  margin: 0;
  height: 100%;
}

button {
  touch-action: manipulation;
}

body {
  background: var(--bg, #fff);
  color: var(--fg, #111);
  -webkit-text-size-adjust: 100%;
}

.app {
  height: 100dvh;
  display: flex;
  flex-direction: column;
}

.start-screen {
  flex: 1;
  width: 100%;
  max-width: 480px;
  margin: 0 auto;
  padding: max(24px, env(safe-area-inset-top)) 16px max(24px, env(safe-area-inset-bottom));
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: stretch;
  gap: 16px;
  text-align: center;
}

.load-progress {
  width: 100%;
}

.main-screen {
  flex: 1;
  min-height: 0;
  width: 100%;
  max-width: 720px;
  margin: 0 auto;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  padding-top: env(safe-area-inset-top);
}

.main-screen[data-tab='enroll'] {
  grid-template-rows: auto 40dvh minmax(0, 1fr);
}

.tabs {
  display: flex;
}

.tab {
  flex: 1;
}

.stage {
  position: relative;
  overflow: hidden;
  background: #000;
}

.camera,
.overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

.camera {
  object-fit: cover;
  transform: scaleX(-1);
}

.overlay {
  pointer-events: none;
}

.panel {
  padding: 16px 16px max(16px, env(safe-area-inset-bottom));
}

.panel-enroll {
  overflow-y: auto;
}

.enroll-form {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
}

.enroll-label {
  grid-column: 1 / -1;
}

.enroll-name {
  min-width: 0;
  font-size: 16px; /* 16px or more keeps iOS Safari from zooming into the field */
}

.people-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.person {
  display: grid;
  grid-template-columns: 56px 1fr auto auto;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
}

.person-thumb {
  width: 56px;
  height: 56px;
  object-fit: cover;
}
```

`src/theme.css`（たたき台。Step 4 で選ばれた案に置き換える）:

```css
:root {
  --bg: #fafafa;
  --fg: #111;
  --box: #ffffff;
  --box-accent: #ffd400;
  --label-fg: #000;
  --label-font: bold 16px sans-serif;
  font-family: system-ui, sans-serif;
}
```

`src/main.ts`（仮置き。CSS を読み込むだけ）:

```ts
import './layout.css';
import './theme.css';
```

- [ ] **Step 2: ビルドが通ることを確かめる**

Run: `npm run build`
Expected: `dist/index.html` と `dist/assets/*.css`, `*.js` ができる。エラーなし。

- [ ] **Step 3: 見た目の案を3つ作り、描画したモックアップで見せる**

ユーザーは見た目の選択を、実際に描画したモックアップで行いたい（ASCII の図では判断できない）。また、iPhone の実機で確かめる。

- `artifact-design` スキルを読み込んでから、1つの HTML ページに3案を並べて Artifact として公開する（ユーザーが iPhone でも開ける）。
- 各案とも、次の3画面をスマホ幅（390×844）で見せる: 起動画面、受付で照合できた状態（顔写真に枠とラベル、「あなたは メイア さんですね?」）、登録タブ（名前の入力欄、登録者1人の一覧）。カメラ映像の代わりに `tests/fixtures/faces/meir-a.jpg` を使う。
- 3案は `index.html` の DOM と `src/layout.css` をそのまま使い、テーマの CSS だけを変える。レイアウト（`.app`, `.main-screen`, `.stage`, `.camera`, `.overlay` の配置や大きさ）は変えない。
- 3案は方向性をはっきり変える（例: 明るく落ち着いた受付端末風／暗い背景でカメラ映像が主役／紙の質感や印刷風の遊び）。number_magic のリソグラフ風の見た目と揃える案を1つ入れてもよい。
- 条件: 文字のコントラストは WCAG AA 以上。外部への通信は Google Fonts だけ可（使わないならシステムフォント）。iOS で入力欄がズームしないよう、`.enroll-name` は 16px 以上のまま。ボタンは指で押しやすい高さ（44px 以上）にする。

- [ ] **Step 4: ユーザーに選んでもらい、`src/theme.css` に反映する**

選ばれた案のテーマを `src/theme.css` に書く。Interfaces に挙げた CSS 変数をすべて定義すること。ユーザーが案を組み合わせたり直したりしたいと言ったら、もう一度描画して見せる。

- [ ] **Step 5: 見え方を確かめてコミットする**

Run: `npm run build`
Expected: エラーなし。

ユーザーが選んだ見た目になっているか、`npm run dev` で開いた起動画面を 390×844 で撮って確かめる（受付と登録の画面は Task 9 の後に確かめる）。

```bash
git add index.html src/layout.css src/theme.css src/main.ts
git commit -F - <<'EOF'
Add page structure, layout and the chosen theme

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PADPRtudUhYxccwepxBWdj
EOF
```

---

### Task 8: 推論 Worker・カメラ・起動画面

**Files:**
- Create: `src/worker.ts`, `src/vision-client.ts`, `src/camera.ts`, `src/ui/dom.ts`, `src/ui/stage.ts`, `src/ui/start-screen.ts`
- Modify: `src/main.ts`（Task 7 の仮置きを置き換える）
- Create: `playwright.config.ts`, `tests/e2e/helpers.ts`
- Test: `tests/e2e/startup.spec.ts`

**Interfaces:**
- Consumes: Task 1（定数）、Task 3（`detectorLayout`, `largestFace`）、Task 5（`FacePipeline`）、Task 6（`worker-protocol.ts` の型、`errors.ts`、`toViewBox`）、Task 7（DOM と CSS 変数）
- Produces:
  - `src/vision-client.ts`: `class InitError extends Error { readonly reason: InitFailure }`、`class VisionClient { init(onProgress: (ratio: number) => void): Promise<void>; analyze(frame: ImageBitmap, embed: boolean): Promise<Analysis> }`（`init` は失敗後にもう一度呼べる。`frame` の所有権は Worker に移る）
  - `src/camera.ts`: `openCamera(video: HTMLVideoElement): Promise<void>`（失敗時は getUserMedia の DOMException で reject）、`closeCamera(video: HTMLVideoElement): void`
  - `src/ui/dom.ts`: `byId<T extends HTMLElement>(id: string): T`
  - `src/ui/stage.ts`: `class Stage { constructor(root: HTMLElement, video: HTMLVideoElement, overlay: HTMLCanvasElement, client: VisionClient); openCamera(): Promise<void>; start(): void; stop(): void; requestEmbedding(): void; setLabel(label: string | null): void; onAnalysis(listener: (analysis: Analysis) => void): () => void }`。`#stage` の `data-face-count` に検出数を書く
  - `src/ui/start-screen.ts`: `class StartScreen { onStart(handler): void; onRetry(handler): void; showLoading(): void; setProgress(ratio: number): void; showError(message: string, canRetry: boolean): void; hide(): void }`
  - モデルの読み込みが終わると `document.body.dataset.models = 'ready'` になる（iOS シミュレーターでの確認に使う）
  - `tests/e2e/helpers.ts`: `APP_URL`, `newProfile(): string`, `launchWithFace(profileDir: string, face: string, outDir: string): Promise<BrowserContext>`, `openApp(context: BrowserContext): Promise<Page>`

- [ ] **Step 1: 失敗する E2E を書く**

E2E は `npm run build` したものを `vite preview`（ポート 4173）で配信し、システムの Google Chrome を Playwright から動かす。カメラは Chrome の仮想カメラに顔写真を流す（`.mjpeg` として渡した1枚の JPEG を繰り返し映す）。

`playwright.config.ts`:

```ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 180_000,
  workers: 1,
  reporter: 'list',
  webServer: {
    command: 'npm run build && npm run preview -- --port 4173 --strictPort',
    url: 'http://localhost:4173/face-reception/',
    timeout: 180_000,
    reuseExistingServer: false,
  },
});
```

`tests/e2e/helpers.ts`:

```ts
import { chromium, type BrowserContext, type Page } from '@playwright/test';
import { copyFileSync, mkdirSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const APP_URL = 'http://localhost:4173/face-reception/';

/** A fresh Chrome profile directory. Reusing one across launches keeps IndexedDB, like reopening Safari. */
export function newProfile(): string {
  return mkdtempSync(join(tmpdir(), 'face-reception-'));
}

/**
 * Launches Chrome whose camera shows one face fixture. Chrome reads a .mjpeg file as the fake camera,
 * and a single JPEG is a valid one-frame MJPEG stream.
 */
export async function launchWithFace(profileDir: string, face: string, outDir: string): Promise<BrowserContext> {
  mkdirSync(outDir, { recursive: true });
  const feed = join(outDir, `${face}.mjpeg`);
  copyFileSync(fileURLToPath(new URL(`../fixtures/faces/${face}.jpg`, import.meta.url)), feed);
  return chromium.launchPersistentContext(profileDir, {
    channel: 'chrome',
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${feed}`],
    permissions: ['camera'],
  });
}

export async function openApp(context: BrowserContext): Promise<Page> {
  const page = await context.newPage();
  await page.goto(APP_URL);
  await page.getByRole('button', { name: 'はじめる' }).click();
  return page;
}
```

`tests/e2e/startup.spec.ts`:

```ts
import { chromium, expect, test } from '@playwright/test';
import { APP_URL, launchWithFace, newProfile, openApp } from './helpers.ts';

test('loads the models and draws a box around the face the camera sees', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  const page = await openApp(context);
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await expect(page.locator('#stage')).toHaveAttribute('data-face-count', '1');
  await expect(page.locator('body')).toHaveAttribute('data-models', 'ready');
  await context.close();
});

test('explains how to allow the camera when permission is denied', async () => {
  const browser = await chromium.launch({ channel: 'chrome', args: ['--use-fake-device-for-media-stream'] });
  const page = await (await browser.newContext()).newPage();
  await page.goto(APP_URL);
  await page.getByRole('button', { name: 'はじめる' }).click();
  await expect(page.locator('#start-error-message')).toContainText('カメラの使用が許可されていません', { timeout: 60_000 });
  await expect(page.getByRole('button', { name: 'もう一度試す' })).toBeVisible();
  await browser.close();
});

test('offers a retry when the models fail to download', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  await context.route('**/models/*.onnx', (route) => route.abort());
  const page = await openApp(context);
  await expect(page.locator('#start-error-message')).toContainText('モデルの読み込みに失敗しました', { timeout: 60_000 });
  await context.unroute('**/models/*.onnx');
  await page.getByRole('button', { name: 'もう一度試す' }).click();
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await context.close();
});
```

- [ ] **Step 2: E2E が失敗することを確かめる**

Run: `npm run test:e2e`
Expected: 3件とも FAIL（「はじめる」を押しても何も起きず、`#main-screen` や `#start-error-message` が出ないままタイムアウトする）。

- [ ] **Step 3: Worker とメイン側の窓口を書く**

`src/worker.ts`:

```ts
import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { DETECT_INPUT_LONG_SIDE, DETECT_SCORE_THRESHOLD, MODELS, NMS_IOU_THRESHOLD } from './config.ts';
import { detectorLayout, largestFace } from './vision/detector.ts';
import { FacePipeline } from './vision/pipeline.ts';
import type { Analysis, WorkerRequest, WorkerResponse } from './worker-protocol.ts';

// Serve the wasm binary from our own build output; GitHub Pages cannot send COOP/COEP, so stay single-threaded.
ort.env.wasm.wasmPaths = { wasm: wasmUrl };
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';

let pipeline: FacePipeline | undefined;
let inputCanvas: OffscreenCanvas | undefined;
let frameCanvas: OffscreenCanvas | undefined;

function send(message: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(message, { transfer });
}

async function fetchModels(baseUrl: string): Promise<[Uint8Array, Uint8Array]> {
  const total = MODELS.yunet.bytes + MODELS.sface.bytes;
  let loaded = 0;
  const fetchOne = async (file: string): Promise<Uint8Array> => {
    const res = await fetch(`${baseUrl}${file}`);
    if (!res.ok || !res.body) throw new Error(`${file}: HTTP ${res.status}`);
    const chunks: Uint8Array[] = [];
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      send({ type: 'progress', loaded: Math.min(loaded, total), total });
    }
    const bytes = new Uint8Array(chunks.reduce((n, c) => n + c.byteLength, 0));
    let offset = 0;
    for (const c of chunks) {
      bytes.set(c, offset);
      offset += c.byteLength;
    }
    return bytes;
  };
  return Promise.all([fetchOne(MODELS.yunet.file), fetchOne(MODELS.sface.file)]);
}

async function init(modelBaseUrl: string): Promise<void> {
  if (pipeline) {
    send({ type: 'ready' });
    return;
  }
  if (typeof OffscreenCanvas === 'undefined') {
    send({ type: 'init-error', reason: 'unsupported', message: 'OffscreenCanvas is unavailable' });
    return;
  }
  let models: [Uint8Array, Uint8Array];
  try {
    models = await fetchModels(modelBaseUrl);
  } catch (error) {
    send({ type: 'init-error', reason: 'network', message: String(error) });
    return;
  }
  try {
    pipeline = await FacePipeline.create(models[0], models[1]);
  } catch (error) {
    send({ type: 'init-error', reason: 'unsupported', message: String(error) });
    return;
  }
  send({ type: 'ready' });
}

function sized(canvas: OffscreenCanvas | undefined, width: number, height: number): OffscreenCanvas {
  if (!canvas) return new OffscreenCanvas(width, height);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return canvas;
}

function context2d(canvas: OffscreenCanvas): OffscreenCanvasRenderingContext2D {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D canvas is unavailable');
  return ctx;
}

async function analyze(frame: ImageBitmap, embed: boolean): Promise<{ analysis: Analysis; transfer: Transferable[] }> {
  try {
    if (!pipeline) throw new Error('models are not loaded');
    const { width, height } = frame;
    const layout = detectorLayout(width, height, DETECT_INPUT_LONG_SIDE);
    inputCanvas = sized(inputCanvas, layout.width, layout.height);
    const input = context2d(inputCanvas);
    input.clearRect(0, 0, layout.width, layout.height);
    input.drawImage(frame, 0, 0, width * layout.scale, height * layout.scale);
    const faces = await pipeline.detect(input.getImageData(0, 0, layout.width, layout.height), layout, {
      scoreThreshold: DETECT_SCORE_THRESHOLD,
      iouThreshold: NMS_IOU_THRESHOLD,
    });
    const analysis: Analysis = { faces, frameWidth: width, frameHeight: height };
    const transfer: Transferable[] = [];
    const largest = largestFace(faces);
    if (embed && largest) {
      frameCanvas = sized(frameCanvas, width, height);
      const full = context2d(frameCanvas);
      full.drawImage(frame, 0, 0);
      const { embedding, aligned } = await pipeline.embed(full.getImageData(0, 0, width, height), largest);
      analysis.largest = { embedding, aligned };
      transfer.push(embedding.buffer as ArrayBuffer, aligned.data.buffer as ArrayBuffer);
    }
    return { analysis, transfer };
  } finally {
    frame.close();
  }
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  if (message.type === 'init') {
    void init(message.modelBaseUrl);
    return;
  }
  analyze(message.frame, message.embed).then(
    ({ analysis, transfer }) => send({ type: 'analysis', id: message.id, analysis }, transfer),
    (error: unknown) => send({ type: 'analyze-error', id: message.id, message: String(error) }),
  );
};
```

`src/vision-client.ts`:

```ts
import type { Analysis, InitFailure, WorkerRequest, WorkerResponse } from './worker-protocol.ts';

export class InitError extends Error {
  readonly reason: InitFailure;

  constructor(reason: InitFailure, message: string) {
    super(message);
    this.reason = reason;
  }
}

interface Pending {
  resolve: (analysis: Analysis) => void;
  reject: (error: Error) => void;
}

interface InitWaiter {
  resolve: () => void;
  reject: (error: Error) => void;
  onProgress: (ratio: number) => void;
}

/** Main-thread handle to the inference worker. */
export class VisionClient {
  private readonly worker: Worker;
  private readonly pending = new Map<number, Pending>();
  private nextId = 1;
  private initWaiter: InitWaiter | undefined;

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handle(event.data);
    this.worker.onerror = (event) => {
      const error = new InitError('unsupported', event.message || 'worker failed');
      this.initWaiter?.reject(error);
      this.initWaiter = undefined;
      for (const p of this.pending.values()) p.reject(error);
      this.pending.clear();
    };
  }

  /** Downloads the models and creates the ONNX sessions. Safe to call again after a failure. */
  init(onProgress: (ratio: number) => void): Promise<void> {
    return new Promise((resolve, reject) => {
      this.initWaiter = { resolve, reject, onProgress };
      this.post({ type: 'init', modelBaseUrl: `${import.meta.env.BASE_URL}models/` });
    });
  }

  /** Detects faces in `frame` (ownership moves to the worker). With `embed`, also embeds the largest face. */
  analyze(frame: ImageBitmap, embed: boolean): Promise<Analysis> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.post({ type: 'analyze', id, frame, embed }, [frame]);
    });
  }

  private post(message: WorkerRequest, transfer: Transferable[] = []): void {
    this.worker.postMessage(message, transfer);
  }

  private handle(message: WorkerResponse): void {
    switch (message.type) {
      case 'progress':
        this.initWaiter?.onProgress(message.loaded / message.total);
        break;
      case 'ready':
        this.initWaiter?.resolve();
        this.initWaiter = undefined;
        break;
      case 'init-error':
        this.initWaiter?.reject(new InitError(message.reason, message.message));
        this.initWaiter = undefined;
        break;
      case 'analysis':
        this.pending.get(message.id)?.resolve(message.analysis);
        this.pending.delete(message.id);
        break;
      case 'analyze-error':
        this.pending.get(message.id)?.reject(new Error(message.message));
        this.pending.delete(message.id);
        break;
    }
  }
}
```

- [ ] **Step 4: カメラ、プレビュー、起動画面を書く**

`src/camera.ts`:

```ts
/** Starts the front camera in `video`. Rejects with the getUserMedia DOMException on failure. */
export async function openCamera(video: HTMLVideoElement): Promise<void> {
  if (!navigator.mediaDevices?.getUserMedia) throw new DOMException('getUserMedia is unavailable', 'NotFoundError');
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
  });
  video.srcObject = stream;
  await video.play();
}

export function closeCamera(video: HTMLVideoElement): void {
  const stream = video.srcObject;
  if (stream instanceof MediaStream) for (const track of stream.getTracks()) track.stop();
  video.srcObject = null;
}
```

`src/ui/dom.ts`:

```ts
export function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`#${id} not found`);
  return element as T;
}
```

`src/ui/stage.ts`:

```ts
import { closeCamera, openCamera } from '../camera.ts';
import type { VisionClient } from '../vision-client.ts';
import { largestFace } from '../vision/detector.ts';
import type { Analysis } from '../worker-protocol.ts';
import { toViewBox } from './geometry.ts';

type Listener = (analysis: Analysis) => void;

const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** Camera preview with face boxes drawn on top. Runs the detection loop and hands each analysis to listeners. */
export class Stage {
  private readonly root: HTMLElement;
  private readonly video: HTMLVideoElement;
  private readonly overlay: HTMLCanvasElement;
  private readonly client: VisionClient;
  private readonly listeners = new Set<Listener>();
  private running = false;
  private generation = 0;
  private embedRequested = false;
  private label: string | null = null;
  private last: Analysis | null = null;

  constructor(root: HTMLElement, video: HTMLVideoElement, overlay: HTMLCanvasElement, client: VisionClient) {
    this.root = root;
    this.video = video;
    this.overlay = overlay;
    this.client = client;
  }

  openCamera(): Promise<void> {
    return openCamera(this.video);
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    void this.loop(++this.generation);
  }

  /** Stops the loop and the camera. */
  stop(): void {
    this.running = false;
    this.generation++;
    closeCamera(this.video);
    this.last = null;
    this.root.dataset.faceCount = '0';
    this.draw();
  }

  /** The next analyzed frame will include the embedding of its largest face. */
  requestEmbedding(): void {
    this.embedRequested = true;
  }

  /** Text drawn above the largest face, or null for none. */
  setLabel(label: string | null): void {
    this.label = label;
    this.draw();
  }

  onAnalysis(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private async loop(generation: number): Promise<void> {
    while (generation === this.generation) {
      if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || this.video.videoWidth === 0) {
        await nextFrame();
        continue;
      }
      const embed = this.embedRequested;
      this.embedRequested = false;
      try {
        const analysis = await this.client.analyze(await createImageBitmap(this.video), embed);
        if (generation !== this.generation) break;
        this.last = analysis;
        this.root.dataset.faceCount = String(analysis.faces.length);
        this.draw();
        for (const listener of this.listeners) listener(analysis);
      } catch (error) {
        console.error(error);
        if (embed) this.embedRequested = true;
        await sleep(200);
      }
      await nextFrame();
    }
  }

  private draw(): void {
    const canvas = this.overlay;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const a = this.last;
    if (!a) return;
    const style = getComputedStyle(canvas);
    const color = style.getPropertyValue('--box').trim() || '#ffffff';
    const accent = style.getPropertyValue('--box-accent').trim() || '#ffd400';
    const labelColor = style.getPropertyValue('--label-fg').trim() || '#000000';
    const largest = largestFace(a.faces);
    for (const face of a.faces) {
      const b = toViewBox(face.box, a.frameWidth, a.frameHeight, width, height, true);
      const isLargest = face === largest;
      ctx.strokeStyle = isLargest ? accent : color;
      ctx.lineWidth = isLargest ? 3 : 2;
      ctx.strokeRect(b.x, b.y, b.width, b.height);
      if (isLargest && this.label) {
        ctx.font = style.getPropertyValue('--label-font').trim() || 'bold 16px sans-serif';
        const padding = 6;
        const labelHeight = 26;
        const labelWidth = ctx.measureText(this.label).width + padding * 2;
        const top = Math.max(0, b.y - labelHeight - 4);
        ctx.fillStyle = accent;
        ctx.fillRect(b.x, top, labelWidth, labelHeight);
        ctx.fillStyle = labelColor;
        ctx.textBaseline = 'middle';
        ctx.fillText(this.label, b.x + padding, top + labelHeight / 2);
      }
    }
  }
}
```

`src/ui/start-screen.ts`:

```ts
import { byId } from './dom.ts';

export class StartScreen {
  private readonly root = byId<HTMLElement>('start-screen');
  private readonly startButton = byId<HTMLButtonElement>('start-button');
  private readonly status = byId<HTMLElement>('load-status');
  private readonly progress = byId<HTMLProgressElement>('load-progress');
  private readonly message = byId<HTMLElement>('load-message');
  private readonly error = byId<HTMLElement>('start-error');
  private readonly errorMessage = byId<HTMLElement>('start-error-message');
  private readonly retryButton = byId<HTMLButtonElement>('retry-button');

  onStart(handler: () => void): void {
    this.startButton.addEventListener('click', handler);
  }

  onRetry(handler: () => void): void {
    this.retryButton.addEventListener('click', handler);
  }

  showLoading(): void {
    this.root.hidden = false;
    this.startButton.hidden = true;
    this.error.hidden = true;
    this.status.hidden = false;
  }

  setProgress(ratio: number): void {
    this.progress.value = ratio;
    this.message.textContent = `モデルを読み込んでいます… ${Math.floor(ratio * 100)}%`;
  }

  showError(message: string, canRetry: boolean): void {
    this.root.hidden = false;
    this.startButton.hidden = true;
    this.status.hidden = true;
    this.error.hidden = false;
    this.errorMessage.textContent = message;
    this.retryButton.hidden = !canRetry;
  }

  hide(): void {
    this.root.hidden = true;
  }
}
```

`src/main.ts`（Task 7 の仮置きを置き換える。タブと各画面は Task 9 で入れる）:

```ts
import './layout.css';
import './theme.css';
import { cameraProblemMessage, classifyCameraError, initFailureMessage } from './errors.ts';
import { byId } from './ui/dom.ts';
import { Stage } from './ui/stage.ts';
import { StartScreen } from './ui/start-screen.ts';
import { InitError, VisionClient } from './vision-client.ts';

const startScreen = new StartScreen();
const mainScreen = byId<HTMLElement>('main-screen');
const client = new VisionClient();
const stage = new Stage(byId('stage'), byId('camera'), byId('overlay'), client);

let modelsReady = false;
let cameraReady = false;

async function boot(): Promise<void> {
  startScreen.showLoading();
  const [camera, models] = await Promise.allSettled([
    cameraReady ? Promise.resolve() : stage.openCamera(),
    modelsReady ? Promise.resolve() : client.init((ratio) => startScreen.setProgress(ratio)),
  ]);
  if (camera.status === 'fulfilled') cameraReady = true;
  if (models.status === 'fulfilled') {
    modelsReady = true;
    document.body.dataset.models = 'ready';
  }
  if (models.status === 'rejected') {
    const reason = models.reason instanceof InitError ? models.reason.reason : 'unsupported';
    startScreen.showError(initFailureMessage(reason), reason === 'network');
    return;
  }
  if (camera.status === 'rejected') {
    startScreen.showError(cameraProblemMessage(classifyCameraError(camera.reason, window.isSecureContext)), true);
    return;
  }
  startScreen.hide();
  mainScreen.hidden = false;
  stage.start();
}

startScreen.onStart(() => void boot());
startScreen.onRetry(() => void boot());
```

- [ ] **Step 5: E2E と単体テストが通ることを確かめる**

Run: `npx tsc && npm test && npm run test:e2e`
Expected: 単体テスト `Tests  59 passed (59)`、E2E `3 passed`。

- [ ] **Step 6: コミットする**

```bash
git add src/worker.ts src/vision-client.ts src/camera.ts src/ui/dom.ts src/ui/stage.ts src/ui/start-screen.ts src/main.ts playwright.config.ts tests/e2e/helpers.ts tests/e2e/startup.spec.ts
git commit -F - <<'EOF'
Run face detection in a Web Worker with camera preview and start screen

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PADPRtudUhYxccwepxBWdj
EOF
```

---

### Task 9: 受付と登録の画面

**Files:**
- Create: `src/ui/reception-panel.ts`, `src/ui/enroll-panel.ts`, `src/ui/thumbnail.ts`
- Modify: `src/main.ts`（Task 8 の版を置き換える）
- Test: `tests/e2e/reception.spec.ts`

**Interfaces:**
- Consumes: Task 1（定数）、Task 4（`matchFace`, `PeopleStore`, `Person`, `openPeopleStore`, `createMemoryPeopleStore`）、Task 6（`ReceptionState`, `EnrollSession`, エラー文言）、Task 8（`Stage`, `StartScreen`, `VisionClient`, `byId`）
- Produces:
  - `src/ui/reception-panel.ts`: `class ReceptionPanel { constructor(stage: Stage, getPeople: () => Person[], onGoEnroll: () => void); activate(): void; deactivate(): void }`
  - `src/ui/enroll-panel.ts`: `class EnrollPanel { constructor(stage: Stage, store: PeopleStore, volatile: boolean, onPeopleChanged: (people: Person[]) => void); activate(): void; deactivate(): void; renderPeople(people: Person[]): void }`
  - `src/ui/thumbnail.ts`: `toJpegBlob(image: RgbaImage): Promise<Blob>`
  - 画面の文言（E2E が照合する）: 「あなたは {名前} さんですね?」「登録されていません」「カメラに顔を映してください」「確認しています…」「まずは登録してください」「{名前} さんを登録しました。」「{n}枚」「{名前} さんを削除」（削除ボタンの aria-label）「全員削除」

- [ ] **Step 1: 失敗する E2E を書く**

`tests/e2e/reception.spec.ts`:

```ts
import { expect, test, type Page } from '@playwright/test';
import { launchWithFace, newProfile, openApp } from './helpers.ts';

async function enroll(page: Page, name: string): Promise<void> {
  await page.getByRole('tab', { name: '登録' }).click();
  await page.getByLabel('名前').fill(name);
  await page.getByRole('button', { name: '撮影' }).click();
  await expect(page.locator('#enroll-status')).toHaveText(`${name} さんを登録しました。`, { timeout: 20_000 });
}

async function setHidden(page: Page, hidden: boolean): Promise<void> {
  await page.evaluate((value) => {
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => value });
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (value ? 'hidden' : 'visible') });
    document.dispatchEvent(new Event('visibilitychange'));
  }, hidden);
}

test('enrolls a face, recognizes another photo of the same person after a restart, and rejects someone else', async ({}, testInfo) => {
  const profile = newProfile();
  const out = testInfo.outputDir;

  let context = await launchWithFace(profile, 'meir-a', out);
  let page = await openApp(context);
  await expect(page.getByRole('tab', { name: '登録' })).toHaveAttribute('aria-selected', 'true', { timeout: 60_000 });
  await enroll(page, 'メイア');
  await expect(page.locator('.person-name')).toHaveText(['メイア']);
  await expect(page.locator('.person-count')).toHaveText(['5枚']);
  await page.getByRole('tab', { name: '受付' }).click();
  await expect(page.locator('#reception-message')).toHaveText('あなたは メイア さんですね?');
  await context.close();

  context = await launchWithFace(profile, 'meir-b', out);
  page = await openApp(context);
  await expect(page.getByRole('tab', { name: '受付' })).toHaveAttribute('aria-selected', 'true', { timeout: 60_000 });
  await expect(page.locator('#reception-message')).toHaveText('あなたは メイア さんですね?');
  await context.close();

  context = await launchWithFace(profile, 'kim', out);
  page = await openApp(context);
  await expect(page.locator('#reception-message')).toHaveText('登録されていません', { timeout: 60_000 });
  await context.close();
});

test('adds photos to an existing name and deletes people', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  const page = await openApp(context);
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await enroll(page, 'キム');
  await enroll(page, 'キム');
  await enroll(page, 'キム2');
  await expect(page.locator('.person-name')).toHaveText(['キム', 'キム2']);
  await expect(page.locator('.person-count')).toHaveText(['10枚', '5枚']);
  await page.getByRole('button', { name: 'キム さんを削除' }).click();
  await expect(page.locator('.person-name')).toHaveText(['キム2']);
  page.once('dialog', (dialog) => void dialog.accept());
  await page.getByRole('button', { name: '全員削除' }).click();
  await expect(page.locator('.person-name')).toHaveCount(0);
  await expect(page.locator('#people-empty')).toBeVisible();
  await page.getByRole('tab', { name: '受付' }).click();
  await expect(page.locator('#reception-message')).toHaveText('まずは登録してください');
  await context.close();
});

test('shows names as plain text, never as HTML', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  const page = await openApp(context);
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await enroll(page, '<b>キム</b>');
  await expect(page.locator('.person-name')).toHaveText(['<b>キム</b>']);
  await page.getByRole('tab', { name: '受付' }).click();
  await expect(page.locator('#reception-message')).toHaveText('あなたは <b>キム</b> さんですね?');
  await expect(page.locator('#main-screen b')).toHaveCount(0);
  await context.close();
});

test('releases the camera while the page is in the background and resumes when it returns', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  const page = await openApp(context);
  await expect(page.locator('#stage')).toHaveAttribute('data-face-count', '1', { timeout: 60_000 });
  await setHidden(page, true);
  await expect(page.locator('#stage')).toHaveAttribute('data-face-count', '0');
  expect(await page.locator('#camera').evaluate((video: HTMLVideoElement) => video.srcObject)).toBeNull();
  await setHidden(page, false);
  await expect(page.locator('#stage')).toHaveAttribute('data-face-count', '1');
  await context.close();
});
```

- [ ] **Step 2: E2E が失敗することを確かめる**

Run: `npm run test:e2e -- tests/e2e/reception.spec.ts`
Expected: 4件とも FAIL（タブが切り替わらない、「撮影」が押せない等でタイムアウト）。

- [ ] **Step 3: 実装する**

`src/ui/thumbnail.ts`:

```ts
import type { RgbaImage } from '../vision/types.ts';

export function toJpegBlob(image: RgbaImage): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return Promise.reject(new Error('2D canvas is unavailable'));
  ctx.putImageData(new ImageData(new Uint8ClampedArray(image.data), image.width, image.height), 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('JPEG encoding failed'))), 'image/jpeg', 0.9);
  });
}
```

`src/ui/reception-panel.ts`:

```ts
import { MATCH_THRESHOLD, RECOGNIZE_INTERVAL_MS } from '../config.ts';
import { matchFace } from '../match.ts';
import { ReceptionState, type ReceptionView } from '../reception-state.ts';
import type { Person } from '../store.ts';
import type { Analysis } from '../worker-protocol.ts';
import { byId } from './dom.ts';
import type { Stage } from './stage.ts';

export class ReceptionPanel {
  private readonly message = byId<HTMLElement>('reception-message');
  private readonly score = byId<HTMLElement>('reception-score');
  private readonly goEnroll = byId<HTMLButtonElement>('go-enroll');
  private readonly state = new ReceptionState();
  private readonly stage: Stage;
  private readonly getPeople: () => Person[];
  private timer: ReturnType<typeof setInterval> | undefined;
  private unsubscribe: (() => void) | undefined;

  constructor(stage: Stage, getPeople: () => Person[], onGoEnroll: () => void) {
    this.stage = stage;
    this.getPeople = getPeople;
    this.goEnroll.addEventListener('click', onGoEnroll);
  }

  activate(): void {
    this.state.reset();
    this.render(this.state.update({ faceCount: 0, peopleCount: this.getPeople().length }));
    this.unsubscribe = this.stage.onAnalysis((analysis) => this.handle(analysis));
    this.stage.requestEmbedding();
    this.timer = setInterval(() => this.stage.requestEmbedding(), RECOGNIZE_INTERVAL_MS);
  }

  deactivate(): void {
    clearInterval(this.timer);
    this.unsubscribe?.();
    this.stage.setLabel(null);
  }

  private handle(analysis: Analysis): void {
    const people = this.getPeople();
    const match = analysis.largest ? matchFace(analysis.largest.embedding, people, MATCH_THRESHOLD) : undefined;
    this.render(this.state.update({ faceCount: analysis.faces.length, peopleCount: people.length, match }));
  }

  private render(view: ReceptionView): void {
    this.goEnroll.hidden = view.kind !== 'no-people';
    this.score.textContent = view.kind === 'matched' || view.kind === 'unknown' ? `類似度 ${view.score.toFixed(2)}` : '';
    switch (view.kind) {
      case 'no-people':
        this.message.textContent = 'まずは登録してください';
        this.stage.setLabel(null);
        break;
      case 'no-face':
        this.message.textContent = 'カメラに顔を映してください';
        this.stage.setLabel(null);
        break;
      case 'checking':
        this.message.textContent = '確認しています…';
        this.stage.setLabel(null);
        break;
      case 'matched': {
        const name = document.createElement('strong');
        name.className = 'reception-name';
        name.textContent = view.name;
        this.message.replaceChildren('あなたは ', name, ' さんですね?');
        this.stage.setLabel(`${view.name} さん`);
        break;
      }
      case 'unknown':
        this.message.textContent = '登録されていません';
        this.stage.setLabel('未登録');
        break;
    }
  }
}
```

`src/ui/enroll-panel.ts`:

```ts
import { ENROLL_INTERVAL_MS, ENROLL_SHOTS, ENROLL_TIMEOUT_MS } from '../config.ts';
import { EnrollSession, type EnrollStatus } from '../enroll-session.ts';
import type { PeopleStore, Person } from '../store.ts';
import type { RgbaImage } from '../vision/types.ts';
import { byId } from './dom.ts';
import type { Stage } from './stage.ts';
import { toJpegBlob } from './thumbnail.ts';

export class EnrollPanel {
  private readonly form = byId<HTMLFormElement>('enroll-form');
  private readonly nameInput = byId<HTMLInputElement>('enroll-name');
  private readonly button = byId<HTMLButtonElement>('enroll-button');
  private readonly status = byId<HTMLElement>('enroll-status');
  private readonly volatileNote = byId<HTMLElement>('volatile-note');
  private readonly list = byId<HTMLUListElement>('people-list');
  private readonly empty = byId<HTMLElement>('people-empty');
  private readonly deleteAllButton = byId<HTMLButtonElement>('delete-all');
  private readonly stage: Stage;
  private readonly store: PeopleStore;
  private readonly onPeopleChanged: (people: Person[]) => void;
  private objectUrls: string[] = [];
  private cancelCapture: (() => void) | undefined;

  constructor(stage: Stage, store: PeopleStore, volatile: boolean, onPeopleChanged: (people: Person[]) => void) {
    this.stage = stage;
    this.store = store;
    this.onPeopleChanged = onPeopleChanged;
    this.volatileNote.hidden = !volatile;
    this.nameInput.addEventListener('input', () => this.updateButton());
    this.form.addEventListener('submit', (event) => {
      event.preventDefault();
      this.startCapture(this.nameInput.value.trim());
    });
    this.deleteAllButton.addEventListener('click', () => void this.deleteAll());
  }

  activate(): void {
    this.status.textContent = '';
    this.updateButton();
  }

  deactivate(): void {
    this.cancelCapture?.();
  }

  renderPeople(people: Person[]): void {
    for (const url of this.objectUrls) URL.revokeObjectURL(url);
    this.objectUrls = [];
    this.list.replaceChildren(...people.map((person) => this.personItem(person)));
    this.empty.hidden = people.length > 0;
    this.deleteAllButton.hidden = people.length === 0;
  }

  private personItem(person: Person): HTMLLIElement {
    const item = document.createElement('li');
    item.className = 'person';
    const thumb = document.createElement('img');
    thumb.className = 'person-thumb';
    thumb.alt = '';
    thumb.width = 56;
    thumb.height = 56;
    const url = URL.createObjectURL(person.thumbnail);
    this.objectUrls.push(url);
    thumb.src = url;
    const name = document.createElement('span');
    name.className = 'person-name';
    name.textContent = person.name;
    const count = document.createElement('span');
    count.className = 'person-count';
    count.textContent = `${person.embeddings.length}枚`;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'button person-delete';
    remove.textContent = '削除';
    remove.setAttribute('aria-label', `${person.name} さんを削除`);
    remove.addEventListener('click', () => void this.deletePerson(person.id));
    item.append(thumb, name, count, remove);
    return item;
  }

  private updateButton(): void {
    this.button.disabled = this.cancelCapture !== undefined || this.nameInput.value.trim() === '';
  }

  private startCapture(name: string): void {
    if (!name || this.cancelCapture) return;
    const session = new EnrollSession<RgbaImage>({ shots: ENROLL_SHOTS, timeoutMs: ENROLL_TIMEOUT_MS, startedAt: performance.now() });
    const handle = (status: EnrollStatus<RgbaImage>) => {
      switch (status.kind) {
        case 'collecting':
          this.status.textContent =
            status.hint === 'one-face'
              ? `1人だけ映ってください（${status.count} / ${status.total}）`
              : `少しずつ顔の向きを変えてください（${status.count} / ${status.total}）`;
          break;
        case 'timeout':
          stop();
          this.status.textContent = `時間内に撮影できませんでした（${status.count} / ${status.total}）。もう一度試してください。`;
          break;
        case 'done':
          stop();
          void this.save(name, status.embeddings, status.thumbnailSource);
          break;
      }
    };
    const unsubscribe = this.stage.onAnalysis((analysis) =>
      handle(
        session.accept(
          { faceCount: analysis.faces.length, embedding: analysis.largest?.embedding, aligned: analysis.largest?.aligned },
          performance.now(),
        ),
      ),
    );
    const timer = setInterval(() => this.stage.requestEmbedding(), ENROLL_INTERVAL_MS);
    const timeout = setTimeout(() => handle(session.accept({ faceCount: 0 }, performance.now())), ENROLL_TIMEOUT_MS + 50);
    const stop = () => {
      unsubscribe();
      clearInterval(timer);
      clearTimeout(timeout);
      this.cancelCapture = undefined;
      this.updateButton();
    };
    this.cancelCapture = () => {
      stop();
      this.status.textContent = '撮影を中止しました。';
    };
    this.status.textContent = `少しずつ顔の向きを変えてください（0 / ${ENROLL_SHOTS}）`;
    this.stage.requestEmbedding();
    this.updateButton();
  }

  private async save(name: string, embeddings: Float32Array[], aligned: RgbaImage): Promise<void> {
    try {
      const thumbnail = await toJpegBlob(aligned);
      await this.store.addEnrollment(name, embeddings, thumbnail);
      this.onPeopleChanged(await this.store.listPeople());
      this.nameInput.value = '';
      this.updateButton();
      this.status.textContent = `${name} さんを登録しました。`;
    } catch (error) {
      console.error(error);
      this.status.textContent = '保存できませんでした。';
    }
  }

  private async deletePerson(id: string): Promise<void> {
    await this.store.deletePerson(id);
    this.onPeopleChanged(await this.store.listPeople());
  }

  private async deleteAll(): Promise<void> {
    if (!window.confirm('登録した人をすべて削除しますか？')) return;
    await this.store.deleteAll();
    this.onPeopleChanged(await this.store.listPeople());
  }
}
```

`src/main.ts`（Task 8 の版を置き換える）:

```ts
import './layout.css';
import './theme.css';
import { cameraProblemMessage, classifyCameraError, initFailureMessage } from './errors.ts';
import { createMemoryPeopleStore, openPeopleStore, type PeopleStore, type Person } from './store.ts';
import { byId } from './ui/dom.ts';
import { EnrollPanel } from './ui/enroll-panel.ts';
import { ReceptionPanel } from './ui/reception-panel.ts';
import { Stage } from './ui/stage.ts';
import { StartScreen } from './ui/start-screen.ts';
import { InitError, VisionClient } from './vision-client.ts';

type Tab = 'reception' | 'enroll';
const TABS: Tab[] = ['reception', 'enroll'];

const startScreen = new StartScreen();
const mainScreen = byId<HTMLElement>('main-screen');
const tabButtons: Record<Tab, HTMLButtonElement> = { reception: byId('tab-reception'), enroll: byId('tab-enroll') };
const tabPanels: Record<Tab, HTMLElement> = { reception: byId('panel-reception'), enroll: byId('panel-enroll') };

const client = new VisionClient();
const stage = new Stage(byId('stage'), byId('camera'), byId('overlay'), client);
const storeReady = openStore();

let people: Person[] = [];
let modelsReady = false;
let cameraReady = false;
let panels: Record<Tab, ReceptionPanel | EnrollPanel> | undefined;
let activeTab: Tab | undefined;

async function openStore(): Promise<{ store: PeopleStore; volatile: boolean }> {
  try {
    return { store: await openPeopleStore(), volatile: false };
  } catch (error) {
    console.error(error);
    return { store: createMemoryPeopleStore(), volatile: true };
  }
}

async function boot(): Promise<void> {
  startScreen.showLoading();
  const [camera, models] = await Promise.allSettled([
    cameraReady ? Promise.resolve() : stage.openCamera(),
    modelsReady ? Promise.resolve() : client.init((ratio) => startScreen.setProgress(ratio)),
  ]);
  if (camera.status === 'fulfilled') cameraReady = true;
  if (models.status === 'fulfilled') {
    modelsReady = true;
    document.body.dataset.models = 'ready';
  }
  if (models.status === 'rejected') {
    const reason = models.reason instanceof InitError ? models.reason.reason : 'unsupported';
    startScreen.showError(initFailureMessage(reason), reason === 'network');
    return;
  }
  if (camera.status === 'rejected') {
    startScreen.showError(cameraProblemMessage(classifyCameraError(camera.reason, window.isSecureContext)), true);
    return;
  }
  if (!panels) panels = await createPanels();
  startScreen.hide();
  mainScreen.hidden = false;
  stage.start();
  if (activeTab) panels[activeTab].activate();
  else selectTab(people.length === 0 ? 'enroll' : 'reception');
}

async function createPanels(): Promise<Record<Tab, ReceptionPanel | EnrollPanel>> {
  const { store, volatile } = await storeReady;
  people = await store.listPeople().catch(() => []);
  const enroll = new EnrollPanel(stage, store, volatile, (list) => {
    people = list;
    enroll.renderPeople(list);
  });
  enroll.renderPeople(people);
  const reception = new ReceptionPanel(stage, () => people, () => selectTab('enroll'));
  for (const tab of TABS) tabButtons[tab].addEventListener('click', () => selectTab(tab));
  return { reception, enroll };
}

function selectTab(tab: Tab): void {
  if (!panels || activeTab === tab) return;
  if (activeTab) panels[activeTab].deactivate();
  activeTab = tab;
  mainScreen.dataset.tab = tab;
  for (const t of TABS) {
    tabButtons[t].setAttribute('aria-selected', String(t === tab));
    tabPanels[t].hidden = t !== tab;
  }
  panels[tab].activate();
}

// iOS stops the camera in the background; release it ourselves and reopen when the page comes back.
document.addEventListener('visibilitychange', () => {
  if (!panels || !activeTab || mainScreen.hidden) return;
  if (document.hidden) {
    panels[activeTab].deactivate();
    stage.stop();
    cameraReady = false;
    return;
  }
  stage.openCamera().then(
    () => {
      cameraReady = true;
      stage.start();
      if (panels && activeTab) panels[activeTab].activate();
    },
    (error: unknown) => {
      mainScreen.hidden = true;
      startScreen.showError(cameraProblemMessage(classifyCameraError(error, window.isSecureContext)), true);
    },
  );
});

startScreen.onStart(() => void boot());
startScreen.onRetry(() => void boot());
```

- [ ] **Step 4: すべてのテストが通ることを確かめる**

Run: `npx tsc && npm test && npm run test:e2e`
Expected: 単体テスト `Tests  59 passed (59)`、E2E `7 passed`。

- [ ] **Step 5: 見え方を確かめる**

Task 7 で選んだ見た目が受付と登録の画面にも効いているか、390×844 で撮って確かめる。Chrome の仮想カメラに `meir-a` を流して登録し、受付で照合できた状態と、登録タブの一覧を撮る（`tests/e2e/helpers.ts` の `launchWithFace` に `viewport: { width: 390, height: 844 }` を渡した一時スクリプトでよい。スクリプトはコミットしない）。崩れていたら `src/theme.css` を直し、E2E をもう一度流す。

- [ ] **Step 6: コミットする**

```bash
git add src/ui/reception-panel.ts src/ui/enroll-panel.ts src/ui/thumbnail.ts src/main.ts tests/e2e/reception.spec.ts src/theme.css
git commit -F - <<'EOF'
Add reception and enrollment tabs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PADPRtudUhYxccwepxBWdj
EOF
```

---

### Task 10: README・公開・実機確認

このタスクの GitHub への操作（リポジトリ作成、push、Pages の有効化）は外部に公開する操作なので、実行前にユーザーの確認を取る。

**Files:**
- Create: `README.md`, `.github/workflows/deploy.yml`, `scripts/ios-sim-check.mjs`

**Interfaces:**
- Consumes: Task 1〜9 のすべて
- Produces: `https://re-yura.github.io/face-reception/` で動くアプリ

- [ ] **Step 1: README、デプロイ設定、iOS シミュレーター確認スクリプトを書く**

`README.md`:

````markdown
# 顔認証デモ（face-reception）

ブラウザだけで動く顔認証デモです。名前と顔を登録すると、カメラに映った人が誰かを当てます。
[RE-yura/attendance_manager](https://github.com/RE-yura/attendance_manager)（2020年、PyQt5 + PyTorch）をブラウザ向けに作り直しました。

公開ページ: https://re-yura.github.io/face-reception/

## 使い方

1. ページを開いて「はじめる」を押し、カメラを許可します。初回は約 39MB のモデルを読み込みます。
2. 「登録」タブで名前を入れて「撮影」を押します。少しずつ顔の向きを変えると 5 枚撮れて、登録が終わります。
3. 「受付」タブでカメラに顔を映すと「あなたは ○○ さんですね?」と出ます。登録していない人なら「登録されていません」と出ます。

## データの扱い

- 顔の特徴量とサムネイルは、この端末のブラウザ（IndexedDB）にだけ保存されます。外部には一切送りません。
- 「登録」タブから 1 人ずつ、または全員まとめて削除できます。

## 制約

- 写真をカメラにかざしても通ってしまいます。なりすまし対策はしていません。
- 照合は、端末ごとに登録した人の中からだけ行います。端末間で登録は共有されません。

## 仕組み

```
カメラ映像 → YuNet（顔検出: 枠 + 目・鼻・口の 5 点）→ 5 点を基準位置に合わせて 112×112 に切り出し
          → SFace（128 次元の特徴量）→ 登録者とのコサイン類似度が 0.363 以上なら本人
```

推論は ONNX Runtime Web（wasm、1 スレッド）を Web Worker で動かしています。

## 開発

```bash
npm install
npm run dev          # http://localhost:5173/face-reception/
npm run dev:phone    # 同じ LAN の iPhone から https で確認（自己署名証明書の警告を許可して進む）
npm test             # 単体テスト（Vitest）
npm run test:e2e     # E2E テスト（Playwright。Google Chrome が必要）
npm run build
```

- モデルの取り直し: `scripts/fetch-models.sh`
- テスト用の顔写真の取り直し: `node scripts/fetch-fixtures.mjs`
- OpenCV の基準値の作り直し: `uvx --with opencv-python-headless==5.0.0.93 --with numpy python scripts/opencv_reference.py`

`main` に push すると、GitHub Actions がテストとビルドをして GitHub Pages に公開します。

## ライセンス表記

- YuNet（顔検出モデル）: MIT License — `public/models/LICENSE-yunet.txt`
- SFace（特徴量モデル）: Apache License 2.0 — `public/models/LICENSE-sface.txt`
- テスト用の顔写真: NASA の公式写真（パブリックドメイン）— `tests/fixtures/faces/README.md`
````

`.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-node@v7
        with:
          node-version: 24
          cache: npm
      - run: npm ci
      - run: npm test
      - run: npm run build
      - uses: actions/configure-pages@v6
      - uses: actions/upload-pages-artifact@v5
        with:
          path: dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v5
```

`scripts/ios-sim-check.mjs`:

```js
// Opens the app in iOS Simulator Safari through safaridriver, presses はじめる and waits for the models to load.
// The simulator has no usable camera, so the app never gets past the camera step; this only checks that
// onnxruntime-web downloads the models and creates its sessions inside iOS WebKit.
// Usage: node ios-sim-check.mjs <app url> <simulator udid> [safaridriver port]
const [url, udid, port = '4444'] = process.argv.slice(2);
if (!url || !udid) throw new Error('usage: node ios-sim-check.mjs <app url> <simulator udid> [port]');
const base = `http://localhost:${port}`;
const call = async (method, path, body) => {
  const res = await fetch(`${base}${path}`, { method, headers: { 'Content-Type': 'application/json' }, body: body && JSON.stringify(body) });
  const json = await res.json();
  if (!res.ok) throw new Error(`${method} ${path}: ${JSON.stringify(json.value)}`);
  return json.value;
};
const run = (id, script) => call('POST', `/session/${id}/execute/sync`, { script, args: [] });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const session = await call('POST', '/session', {
  capabilities: { alwaysMatch: { browserName: 'safari', platformName: 'iOS', 'safari:useSimulator': true, 'safari:deviceUDID': udid } },
});
const id = session.sessionId;
try {
  await call('POST', `/session/${id}/url`, { url });
  // A WebDriver element click does not reach the page in the simulator; click from script instead.
  await run(id, `document.getElementById('start-button').click()`);
  const started = Date.now();
  let state;
  while (Date.now() - started < 180_000) {
    state = await run(id, `return { models: document.body.dataset.models || null, progress: document.getElementById('load-message').textContent, error: document.getElementById('start-error').hidden ? null : document.getElementById('start-error-message').textContent }`);
    if (state.models === 'ready' || (state.error && state.error.includes('モデル')) || (state.error && state.error.includes('対応'))) break;
    await sleep(1000);
  }
  console.log(JSON.stringify({ seconds: Math.round((Date.now() - started) / 1000), ...state }));
  if (state.models !== 'ready') process.exitCode = 1;
} finally {
  await call('DELETE', `/session/${id}`);
}
```

- [ ] **Step 2: iOS シミュレーターの Safari でモデルが読み込めることを確かめる**

シミュレーターにはカメラがないので、確かめるのは「onnxruntime-web が iOS の WebKit でモデルを読み込み、セッションを作れること」だけ。計画時に iOS 26.5 の iPhone 17 シミュレーターで3秒で `ready` になった。WebDriver の要素クリックはシミュレーターのページに届かないため、スクリプトは JS から `click()` している。

```bash
npm run build
(npm run preview -- --port 4173 --strictPort > /dev/null 2>&1 &)
UDID=$(xcrun simctl list devices available | grep -m1 -E '^\s+iPhone' | grep -oE '[0-9A-F-]{36}')
xcrun simctl boot "$UDID"; xcrun simctl bootstatus "$UDID" -b
(safaridriver -p 4444 > /dev/null 2>&1 &)
sleep 3
node scripts/ios-sim-check.mjs http://localhost:4173/face-reception/ "$UDID" 4444
pkill -f 'vite preview --port 4173'; pkill -f 'safaridriver -p 4444'; xcrun simctl shutdown "$UDID"
```

Expected: `{"seconds":...,"error":null,"models":"ready",...}` が出て、終了コード 0。

- [ ] **Step 3: コミットし、`main` に取り込む**

```bash
git add README.md .github/workflows/deploy.yml scripts/ios-sim-check.mjs
git commit -F - <<'EOF'
Add README, GitHub Pages workflow and iOS simulator check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PADPRtudUhYxccwepxBWdj
EOF
npm test && npm run test:e2e
git switch main
git merge --ff-only feature/initial-app
```

Expected: テストがすべて通り、`main` が `feature/initial-app` と同じコミットになる。

- [ ] **Step 4: ユーザーに確認してから GitHub に公開する**

ユーザーに「`RE-yura/face-reception` を public で作成し、GitHub Pages で公開してよいか」を確認する（GitHub Pages を private リポジトリで使うには有料プランが必要）。了承を得てから実行する。Pages を先に有効にしてから push する（有効になっていないと最初のデプロイが失敗する）。

```bash
gh repo create RE-yura/face-reception --public --description "ブラウザだけで動く顔認証デモ（ONNX Runtime Web）" --source . --remote origin
gh api -X POST repos/RE-yura/face-reception/pages -f build_type=workflow
git push -u origin main
gh run watch --exit-status "$(gh run list --workflow deploy.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

Expected: ワークフローの build と deploy が成功する。

- [ ] **Step 5: 公開されたページを確かめる**

```bash
curl -sI https://re-yura.github.io/face-reception/ | head -1
curl -sI https://re-yura.github.io/face-reception/models/face_recognition_sface_2021dec.onnx | grep -iE '^(HTTP|content-length)'
```

Expected: どちらも `HTTP/2 200`。モデルの `content-length` が `38696353`（LFS のポインタの小さなファイルではないこと）。

続けて、Step 2 のスクリプトを公開 URL（`https://re-yura.github.io/face-reception/`）に向けて流し、`"models":"ready"` を確かめる。

- [ ] **Step 6: ユーザーに iPhone 実機で確かめてもらう**

自動テストでは確かめられない点を、ユーザーに iPhone の Safari で確認してもらう。結果を聞いてから完了とする。

1. 公開 URL を開いて「はじめる」→ カメラを許可 → 読み込みの進み具合が出て、登録タブに進む
2. 自分の名前で登録 → 受付タブで「あなたは ○○ さんですね?」と出る。枠が顔に重なっている
3. 登録していない人（別の人、または別の画面に映した他人の写真）で「登録されていません」と出る
4. ページを再読み込みしても登録が残っている
5. ホーム画面に戻ってから Safari に戻ると、カメラと検出が再開する
6. 名前の入力欄をタップしても画面がズームしない。ボタンを素早く2回タップしてもズームしない（number_magic では `touch-action: manipulation` だけではダブルタップズームが止まらず、touchend のガードが必要だった。ズームしたら同じ対策を入れる）
7. 登録者を削除できる

問題があれば systematic-debugging スキルで原因を調べてから直す。
