# 顔認証デモ（face-reception）

ブラウザだけで動く顔認証デモです。名前と顔を登録すると、カメラに映った人が誰かを当てます。
[RE-yura/attendance_manager](https://github.com/RE-yura/attendance_manager)（2020年、PyQt5 + PyTorch）をブラウザ向けに作り直しました。

公開ページ: https://re-yura.github.io/face-reception/

## 使い方

1. ページを開いて「はじめる」を押し、カメラを許可します。初回は約 40MB（モデル約 36MB と推論エンジン約 4MB）を読み込みます。
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
