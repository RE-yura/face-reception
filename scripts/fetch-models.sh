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
