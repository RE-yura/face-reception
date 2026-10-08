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
