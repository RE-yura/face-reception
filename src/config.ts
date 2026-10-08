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

/** A model download gives up when no data arrives for this long. */
export const DOWNLOAD_IDLE_TIMEOUT_MS = 20_000;
