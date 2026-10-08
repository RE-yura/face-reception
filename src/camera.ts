/**
 * Starts the front camera in `video`. Resolves false, with the camera released again, when the page went to the
 * background while waiting for it. Rejects with the getUserMedia DOMException on failure.
 */
export async function openCamera(video: HTMLVideoElement): Promise<boolean> {
  if (!navigator.mediaDevices?.getUserMedia) throw new DOMException('getUserMedia is unavailable', 'NotFoundError');
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
  });
  if (document.hidden) {
    stopStream(stream);
    return false;
  }
  closeCamera(video); // an earlier call may have set its own stream while this one waited
  video.srcObject = stream;
  await video.play();
  return true;
}

export function closeCamera(video: HTMLVideoElement): void {
  const stream = video.srcObject;
  if (stream instanceof MediaStream) stopStream(stream);
  video.srcObject = null;
}

export function cameraIsLive(video: HTMLVideoElement): boolean {
  const stream = video.srcObject;
  return stream instanceof MediaStream && stream.getVideoTracks().some((track) => track.readyState === 'live');
}

function stopStream(stream: MediaStream): void {
  for (const track of stream.getTracks()) track.stop();
}
