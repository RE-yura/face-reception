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
