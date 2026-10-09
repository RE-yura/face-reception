import { chromium, expect, test, type BrowserContext } from '@playwright/test';
import { MODELS } from '../../src/config.ts';
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

test('reloads the page on retry after a camera denial, so a changed camera setting takes effect', async () => {
  const browser = await chromium.launch({ channel: 'chrome', args: ['--use-fake-device-for-media-stream'] });
  const page = await (await browser.newContext()).newPage();
  await page.goto(APP_URL);
  await page.getByRole('button', { name: 'はじめる' }).click();
  await expect(page.locator('#start-error-message')).toContainText('カメラの使用が許可されていません', { timeout: 60_000 });
  await page.getByRole('button', { name: 'もう一度試す' }).click();
  await expect(page.getByRole('button', { name: 'はじめる' })).toBeVisible();
  await expect(page.locator('#start-error')).toBeHidden();
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

test('offers a retry when a model download returns something other than the model, such as a Wi-Fi login page', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  const sface = '**/models/face_recognition_sface_2021dec.onnx';
  await context.route(sface, (route) =>
    route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>Wi-Fi login</title>' }),
  );
  const page = await openApp(context);
  await expect(page.locator('#start-error-message')).toContainText('モデルの読み込みに失敗しました', { timeout: 60_000 });
  await context.unroute(sface);
  await page.getByRole('button', { name: 'もう一度試す' }).click();
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await context.close();
});

/** Holds every request matching `pattern` until the returned function is called. */
async function holdRequests(context: BrowserContext, pattern: string): Promise<{ requested: () => boolean; release: () => void }> {
  let requested = false;
  let release = () => {};
  const gate = new Promise<void>((resolve) => (release = resolve));
  await context.route(pattern, async (route) => {
    requested = true;
    await gate;
    await route.continue();
  });
  return { requested: () => requested, release };
}

test('starts loading the inference engine while the large model is still downloading', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  const sface = await holdRequests(context, '**/models/face_recognition_sface_2021dec.onnx');
  const wasm = await holdRequests(context, '**/*.wasm');
  wasm.release();
  const page = await openApp(context);
  await expect.poll(sface.requested).toBe(true);
  await expect.poll(wasm.requested, { timeout: 15_000 }).toBe(true);
  sface.release();
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await context.close();
});

test('says it is getting ready, not stuck at 100%, once everything has downloaded', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  // Records every text the loading message shows.
  await context.addInitScript(() => {
    const w = window as unknown as { __loadMessages: string[] };
    w.__loadMessages = [];
    document.addEventListener('DOMContentLoaded', () => {
      const message = document.getElementById('load-message')!;
      new MutationObserver(() => w.__loadMessages.push(message.textContent ?? '')).observe(message, {
        childList: true,
        characterData: true,
        subtree: true,
      });
    });
  });
  const page = await openApp(context);
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  const messages = await page.evaluate(() => (window as unknown as { __loadMessages: string[] }).__loadMessages);
  expect(messages.length).toBeGreaterThan(1);
  expect(messages.filter((m) => m.includes('100%'))).toEqual([]);
  expect(messages.at(-1)).toBe('準備しています…');
  await context.close();
});

test('offers a retry when the inference engine fails to download', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  await context.route('**/*.wasm', (route) => route.abort());
  const page = await openApp(context);
  await expect(page.locator('#start-error-message')).toContainText('モデルの読み込みに失敗しました', { timeout: 60_000 });
  await context.unroute('**/*.wasm');
  await page.getByRole('button', { name: 'もう一度試す' }).click();
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await context.close();
});

test('stops downloading the large model once the browser turns out unable to run the detector', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  // The right size, so the download succeeds, but not a model ONNX Runtime can load.
  await context.route(`**/models/${MODELS.yunet.file}`, (route) => route.fulfill({ status: 200, body: Buffer.alloc(MODELS.yunet.bytes) }));
  await holdRequests(context, `**/models/${MODELS.sface.file}`);
  let sfaceAborted = false;
  context.on('requestfailed', (request) => {
    if (request.url().endsWith(MODELS.sface.file)) sfaceAborted = true;
  });
  const page = await openApp(context);
  await expect(page.locator('#start-error-message')).toContainText('このブラウザには対応していません', { timeout: 60_000 });
  await expect.poll(() => sfaceAborted).toBe(true);
  await context.close();
});

test('starts downloading the models when the page opens, with the progress under the start button', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  const sface = await holdRequests(context, `**/models/${MODELS.sface.file}`);
  const page = await context.newPage();
  await page.goto(APP_URL);
  await expect.poll(sface.requested).toBe(true);
  const start = page.getByRole('button', { name: 'はじめる' });
  await expect(start).toBeVisible();
  await expect(page.locator('#load-message')).toContainText('モデルを読み込んでいます');
  const whileLoading = await start.boundingBox();
  sface.release();
  await expect(page.locator('body')).toHaveAttribute('data-models', 'ready', { timeout: 60_000 });
  await expect(page.locator('#load-status')).toBeHidden();
  // The progress keeps its space, so the button does not move under a finger about to tap it.
  expect(await start.boundingBox()).toEqual(whileLoading);
  await start.click();
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await context.close();
});

test('downloads again on tap when the download failed before the tap, without an error first', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  await context.route('**/models/*.onnx', (route) => route.abort());
  const failed = context.waitForEvent('requestfailed', { predicate: (request) => request.url().includes('/models/'), timeout: 30_000 });
  const page = await context.newPage();
  await page.goto(APP_URL);
  await failed;
  await expect(page.locator('#load-status')).toBeHidden();
  await expect(page.locator('#start-error')).toBeHidden();
  await context.unroute('**/models/*.onnx');
  await page.getByRole('button', { name: 'はじめる' }).click();
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await context.close();
});

test('keeps the start button in place from the first paint, before the app script has run', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  // Holds the app script, so the page is painted before it runs, as on a slow first visit.
  const script = await holdRequests(context, '**/assets/index-*.js');
  const page = await context.newPage();
  await page.goto(APP_URL, { waitUntil: 'commit' });
  const start = page.getByRole('button', { name: 'はじめる' });
  await expect(start).toBeVisible();
  const beforeScript = await start.boundingBox();
  script.release();
  await expect(page.locator('#load-status')).toBeVisible();
  expect(await start.boundingBox()).toEqual(beforeScript);
  await context.close();
});

test('reports an unusable browser on tap without leaving the camera on', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  // The right size, so the download succeeds, but not a model ONNX Runtime can load.
  await context.route(`**/models/${MODELS.yunet.file}`, (route) => route.fulfill({ status: 200, body: Buffer.alloc(MODELS.yunet.bytes) }));
  const sfaceStopped = context.waitForEvent('requestfailed', {
    predicate: (request) => request.url().endsWith(MODELS.sface.file),
    timeout: 30_000,
  });
  const page = await context.newPage();
  await page.goto(APP_URL);
  await sfaceStopped;
  await expect(page.locator('#load-status')).toBeHidden();
  await page.getByRole('button', { name: 'はじめる' }).click();
  await expect(page.locator('#start-error-message')).toContainText('このブラウザには対応していません', { timeout: 60_000 });
  const cameraLive = await page.evaluate(() => {
    const stream = (document.getElementById('camera') as HTMLVideoElement).srcObject;
    return stream instanceof MediaStream && stream.getVideoTracks().some((track) => track.readyState === 'live');
  });
  expect(cameraLive).toBe(false);
  await context.close();
});

test('keeps the screen still when the start button is pressed, leaving the button in place as pressed', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  // Holds the large model, so the loading screen stays up.
  await holdRequests(context, `**/models/${MODELS.sface.file}`);
  const page = await context.newPage();
  await page.goto(APP_URL);
  await expect(page.locator('#load-status')).toBeVisible();
  const title = page.locator('.title');
  const before = await title.boundingBox();
  const start = page.getByRole('button', { name: 'はじめる' });
  await start.click();
  await expect(start).toBeVisible();
  await expect(start).toBeDisabled();
  expect(await title.boundingBox()).toEqual(before);
  await context.close();
});
