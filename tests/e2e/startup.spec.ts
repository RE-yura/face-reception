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
