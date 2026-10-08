import { expect, test, type BrowserContext, type Page } from '@playwright/test';
import { countMutations, launchWithFace, newProfile, openApp, setHidden } from './helpers.ts';

/** Records every camera stream, and lets a test hold getUserMedia until it releases it. */
async function instrumentCamera(context: BrowserContext): Promise<void> {
  await context.addInitScript(() => {
    const w = window as unknown as { __streams: MediaStream[]; __holdCamera: boolean; __pendingCamera: (() => void)[] };
    w.__streams = [];
    w.__holdCamera = false;
    w.__pendingCamera = [];
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      if (w.__holdCamera) await new Promise<void>((resolve) => w.__pendingCamera.push(resolve));
      const stream = await original(constraints);
      w.__streams.push(stream);
      return stream;
    };
  });
}

const liveCameras = (page: Page) =>
  page.evaluate(
    () =>
      (window as unknown as { __streams: MediaStream[] }).__streams.filter((s) =>
        s.getVideoTracks().some((t) => t.readyState === 'live'),
      ).length,
  );

const holdCamera = (page: Page, hold: boolean) =>
  page.evaluate((value) => {
    const w = window as unknown as { __holdCamera: boolean; __pendingCamera: (() => void)[] };
    w.__holdCamera = value;
    if (!value) for (const release of w.__pendingCamera.splice(0)) release();
  }, hold);

async function enrollKim(context: BrowserContext): Promise<void> {
  const page = await openApp(context);
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await page.getByLabel('名前').fill('キム');
  await page.getByRole('button', { name: '撮影' }).click();
  await expect(page.locator('#enroll-status')).toHaveText('キム さんを登録しました。', { timeout: 20_000 });
  await page.close();
}

/** On the 登録 tab the reception panel is hidden and must not keep updating. */
async function expectReceptionIdleOnEnrollTab(page: Page): Promise<void> {
  await page.getByRole('tab', { name: '登録' }).click();
  expect(await countMutations(page, '#panel-reception', 1_500)).toBe(0);
}

test('keeps the camera off when the page goes to the background during the model download', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  await instrumentCamera(context);
  await enrollKim(context);

  let release = () => {};
  const gate = new Promise<void>((resolve) => (release = resolve));
  await context.route('**/models/face_recognition_sface_2021dec.onnx', async (route) => {
    await gate;
    await route.continue();
  });
  const page = await openApp(context);
  await setHidden(page, true);
  release();
  await expect(page.locator('body')).toHaveAttribute('data-models', 'ready', { timeout: 60_000 });
  await page.waitForTimeout(500);
  expect(await liveCameras(page)).toBe(0);

  await setHidden(page, false);
  await expect(page.locator('#stage')).toHaveAttribute('data-face-count', '1', { timeout: 10_000 });
  expect(await liveCameras(page)).toBe(1);
  await expectReceptionIdleOnEnrollTab(page);

  await setHidden(page, true);
  await expect.poll(() => liveCameras(page)).toBe(0);
  await context.close();
});

test('keeps the camera off when the page is hidden again before the camera opens', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  await instrumentCamera(context);
  await enrollKim(context);
  const page = await openApp(context);
  await expect(page.locator('#stage')).toHaveAttribute('data-face-count', '1', { timeout: 60_000 });

  await holdCamera(page, true);
  await setHidden(page, true);
  await setHidden(page, false);
  await setHidden(page, true);
  await holdCamera(page, false);
  await page.waitForTimeout(500);
  expect(await liveCameras(page)).toBe(0);

  await setHidden(page, false);
  await expect(page.locator('#stage')).toHaveAttribute('data-face-count', '1', { timeout: 10_000 });
  expect(await liveCameras(page)).toBe(1);
  await expectReceptionIdleOnEnrollTab(page);
  await context.close();
});

test('asks for a reload when face processing stops answering', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  // Lets the test make the worker look stuck by dropping its analyze requests.
  await context.addInitScript(() => {
    const w = window as unknown as { __dropAnalyze: boolean };
    w.__dropAnalyze = false;
    const original = Worker.prototype.postMessage as (this: Worker, message: unknown, transfer: Transferable[]) => void;
    Worker.prototype.postMessage = function (this: Worker, message: unknown, transfer?: Transferable[]) {
      if (w.__dropAnalyze && (message as { type?: string }).type === 'analyze') return;
      original.call(this, message, transfer ?? []);
    } as typeof Worker.prototype.postMessage;
  });
  const page = await openApp(context);
  await expect(page.locator('#stage')).toHaveAttribute('data-face-count', '1', { timeout: 60_000 });
  await page.evaluate(() => ((window as unknown as { __dropAnalyze: boolean }).__dropAnalyze = true));
  await expect(page.locator('#start-error-message')).toHaveText('顔の処理が止まってしまいました。ページを再読み込みしてください。', {
    timeout: 15_000,
  });
  await expect(page.locator('#main-screen')).toBeHidden();
  expect(await page.locator('#camera').evaluate((video: HTMLVideoElement) => video.srcObject)).toBeNull();
  await page.getByRole('button', { name: 'ページを再読み込み' }).click();
  await expect(page.getByRole('button', { name: 'はじめる' })).toBeVisible();
  await context.close();
});
