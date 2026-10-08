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
