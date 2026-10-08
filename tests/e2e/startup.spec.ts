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
