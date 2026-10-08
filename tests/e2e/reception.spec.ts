import { expect, test, type Page } from '@playwright/test';
import { launchWithFace, newProfile, openApp } from './helpers.ts';

async function enroll(page: Page, name: string): Promise<void> {
  await page.getByRole('tab', { name: '登録' }).click();
  await page.getByLabel('名前').fill(name);
  await page.getByRole('button', { name: '撮影' }).click();
  await expect(page.locator('#enroll-status')).toHaveText(`${name} さんを登録しました。`, { timeout: 20_000 });
}

async function setHidden(page: Page, hidden: boolean): Promise<void> {
  await page.evaluate((value) => {
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => value });
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (value ? 'hidden' : 'visible') });
    document.dispatchEvent(new Event('visibilitychange'));
  }, hidden);
}

test('enrolls a face, recognizes another photo of the same person after a restart, and rejects someone else', async ({}, testInfo) => {
  const profile = newProfile();
  const out = testInfo.outputDir;

  let context = await launchWithFace(profile, 'meir-a', out);
  let page = await openApp(context);
  await expect(page.getByRole('tab', { name: '登録' })).toHaveAttribute('aria-selected', 'true', { timeout: 60_000 });
  await enroll(page, 'メイア');
  await expect(page.locator('.person-name')).toHaveText(['メイア']);
  await expect(page.locator('.person-count')).toHaveText(['5枚']);
  await page.getByRole('tab', { name: '受付' }).click();
  await expect(page.locator('#reception-message')).toHaveText('あなたは メイア さんですね?');
  await context.close();

  context = await launchWithFace(profile, 'meir-b', out);
  page = await openApp(context);
  await expect(page.getByRole('tab', { name: '受付' })).toHaveAttribute('aria-selected', 'true', { timeout: 60_000 });
  await expect(page.locator('#reception-message')).toHaveText('あなたは メイア さんですね?');
  await context.close();

  context = await launchWithFace(profile, 'kim', out);
  page = await openApp(context);
  await expect(page.locator('#reception-message')).toHaveText('登録されていません', { timeout: 60_000 });
  await context.close();
});

test('adds photos to an existing name and deletes people', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  const page = await openApp(context);
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await enroll(page, 'キム');
  await enroll(page, 'キム');
  await enroll(page, 'キム2');
  await expect(page.locator('.person-name')).toHaveText(['キム', 'キム2']);
  await expect(page.locator('.person-count')).toHaveText(['10枚', '5枚']);
  await page.getByRole('button', { name: 'キム さんを削除' }).click();
  await expect(page.locator('.person-name')).toHaveText(['キム2']);
  page.once('dialog', (dialog) => void dialog.accept());
  await page.getByRole('button', { name: '全員削除' }).click();
  await expect(page.locator('.person-name')).toHaveCount(0);
  await expect(page.locator('#people-empty')).toBeVisible();
  await page.getByRole('tab', { name: '受付' }).click();
  await expect(page.locator('#reception-message')).toHaveText('まずは登録してください');
  await context.close();
});

test('shows names as plain text, never as HTML', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  const page = await openApp(context);
  await expect(page.locator('#main-screen')).toBeVisible({ timeout: 60_000 });
  await enroll(page, '<b>キム</b>');
  await expect(page.locator('.person-name')).toHaveText(['<b>キム</b>']);
  await page.getByRole('tab', { name: '受付' }).click();
  await expect(page.locator('#reception-message')).toHaveText('あなたは <b>キム</b> さんですね?');
  await expect(page.locator('#main-screen b')).toHaveCount(0);
  await context.close();
});

test('releases the camera while the page is in the background and resumes when it returns', async ({}, testInfo) => {
  const context = await launchWithFace(newProfile(), 'kim', testInfo.outputDir);
  const page = await openApp(context);
  await expect(page.locator('#stage')).toHaveAttribute('data-face-count', '1', { timeout: 60_000 });
  await setHidden(page, true);
  await expect(page.locator('#stage')).toHaveAttribute('data-face-count', '0');
  expect(await page.locator('#camera').evaluate((video: HTMLVideoElement) => video.srcObject)).toBeNull();
  await setHidden(page, false);
  await expect(page.locator('#stage')).toHaveAttribute('data-face-count', '1');
  await context.close();
});
