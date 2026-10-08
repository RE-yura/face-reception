import { chromium, type BrowserContext, type Page } from '@playwright/test';
import { copyFileSync, mkdirSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const APP_URL = 'http://localhost:4173/face-reception/';

/** A fresh Chrome profile directory. Reusing one across launches keeps IndexedDB, like reopening Safari. */
export function newProfile(): string {
  return mkdtempSync(join(tmpdir(), 'face-reception-'));
}

/**
 * Launches Chrome whose camera shows one face fixture. Chrome reads a .mjpeg file as the fake camera,
 * and a single JPEG is a valid one-frame MJPEG stream.
 */
export async function launchWithFace(profileDir: string, face: string, outDir: string): Promise<BrowserContext> {
  mkdirSync(outDir, { recursive: true });
  const feed = join(outDir, `${face}.mjpeg`);
  copyFileSync(fileURLToPath(new URL(`../fixtures/faces/${face}.jpg`, import.meta.url)), feed);
  return chromium.launchPersistentContext(profileDir, {
    channel: 'chrome',
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${feed}`],
    permissions: ['camera'],
  });
}

export async function openApp(context: BrowserContext): Promise<Page> {
  const page = await context.newPage();
  await page.goto(APP_URL);
  await page.getByRole('button', { name: 'はじめる' }).click();
  return page;
}
