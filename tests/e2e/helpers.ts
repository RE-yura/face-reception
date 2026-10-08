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

/** Fakes the page going to the background (screen lock, app switch) or coming back. */
export async function setHidden(page: Page, hidden: boolean): Promise<void> {
  await page.evaluate((value) => {
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => value });
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (value ? 'hidden' : 'visible') });
    document.dispatchEvent(new Event('visibilitychange'));
  }, hidden);
}

/** Counts DOM mutations under `selector` during `ms` milliseconds. */
export function countMutations(page: Page, selector: string, ms: number): Promise<number> {
  return page.evaluate(
    ([sel, duration]) =>
      new Promise<number>((resolve) => {
        let count = 0;
        const observer = new MutationObserver((records) => (count += records.length));
        observer.observe(document.querySelector(sel as string)!, { subtree: true, childList: true, characterData: true });
        setTimeout(() => {
          observer.disconnect();
          resolve(count);
        }, duration as number);
      }),
    [selector, ms] as const,
  );
}

export async function openApp(context: BrowserContext): Promise<Page> {
  const page = await context.newPage();
  await page.goto(APP_URL);
  await page.getByRole('button', { name: 'はじめる' }).click();
  return page;
}
