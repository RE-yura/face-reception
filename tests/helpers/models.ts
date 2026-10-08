import { readFileSync } from 'node:fs';

export function readModel(file: string): Uint8Array {
  return readFileSync(new URL(`../../public/models/${file}`, import.meta.url));
}
