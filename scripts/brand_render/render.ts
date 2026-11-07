#!/usr/bin/env bun
import { mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { chromium } from 'playwright-core';

const root = resolve(import.meta.dir, '../..');
const out = join(root, 'docs', 'infographics');
mkdirSync(out, { recursive: true });

const htmlPath = join(import.meta.dir, 'brand.html');

const sheets: [string, string, number, number, 'png' | 'jpeg'][] = [
  ['ranking', 'ranking.png', 1600, 900, 'png'],
  ['latency', 'latency-web.jpg', 1600, 900, 'jpeg'],
  ['negative', 'negative-web.jpg', 1600, 900, 'jpeg'],
  ['paraphrase', 'paraphrase.png', 1600, 900, 'png'],
  ['comparison', 'comparison.png', 1600, 900, 'png'],
];

const cr = await chromium.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
});

try {
  const ctx = await cr.newContext({
    viewport: { width: 1600, height: 900 },
    deviceScaleFactor: 2, // 2x Retina resolution
  });

  const page = await ctx.newPage();

  for (const [sheet, filename, width, height, format] of sheets) {
    const url = `file://${htmlPath}?sheet=${sheet}`;
    await page.goto(url);
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(400);

    const dest = join(out, filename);
    await page.screenshot({
      path: dest,
      type: format,
      quality: format === 'jpeg' ? 95 : undefined,
      clip: { x: 0, y: 0, width, height },
    });
    console.log(`Rendered: ${filename} (${width}x${height} ${format})`);
  }
} finally {
  await cr.close();
}
