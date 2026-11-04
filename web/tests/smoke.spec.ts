import { expect, test } from '@playwright/test';

test('site loads, renders the WebGL canvas and logs no console errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', (err) => errors.push(err.message));
  await page.goto('./');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Safe Deep RL Traffic Signals');
  await expect(page.locator('canvas')).toHaveCount(1, { timeout: 90_000 });
  await expect(page.getByTestId('hud')).toBeVisible({ timeout: 90_000 });
  await page.waitForTimeout(2000);
  expect(errors, errors.join('\n')).toEqual([]);
});
