import { defineConfig } from '@playwright/test';
import { existsSync } from 'node:fs';

// A pre-installed Chromium can be pointed to with PW_CHROMIUM (the media capture script uses the same binary).
const executablePath = process.env.PW_CHROMIUM ?? (existsSync('/opt/pw-browsers/chromium') ? '/opt/pw-browsers/chromium' : undefined);

export default defineConfig({
  testDir: './tests',
  timeout: 120_000,
  use: {
    baseURL: 'http://127.0.0.1:4173/dhaka-rl-signals/',
    headless: true,
    launchOptions: { executablePath, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] },
  },
  webServer: {
    command: 'npx vite preview --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173/dhaka-rl-signals/',
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
