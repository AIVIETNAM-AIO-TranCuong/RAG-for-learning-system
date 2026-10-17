import { mkdir, readFile, copyFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { normalizeBaseUrl } from './api.mjs';

const root = fileURLToPath(new URL('.', import.meta.url));
const cloud = process.argv.includes('--cloud');
const output = join(root, cloud ? '../public' : 'public');
const value = (process.env.PUBLIC_API_URL || '').trim();
const apiUrl = value ? normalizeBaseUrl(value) : cloud ? '/api' : '';
if (apiUrl && apiUrl !== '/api' && new URL(apiUrl).protocol !== 'https:') {
  throw new Error('PUBLIC_API_URL must be an HTTPS backend URL for the public website.');
}
const attribute = apiUrl.replace(/[&"<>]/g, char => ({ '&': '&amp;', '"': '&quot;', '<': '&lt;', '>': '&gt;' })[char]);
await mkdir(output, { recursive: true });
const html = (await readFile(join(root, 'index.html'), 'utf8'))
  .replace('<meta name="rag-api-url" content="">', `<meta name="rag-api-url" content="${attribute}">`);
await writeFile(join(output, 'index.html'), html);
for (const name of ['styles.css', 'app.mjs', 'api.mjs']) await copyFile(join(root, name), join(output, name));
console.log(`Static UI built: 4 runtime files. Backend: ${apiUrl ? 'configured' : 'configure from the UI later'}.`);
