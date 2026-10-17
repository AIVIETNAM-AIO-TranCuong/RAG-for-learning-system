import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createApi, normalizeBaseUrl, defaultApiUrl, createWorkspaceState, gradeQuiz, optionLetter, validateUpload } from './api.mjs';
import { createSampleApi } from './sample.mjs';

test('API URLs allow HTTP paths and reject credentials and unsafe protocols', () => {
  assert.equal(normalizeBaseUrl(' http://localhost:8000/api/ '), 'http://localhost:8000/api');
  for (const url of ['javascript:alert(1)', 'file:///etc/passwd', 'https://user:secret@example.com', 'https://example.com/?key=secret']) {
    assert.throws(() => normalizeBaseUrl(url));
  }
});

test('public deployments never default to a visitor localhost backend', () => {
  assert.equal(defaultApiUrl('localhost'), 'http://127.0.0.1:8000');
  assert.equal(defaultApiUrl('127.0.0.1'), 'http://127.0.0.1:8000');
  assert.equal(defaultApiUrl('rag-learning-ui.vercel.app'), '');
  assert.equal(defaultApiUrl('rag-learning-ui.vercel.app', 'https://api.example.com/'), 'https://api.example.com');
  assert.equal(defaultApiUrl('rag-learning-ui.vercel.app', '/api', 'https://rag-learning-ui.vercel.app'), 'https://rag-learning-ui.vercel.app/api');
});

test('file size validation accepts the exact limit and rejects oversized and empty files before fetch', async () => {
  assert.doesNotThrow(() => validateUpload({ size: 4_000_000 }));
  assert.throws(() => validateUpload({ size: 4_000_001 }), /4 MB/);
  assert.throws(() => validateUpload({ size: 0 }), /File trống/);
  let calls = 0;
  const api = createApi('https://example.com/api', () => { calls++; });
  assert.throws(() => api.upload(new File([new Uint8Array(4_000_001)], 'large.pdf')), /4 MB/);
  assert.equal(calls, 0);
});

test('all generation requests use the existing single-document contracts', async () => {
  const requests = [];
  const api = createApi('http://localhost:8000/', async (url, init) => {
    requests.push({ path: new URL(url).pathname, method: init.method, body: JSON.parse(init.body) });
    return Response.json({});
  });
  await api.ask('Câu hỏi', null);
  await api.ask('Câu hỏi', 'doc-1');
  await api.summarize('doc-1');
  await api.quiz('doc-1', '', 5);
  await api.flashcards('doc-1', 'RAG', 8);
  assert.deepEqual(requests, [
    { path: '/ask', method: 'POST', body: { question: 'Câu hỏi', document_id: null } },
    { path: '/ask', method: 'POST', body: { question: 'Câu hỏi', document_id: 'doc-1' } },
    { path: '/summarize', method: 'POST', body: { document_id: 'doc-1' } },
    { path: '/quiz', method: 'POST', body: { document_id: 'doc-1', topic: null, n_questions: 5 } },
    { path: '/flashcards', method: 'POST', body: { document_id: 'doc-1', topic: 'RAG', n_cards: 8 } },
  ]);
});

test('upload uses multipart and preserves existing indexes', async () => {
  const api = createApi('http://localhost:8000', async (url, init) => {
    assert.equal(new URL(url).pathname, '/upload');
    assert.equal(init.body.get('force_reindex'), 'false');
    assert.equal(init.body.get('file').name, 'notes.md');
    assert.equal(init.headers, undefined);
    return Response.json({ document_id: 'doc-1' });
  });
  await api.upload(new File(['hello'], 'notes.md'));
});

test('generation errors preserve backend detail and never auto retry', async () => {
  let calls = 0;
  const api = createApi('http://localhost:8000', async () => { calls++; return Response.json({ detail: 'Thiếu API key' }, { status: 500 }); });
  await assert.rejects(api.ask('Câu hỏi', 'doc-1'), /Thiếu API key/);
  assert.equal(calls, 1);
});

test('network failure gives a readable recovery message', async () => {
  const api = createApi('http://localhost:8000', async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(api.health(), /Không kết nối được API/);
});

test('quiz scoring supports option labels and rejects unanswered questions', () => {
  assert.equal(optionLetter('B. Đáp án', 0), 'B');
  assert.equal(optionLetter('Không có nhãn', 2), 'C');
  assert.equal(gradeQuiz([{ answer: 'A' }, { answer: ' b ' }, { answer: 'C' }], { 0: 'A', 1: 'B' }), 2);
});

test('workspace instances cannot leak results or drafts between documents', () => {
  const first = createWorkspaceState(), second = createWorkspaceState();
  first.messages.push({ answer: 'A' }); first.answers[0] = 'B'; first.cards.push({ front: 'A' }); first.draft = 'A';
  assert.deepEqual(second.messages, []); assert.deepEqual(second.answers, {}); assert.deepEqual(second.cards, []); assert.equal(second.draft, '');
});

test('sample supports the four tools and never uploads to the real backend', async () => {
  const api = createSampleApi(); const docs = await api.documents();
  const [answer, summary, quiz, cards] = await Promise.all([
    api.ask('test', docs[0].document_id), api.summarize(docs[0].document_id),
    api.quiz(docs[0].document_id, null, 5), api.flashcards(docs[0].document_id, null, 8),
  ]);
  assert.equal(answer.citations.length, 2); assert.ok(summary.content); assert.equal(quiz.items.length, 5); assert.equal(cards.cards.length, 8);
  await assert.rejects(api.upload(new File(['test'], 'test.md')), /Chuyển sang Kết nối API/);
});
