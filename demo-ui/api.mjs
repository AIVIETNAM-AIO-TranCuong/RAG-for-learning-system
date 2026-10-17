/** Thin adapter for the existing API. No changes to backend contracts. */
export function normalizeBaseUrl(value) {
  const url = new URL(value.trim());
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('Dùng địa chỉ HTTP/HTTPS, không kèm thông tin đăng nhập, query hoặc fragment.');
  }
  return url.href.replace(/\/+$/, '');
}

export function createApi(baseUrl, fetcher = globalThis.fetch) {
  const base = normalizeBaseUrl(baseUrl);
  async function request(path, { method = 'GET', body, timeout = 360000 } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const isForm = body instanceof FormData;
      const response = await fetcher(`${base}${path}`, {
        method,
        signal: controller.signal,
        headers: body && !isForm ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? isForm ? body : JSON.stringify(body) : undefined,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data.detail || data.message || `Yêu cầu thất bại (HTTP ${response.status}).`;
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
      }
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Hết thời gian chờ. Backend có thể vẫn đang xử lý; kiểm tra trước khi gửi lại.');
      if (error instanceof TypeError) throw new Error('Không kết nối được API. Kiểm tra địa chỉ, backend và cấu hình CORS.');
      throw error;
    } finally { clearTimeout(timer); }
  }
  return {
    health: () => request('/health', { timeout: 10000 }),
    documents: () => request('/documents', { timeout: 15000 }),
    ask: (question, documentId) => request('/ask', { method: 'POST', body: { question, document_id: documentId } }),
    summarize: documentId => request('/summarize', { method: 'POST', body: { document_id: documentId } }),
    quiz: (documentId, topic, count) => request('/quiz', { method: 'POST', body: { document_id: documentId, topic: topic || null, n_questions: count } }),
    flashcards: (documentId, topic, count) => request('/flashcards', { method: 'POST', body: { document_id: documentId, topic: topic || null, n_cards: count } }),
    upload: file => {
      const form = new FormData();
      form.append('file', file);
      form.append('force_reindex', 'false');
      return request('/upload', { method: 'POST', body: form });
    },
  };
}

export function createWorkspaceState() {
  return { messages: [], draft: '', summary: null, quiz: null, answers: {}, graded: false,
    cards: [], cardIndex: 0, flipped: false, quizCount: 5, cardCount: 8,
    quizTopic: '', flashcardsTopic: '', error: null };
}

export function optionLetter(option, index) {
  return String(option).match(/^\s*([A-D])(?:[.)\s:：]|$)/i)?.[1]?.toUpperCase() || String.fromCharCode(65 + index);
}

export function gradeQuiz(items, answers) {
  return items.reduce((score, item, index) => score + (answers[index] === String(item.answer).trim().toUpperCase() ? 1 : 0), 0);
}
