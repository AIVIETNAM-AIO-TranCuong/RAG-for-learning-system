/** Thin adapter for the existing API. No changes to backend contracts. */
export function normalizeBaseUrl(value) {
  const url = new URL(value.trim());
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('Dùng địa chỉ HTTP/HTTPS, không kèm thông tin đăng nhập, query hoặc fragment.');
  }
  return url.href.replace(/\/+$/, '');
}

export const MAX_UPLOAD_BYTES = 4_000_000;
export const MAX_SYSTEM_PROMPT_LENGTH = 8000;
export const DEFAULT_SYSTEM_PROMPT = 'Bạn là trợ lý học tập. Trả lời rõ ràng bằng ngôn ngữ của câu hỏi, chỉ dựa trên các đoạn tài liệu được cung cấp. Trích dẫn nguồn bằng [S1], [S2]… Nếu tài liệu không đủ thông tin, hãy nói rõ. Giải thích dễ hiểu và dùng ví dụ từ tài liệu khi phù hợp.';

export function validateUpload(file) {
  if (file.size > MAX_UPLOAD_BYTES) throw new Error('File vượt giới hạn 4.000.000 byte (4 MB).');
  if (!file.size) throw new Error('File trống, vui lòng chọn tài liệu có nội dung.');
}

export function defaultApiUrl(hostname, configuredUrl = '', origin = '') {
  if (configuredUrl === '/api') return normalizeBaseUrl(`${origin}/api`);
  if (configuredUrl.trim()) return normalizeBaseUrl(configuredUrl);
  return ['localhost', '127.0.0.1', '[::1]', '::1'].includes(hostname)
    ? 'http://127.0.0.1:8000' : '';
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
    ask: (question, documentId, systemPrompt) => request('/ask', { method: 'POST', body: {
      question, document_id: documentId, ...(systemPrompt === undefined ? {} : { system_prompt: systemPrompt }),
    } }),
    summarize: documentId => request('/summarize', { method: 'POST', body: { document_id: documentId } }),
    quiz: (documentId, topic, count) => request('/quiz', { method: 'POST', body: { document_id: documentId, topic: topic || null, n_questions: count } }),
    flashcards: (documentId, topic, count) => request('/flashcards', { method: 'POST', body: { document_id: documentId, topic: topic || null, n_cards: count } }),
    upload: file => {
      validateUpload(file);
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
