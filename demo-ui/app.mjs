import { createApi, normalizeBaseUrl, defaultApiUrl, createWorkspaceState, gradeQuiz, optionLetter, validateUpload, DEFAULT_SYSTEM_PROMPT, MAX_SYSTEM_PROMPT_LENGTH } from './api.mjs';

const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
const icon = name => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
// Old preview links now open the live workspace too.
const pageUrl = new URL(location.href);
if (pageUrl.searchParams.has('sample')) {
  pageUrl.searchParams.delete('sample');
  history.replaceState(null, '', pageUrl);
}
let savedUrl;
try { savedUrl = localStorage.getItem('rag-demo:api-url'); } catch { /* Storage can be disabled. */ }
let savedSystemPrompt = null;
try { savedSystemPrompt = localStorage.getItem('rag-demo:system-prompt'); } catch { /* Keep editing available without storage. */ }
const configuredUrl = document.querySelector('meta[name="rag-api-url"]')?.content || '';
const state = { url: savedUrl || defaultApiUrl(location.hostname, configuredUrl, location.origin), tab: 'chat',
  documents: [], selected: null, loading: true, health: null, connectionError: '', documentsError: '', job: null,
  scopes: new Map(), loadId: 0, uploadItems: [],
  systemPrompt: (savedSystemPrompt ?? DEFAULT_SYSTEM_PROMPT).slice(0, MAX_SYSTEM_PROMPT_LENGTH) };
let api;
let toastTimer;
const panel = $('#workspace-panel');
const scopeKey = () => `${state.url}:${state.selected || '__all__'}`;
function workspace() {
  const key = scopeKey();
  if (!state.scopes.has(key)) state.scopes.set(key, createWorkspaceState());
  return state.scopes.get(key);
}
const selectedDocument = () => state.documents.find(doc => doc.document_id === state.selected);
const sourceName = () => selectedDocument()?.filename || (state.url ? 'Toàn bộ thư viện' : 'Chưa kết nối backend');
const busy = type => state.job?.type === type && state.job.scope === scopeKey();
const blocked = () => Boolean(state.job) || state.loading || !state.documents.length;
function toast(message) {
  clearTimeout(toastTimer); $('#toast').textContent = message; $('#toast').hidden = false;
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 3500);
}
function errorBox(error, retry = 'retry') {
  return `<div class="error-box" role="alert"><strong>Chưa thực hiện được</strong><p>${esc(error)}</p><button class="button" data-action="${retry}">${icon('refresh')}Thử lại</button></div>`;
}

// All model text is escaped first. Raw HTML and Markdown links are never rendered.
function markdown(value) {
  const inline = text => esc(text).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[(S\d+)\]/g, '<button class="citation" data-citation="$1" aria-label="Xem trích dẫn $1">$1</button>');
  const output = []; let list = null; let code = false; let codeLines = [];
  for (const line of String(value || '').split('\n')) {
    if (line.startsWith('```')) {
      if (list) { output.push(`</${list}>`); list = null; }
      if (code) { output.push(`<pre><code>${esc(codeLines.join('\n'))}</code></pre>`); codeLines = []; }
      code = !code; continue;
    }
    if (code) { codeLines.push(line); continue; }
    const bullet = line.match(/^\s*(?:[-*]|\d+\.)\s+(.+)/);
    if (bullet) {
      const kind = /^\s*\d+\./.test(line) ? 'ol' : 'ul';
      if (list !== kind) { if (list) output.push(`</${list}>`); output.push(`<${kind}>`); list = kind; }
      output.push(`<li>${inline(bullet[1])}</li>`); continue;
    }
    if (list) { output.push(`</${list}>`); list = null; }
    const heading = line.match(/^#{1,6}\s+(.+)/);
    if (heading) output.push(`<h3>${inline(heading[1])}</h3>`);
    else if (line.trim()) output.push(`<p>${inline(line)}</p>`);
  }
  if (list) output.push(`</${list}>`);
  if (codeLines.length) output.push(`<pre><code>${esc(codeLines.join('\n'))}</code></pre>`);
  return output.join('');
}

function renderHeader() {
  $('#system-prompt-editor').innerHTML = `<div class="system-prompt-heading"><label for="system-prompt">System prompt</label><button class="button" type="button" data-action="reset-system-prompt" ${busy('chat') ? 'disabled' : ''}>Khôi phục mặc định</button></div><textarea id="system-prompt" rows="3" maxlength="${MAX_SYSTEM_PROMPT_LENGTH}" aria-describedby="system-prompt-hint" ${busy('chat') ? 'disabled' : ''}>${esc(state.systemPrompt)}</textarea><div class="system-prompt-footer"><p id="system-prompt-hint">Áp dụng cho các câu hỏi tiếp theo · Lưu riêng trên trình duyệt này. Để trống để dùng hướng dẫn mặc định.</p><span id="system-prompt-count">${state.systemPrompt.length}/${MAX_SYSTEM_PROMPT_LENGTH}</span></div>`;
  $('#settings-button').innerHTML = `<span class="status-dot ${state.health ? 'online' : ''}"></span>${state.health ? 'API đã kết nối' : !state.url ? 'Kết nối backend' : state.loading ? 'Đang kết nối' : 'API ngoại tuyến'}${icon('settings')}`;
  $('#scope-label').innerHTML = `Nguồn đang chọn <span aria-hidden="true">/</span> <strong>${esc(sourceName())}</strong>`;
  $('#mode-notice').innerHTML = !state.url
    ? '<strong>Giao diện đã sẵn sàng.</strong> Kết nối backend HTTPS để đọc tài liệu và dùng các công cụ học tập. <button data-action="settings">Thiết lập kết nối</button>'
    : state.connectionError
    ? `${esc(state.connectionError)} <button data-action="settings">Đổi địa chỉ API</button> hoặc <button data-action="reload">thử kết nối lại</button>.`
    : state.health?.llm_configured === false ? '<strong>Đã kết nối thư viện thật.</strong> Backend chưa có API key để sinh nội dung. Cấu hình khóa ở backend rồi khởi động lại API.'
    : 'Từ tài liệu của bạn: hỏi đáp có trích dẫn, tóm tắt và công cụ ôn tập trong cùng một không gian.';
  document.querySelectorAll('[data-tab]').forEach(button => {
    const selected = button.dataset.tab === state.tab;
    button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
  });
  panel.setAttribute('aria-labelledby', `tab-${state.tab}`);
  $('#upload-button').disabled = Boolean(state.job) || !state.url;
  $('#upload-button').title = state.url ? 'Thêm tài liệu · tối đa 4 MB/file' : 'Kết nối backend để thêm tài liệu';
  $('#refresh-button').disabled = state.loading || Boolean(state.job) || !state.url;
}

function renderSources() {
  const search = $('#source-search').value.toLocaleLowerCase('vi');
  const docs = state.documents.filter(doc => doc.filename.toLocaleLowerCase('vi').includes(search));
  $('#document-count').textContent = state.documents.length || '';
  const sourceList = $('#source-list');
  if (!state.url) sourceList.innerHTML = '<p class="muted">Thư viện sẽ xuất hiện sau khi bạn kết nối backend.</p><button class="button" data-action="settings">Kết nối backend</button>';
  else if (state.loading) sourceList.innerHTML = '<div class="busy-row"><span class="spinner"></span>Đang lấy tài liệu…</div>';
  else if (state.documentsError) sourceList.innerHTML = errorBox(state.documentsError, 'reload');
  else if (!state.documents.length) sourceList.innerHTML = '<div class="muted">Chưa có tài liệu. Thêm nguồn đầu tiên để bắt đầu.</div>';
  else sourceList.innerHTML = `<p class="list-label">THƯ VIỆN CỦA BẠN</p>${docs.map(doc => `
    <button class="source-item" data-source="${esc(doc.document_id)}" aria-pressed="${state.selected === doc.document_id}" title="${esc(doc.filename)}">
      <span class="file-icon">${icon('file')}</span><span class="file-text"><span class="file-name">${esc(doc.filename)}</span><span class="file-meta">${esc(doc.filename.split('.').at(-1).toUpperCase())} · ${Number(doc.max_page || 1)} trang · ${Number(doc.total_chunks || 0)} đoạn</span></span>${state.selected === doc.document_id ? '<svg class="selected-mark" aria-hidden="true"><use href="#i-check"/></svg>' : ''}
    </button>`).join('')}${!docs.length ? '<p class="muted">Không tìm thấy tài liệu khớp tên.</p>' : ''}
    <button class="source-item" data-source="" aria-pressed="${state.selected === null}" title="Hỏi đáp trên toàn bộ tài liệu">${icon('book')}<span class="file-text"><span class="file-name">Toàn bộ thư viện</span><span class="file-meta">Dành cho hỏi đáp</span></span></button>`;
  $('#upload-status').innerHTML = state.uploadItems.map(item => `<div class="upload-item ${item.error ? 'error' : ''}"><strong>${esc(item.name)}</strong><br>${esc(item.status)}</div>`).join('');
}

function renderChat() {
  const w = workspace();
  const messages = w.messages.map((message, index) => `<article class="message" data-message="${index}"><div class="question">${esc(message.question)}</div><div class="answer">${markdown(message.answer)}</div><div class="answer-actions"><button class="icon-button" data-action="copy-answer" data-index="${index}" title="Sao chép câu trả lời" aria-label="Sao chép câu trả lời">${icon('copy')}</button>${message.citations?.length ? `<button class="source-chip" data-citation="${esc(message.citations[0].marker)}" data-message-index="${index}">${icon('file')}<span>${message.citations.length} trích đoạn nguồn</span></button>` : ''}<span>${esc(message.source)}</span></div></article>`).join('');
  const suggestions = ['Giải thích những ý chính trong tài liệu', 'Các khái niệm quan trọng liên hệ thế nào?', 'Gợi ý những câu hỏi để ôn tập'];
  panel.innerHTML = `<div class="chat-stream"><div class="chat-inner">${messages || `<div class="welcome"><div class="welcome-mark">${icon('spark')}</div><h2>Bắt đầu từ một câu hỏi.</h2><p>Khám phá ý chính, kết nối kiến thức và tìm câu trả lời có trích dẫn từ nguồn bạn chọn.</p><div class="suggestions">${suggestions.map(text => `<button class="button suggestion" data-suggestion="${esc(text)}">${esc(text)}${icon('arrow')}</button>`).join('')}</div></div>`}${busy('chat') ? `<div class="question">${esc(state.job.question)}</div><div class="busy-row" role="status"><span class="spinner"></span>Đang đọc nguồn và soạn câu trả lời…</div>` : ''}${w.error?.type === 'chat' ? errorBox(w.error.message) : ''}</div></div>
    <div class="composer-wrap"><form class="composer" id="chat-form"><textarea id="question" aria-label="Câu hỏi" placeholder="Hỏi điều gì đó từ tài liệu của bạn…" rows="2">${esc(w.draft)}</textarea><div class="composer-bottom"><div class="composer-context">${icon('file')}<span>${esc(sourceName())}</span></div><button id="send-button" class="button primary send-button" type="submit" aria-label="Gửi câu hỏi" title="Gửi câu hỏi" ${blocked() || !w.draft.trim() ? 'disabled' : ''}>${icon('send')}</button></div></form><p class="composer-note">Mỗi câu hỏi được xử lý độc lập. Enter để gửi · Shift + Enter xuống dòng</p></div>`;
}

function saveSystemPrompt(value) {
  state.systemPrompt = value.slice(0, MAX_SYSTEM_PROMPT_LENGTH);
  try { localStorage.setItem('rag-demo:system-prompt', state.systemPrompt); } catch { /* Session editing still works. */ }
  if ($('#system-prompt-count')) $('#system-prompt-count').textContent = `${state.systemPrompt.length}/${MAX_SYSTEM_PROMPT_LENGTH}`;
}

function noSource() {
  panel.innerHTML = `<div class="empty-state">${icon('file')}<h3>Chọn một tài liệu để tiếp tục</h3><p>Công cụ này làm việc với một nguồn. Chọn tài liệu trong thư viện để tạo nội dung.</p><button class="button" data-action="sources">Chọn tài liệu</button></div>`;
}
const loading = text => `<div class="busy-row" role="status"><span class="spinner"></span>${text}</div>`;
const empty = (glyph, title, text) => `<div class="empty-state">${icon(glyph)}<h3>${title}</h3><p>${text}</p></div>`;
function renderSummary() {
  if (!state.selected) return noSource();
  const w = workspace();
  panel.innerHTML = `<div class="learning"><div class="section-heading"><div><h2>Nắm ý chính, giữ mạch kiến thức.</h2><p>Một bản tóm tắt gọn, dễ đọc từ tài liệu đang chọn.</p></div><button class="button primary" data-action="summary" ${blocked() ? 'disabled' : ''}>${icon('spark')}${w.summary ? 'Tạo lại' : 'Tạo tóm tắt'}</button></div>${w.error?.type === 'summary' ? errorBox(w.error.message) : ''}${busy('summary') ? loading('Đang tổng hợp các ý chính. Tài liệu dài có thể cần thêm thời gian…') : w.summary ? `<div class="study-content"><div class="answer">${markdown(w.summary.content)}</div><button class="button" data-action="copy-summary">${icon('copy')}Sao chép</button></div>` : empty('align', 'Bức tranh tổng thể của tài liệu', 'Tạo bản tóm tắt để nhìn nhanh các ý chính trước khi đi sâu vào từng phần.')}</div>`;
}

function studyForm(type) {
  const w = workspace(); const quiz = type === 'quiz'; const field = quiz ? 'quizCount' : 'cardCount';
  return `<form class="form-row" id="${type}-form"><div class="form-field"><label for="${type}-topic">Chủ đề <span class="muted">(tùy chọn)</span></label><input id="${type}-topic" data-field="${type}Topic" value="${esc(w[`${type}Topic`])}" placeholder="${quiz ? 'Ví dụ: RLHF và DPO' : 'Ví dụ: Các khái niệm cốt lõi'}"></div><div class="form-field count"><label for="${type}-count">${quiz ? 'Số câu hỏi' : 'Số thẻ'}</label><div class="stepper"><button type="button" data-step="-1" data-field="${field}" aria-label="Giảm số lượng">−</button><input id="${type}-count" data-field="${field}" type="number" min="1" max="${quiz ? 10 : 15}" value="${w[field]}" required aria-label="${quiz ? 'Số câu hỏi' : 'Số thẻ'}"><button type="button" data-step="1" data-field="${field}" aria-label="Tăng số lượng">+</button></div></div><button type="submit" class="button primary" ${blocked() ? 'disabled' : ''}>${icon('spark')}${quiz ? 'Tạo trắc nghiệm' : 'Tạo thẻ nhớ'}</button></form>`;
}
function renderQuiz() {
  if (!state.selected) return noSource();
  const w = workspace(); const items = w.quiz?.items || [];
  panel.innerHTML = `<div class="learning"><div class="section-heading"><div><h2>Thử xem bạn đã hiểu đến đâu.</h2><p>Làm bài, kiểm tra đáp án và học thêm từ lời giải thích.</p></div></div>${studyForm('quiz')}${w.error?.type === 'quiz' ? errorBox(w.error.message) : ''}${busy('quiz') ? loading('Đang tạo câu hỏi từ tài liệu…') : !items.length ? empty('quiz', 'Biến kiến thức thành phản xạ', 'Chọn số câu hỏi và một chủ đề muốn luyện tập. Bạn sẽ thấy điểm và giải thích sau khi nộp bài.') : `<div class="study-content">${w.graded ? `<div class="quiz-score" role="status"><strong>${gradeQuiz(items, w.answers)} / ${items.length} câu đúng</strong><p>Xem lời giải bên dưới để củng cố những phần cần ôn lại.</p></div>` : ''}<form id="quiz-answers">${items.map((item, index) => `<fieldset class="quiz-question"><legend>${index + 1}. ${esc(item.question)}</legend>${item.options.map((option, optionIndex) => {
    const letter = optionLetter(option, optionIndex); const correct = String(item.answer).trim().toUpperCase();
    const result = w.graded ? correct === letter ? 'correct' : w.answers[index] === letter ? 'incorrect' : '' : '';
    return `<label class="quiz-option ${result}"><input type="radio" name="answer-${index}" data-answer="${index}" value="${letter}" ${w.answers[index] === letter ? 'checked' : ''} ${w.graded ? 'disabled' : ''}><span>${esc(option)}</span>${w.graded && correct === letter ? icon('check') : ''}</label>`;
  }).join('')}${w.graded ? `<p class="explanation"><strong>Đáp án ${esc(item.answer)}.</strong> ${esc(item.explanation)}</p>` : ''}</fieldset>`).join('')}${w.graded ? '<button type="button" class="button" data-action="redo-quiz">Làm lại bộ câu hỏi</button>' : `<p class="muted" id="quiz-progress">Đã chọn ${Object.keys(w.answers).length}/${items.length} câu</p><button class="button primary" type="submit">Nộp bài và xem kết quả</button>`}</form></div>`}</div>`;
}
function renderFlashcards() {
  if (!state.selected) return noSource();
  const w = workspace(); const card = w.cards[w.cardIndex];
  panel.innerHTML = `<div class="learning"><div class="section-heading"><div><h2>Một khái niệm, mỗi lần lật.</h2><p>Chủ động nhớ lại trước khi xem đáp án.</p></div></div>${studyForm('flashcards')}${w.error?.type === 'flashcards' ? errorBox(w.error.message) : ''}${busy('flashcards') ? loading('Đang chọn các khái niệm để tạo thẻ…') : !card ? empty('cards', 'Ghi nhớ theo cách chủ động', 'Tạo bộ thẻ từ nguồn đang chọn, rồi lật từng thẻ để tự kiểm tra kiến thức.') : `<div class="flashcard-wrap"><button class="flashcard ${w.flipped ? 'flipped' : ''}" data-action="flip" aria-label="${w.flipped ? 'Lật về câu hỏi' : 'Lật xem đáp án'}"><span class="card-side">${w.flipped ? 'ĐÁP ÁN' : 'TỰ KIỂM TRA'}</span><span class="card-text">${esc(w.flipped ? card.back : card.front)}</span><span class="card-hint">Nhấn vào thẻ để ${w.flipped ? 'xem câu hỏi' : 'xem đáp án'}</span></button><div class="card-progress" aria-hidden="true"><div style="width:${(w.cardIndex + 1) / w.cards.length * 100}%"></div></div><div class="card-controls"><button class="icon-button previous" data-action="previous-card" aria-label="Thẻ trước" ${w.cardIndex === 0 ? 'disabled' : ''}>${icon('arrow')}</button><span class="card-counter">${w.cardIndex + 1} / ${w.cards.length}</span><button class="icon-button" data-action="next-card" aria-label="Thẻ tiếp" ${w.cardIndex === w.cards.length - 1 ? 'disabled' : ''}>${icon('arrow')}</button><button class="button" data-action="shuffle">${icon('shuffle')}Xáo trộn</button></div><p class="card-keyboard">← → Chuyển thẻ · Space Lật thẻ</p></div>`}</div>`;
}

function render({ bottom = false } = {}) {
  renderHeader(); renderSources();
  ({ chat: renderChat, summary: renderSummary, quiz: renderQuiz, flashcards: renderFlashcards })[state.tab]();
  if (bottom) { const stream = $('.chat-stream'); if (stream) stream.scrollTop = stream.scrollHeight; }
}

async function loadSources() {
  const loadId = ++state.loadId;
  state.loading = true; state.connectionError = ''; state.documentsError = ''; state.health = null;
  if (!state.url) { state.loading = false; state.documents = []; state.selected = null; render(); return; }
  api = createApi(state.url);
  render();
  const [health, documents] = await Promise.allSettled([api.health(), api.documents()]);
  if (loadId !== state.loadId) return;
  state.loading = false;
  state.health = health.status === 'fulfilled' ? health.value : null;
  state.connectionError = health.status === 'rejected' ? health.reason.message : '';
  if (documents.status === 'fulfilled' && Array.isArray(documents.value)) {
    state.documents = documents.value;
    if (!state.documents.some(doc => doc.document_id === state.selected)) state.selected = state.documents[0]?.document_id || null;
  } else { state.documents = []; state.selected = null; state.documentsError = documents.reason?.message || 'Danh sách tài liệu không đúng định dạng.'; }
  render();
}

async function generate(type) {
  if (blocked()) return;
  const w = workspace(); const key = scopeKey(); const selected = state.selected; const name = sourceName();
  const question = w.draft.trim();
  const systemPrompt = state.systemPrompt;
  if (type === 'chat' && !question) return;
  if (type !== 'chat' && !selected) return toast('Chọn một tài liệu trước khi tạo nội dung.');
  const countField = type === 'quiz' ? 'quizCount' : 'cardCount';
  if (['quiz', 'flashcards'].includes(type)) {
    const count = Number(w[countField]); const min = 1; const max = type === 'quiz' ? 10 : 15;
    if (!Number.isInteger(count) || count < min || count > max) return toast(`Chọn số lượng từ ${min} đến ${max}.`);
  }
  w.error = null; state.job = { type, scope: key, question }; render({ bottom: true });
  try {
    const result = type === 'chat' ? await api.ask(question, selected, systemPrompt) : type === 'summary' ? await api.summarize(selected)
      : type === 'quiz' ? await api.quiz(selected, w.quizTopic.trim(), w.quizCount)
      : await api.flashcards(selected, w.flashcardsTopic.trim(), w.cardCount);
    if (type === 'chat') {
      if (typeof result.answer !== 'string') throw new Error('API trả về câu trả lời không hợp lệ.');
      w.messages.push({ question, answer: result.answer, citations: result.citations || [], source: name });
      if (w.draft.trim() === question) w.draft = '';
    } else if (type === 'summary') { if (!result.content) throw new Error('API chưa trả nội dung tóm tắt.'); w.summary = result; }
    else if (type === 'quiz') { if (!result.items?.length) throw new Error('API chưa trả câu hỏi nào.'); w.quiz = result; w.answers = {}; w.graded = false; }
    else { if (!result.cards?.length) throw new Error('API chưa trả thẻ nào.'); w.cards = result.cards; w.cardIndex = 0; w.flipped = false; }
  } catch (error) { w.error = { type, message: error.message }; }
  finally { state.job = null; render({ bottom: scopeKey() === key }); }
}

async function upload(files) {
  if (state.job || !state.url || !files.length) return;
  state.job = { type: 'upload' }; state.uploadItems = [...files].map(file => ({ name: file.name, status: 'Đang chờ' })); render();
  for (let index = 0; index < files.length; index++) {
    const file = files[index]; const item = state.uploadItems[index];
    if (!/\.(pdf|docx|pptx|txt|md)$/i.test(file.name)) { item.status = 'Định dạng chưa được hỗ trợ.'; item.error = true; renderSources(); continue; }
    try { validateUpload(file); } catch (error) { item.status = error.message; item.error = true; renderSources(); continue; }
    item.status = 'Đang tải lên và lập chỉ mục…'; renderSources();
    try { const result = await api.upload(file); item.status = `Đã sẵn sàng · ${result.total_chunks} đoạn`; }
    catch (error) { item.status = error.message; item.error = true; }
    renderSources();
  }
  state.job = null; $('#file-input').value = ''; await loadSources();
}

async function reconnect() {
  if (state.job) return toast('Chờ tác vụ hiện tại hoàn tất trước khi đổi kết nối.');
  state.documents = []; state.selected = null; state.uploadItems = [];
  await loadSources();
}
function showSettings() {
  $('#api-url').value = state.url;
  $('#api-url').placeholder = location.protocol === 'https:' ? 'https://api.example.com' : 'http://127.0.0.1:8000';
  $('#settings-error').textContent = ''; $('#settings-dialog').showModal();
}
function showSources() {
  if (matchMedia('(max-width: 900px)').matches) {
    $('#sources-dialog').append($('#sources')); $('#sources-dialog').showModal();
  } else { $('#source-search').focus(); }
}
function openCitation(marker, messageIndex) {
  const w = workspace();
  const citation = messageIndex !== undefined ? w.messages[messageIndex]?.citations?.find(item => item.marker === marker)
    : [...w.messages].reverse().flatMap(item => item.citations || []).find(item => item.marker === marker);
  if (!citation) return toast('Không có đoạn nguồn tương ứng trong phản hồi API.');
  const meta = citation.chunk?.metadata || {};
  $('#citation-content').innerHTML = `<div class="dialog-heading"><div><p class="eyebrow">TRÍCH ĐOẠN NGUỒN</p><h2 id="citation-title">${esc(meta.filename || 'Tài liệu')}</h2></div><button class="icon-button" data-close="citation-dialog" aria-label="Đóng trích dẫn">${icon('x')}</button></div><span class="citation-label">${esc(marker)} · Trang / slide ${esc(meta.page ?? '—')}</span><p class="citation-excerpt">${esc(citation.chunk?.content || 'Không có nội dung đoạn nguồn.')}</p>`;
  $('#citation-dialog').showModal();
}
async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast('Đã sao chép.'); }
  catch { toast('Trình duyệt chưa cho phép sao chép. Bạn có thể chọn và sao chép nội dung trực tiếp.'); }
}

document.addEventListener('click', event => {
  const target = event.target.closest('button'); if (!target || target.disabled) return;
  if (target.dataset.close) return $(`#${target.dataset.close}`).close();
  if (target.dataset.tab) { state.tab = target.dataset.tab; render(); return; }
  if (target.dataset.source !== undefined) {
    state.selected = target.dataset.source || null;
    if ($('#sources-dialog').open) $('#sources-dialog').close(); render(); return;
  }
  if (target.dataset.suggestion) { workspace().draft = target.dataset.suggestion; render(); $('#question')?.focus(); return; }
  if (target.dataset.citation) {
    const article = target.closest('[data-message]');
    openCitation(target.dataset.citation, article ? Number(article.dataset.message) : target.dataset.messageIndex === undefined ? undefined : Number(target.dataset.messageIndex)); return;
  }
  if (target.dataset.step) {
    const field = target.dataset.field; const min = 1; const max = field === 'quizCount' ? 10 : 15;
    workspace()[field] = Math.max(min, Math.min(max, Number(workspace()[field]) + Number(target.dataset.step))); render(); return;
  }
  const action = target.dataset.action;
  const w = workspace();
  if (action === 'reset-system-prompt') {
    saveSystemPrompt(DEFAULT_SYSTEM_PROMPT);
    if ($('#system-prompt')) { $('#system-prompt').value = state.systemPrompt; $('#system-prompt').focus(); }
  }
  else if (action === 'summary') generate('summary');
  else if (action === 'retry') generate(w.error?.type || state.tab);
  else if (action === 'reload') loadSources();
  else if (action === 'settings') showSettings();
  else if (action === 'sources') showSources();
  else if (action === 'copy-answer') copy(w.messages[Number(target.dataset.index)].answer);
  else if (action === 'copy-summary') copy(w.summary.content);
  else if (action === 'redo-quiz') { w.answers = {}; w.graded = false; render(); }
  else if (action === 'flip') { w.flipped = !w.flipped; render(); $('.flashcard')?.focus({ preventScroll: true }); }
  else if (action === 'previous-card' || action === 'next-card') {
    w.cardIndex = Math.max(0, Math.min(w.cards.length - 1, w.cardIndex + (action === 'next-card' ? 1 : -1))); w.flipped = false; render();
  } else if (action === 'shuffle') {
    for (let i = w.cards.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [w.cards[i], w.cards[j]] = [w.cards[j], w.cards[i]]; }
    w.cardIndex = 0; w.flipped = false; render(); toast('Đã xáo trộn bộ thẻ.');
  }
});

document.addEventListener('input', event => {
  const { target } = event; const w = workspace();
  if (target.id === 'question') { w.draft = target.value; $('#send-button').disabled = blocked() || !target.value.trim(); }
  else if (target.id === 'system-prompt') saveSystemPrompt(target.value);
  else if (target.dataset.field) w[target.dataset.field] = target.type === 'number' ? Number(target.value) : target.value;
});
document.addEventListener('change', event => {
  const target = event.target;
  if (target.dataset.answer !== undefined) {
    workspace().answers[Number(target.dataset.answer)] = target.value;
    $('#quiz-progress').textContent = `Đã chọn ${Object.keys(workspace().answers).length}/${workspace().quiz.items.length} câu`;
  }
});
document.addEventListener('submit', event => {
  event.preventDefault();
  if (event.target.id === 'chat-form') generate('chat');
  else if (event.target.id === 'quiz-form') generate('quiz');
  else if (event.target.id === 'flashcards-form') generate('flashcards');
  else if (event.target.id === 'quiz-answers') {
    const w = workspace();
    if (Object.keys(w.answers).length !== w.quiz.items.length) return toast('Chọn đáp án cho tất cả câu hỏi trước khi nộp bài.');
    w.graded = true; render(); panel.scrollTop = 0;
  } else if (event.target.id === 'settings-form') {
    if (state.job) { $('#settings-error').textContent = 'Chờ tác vụ hiện tại hoàn tất trước khi đổi kết nối.'; return; }
    try {
      const nextUrl = normalizeBaseUrl($('#api-url').value);
      if (location.protocol === 'https:' && new URL(nextUrl).protocol !== 'https:') {
        throw new Error('Bản public cần địa chỉ backend HTTPS. Backend localhost chỉ dùng với UI chạy trên máy.');
      }
      state.url = nextUrl;
      try { localStorage.setItem('rag-demo:api-url', state.url); } catch { /* Optional preference. */ }
      $('#settings-dialog').close(); reconnect();
    } catch (error) { $('#settings-error').textContent = error.message; }
  }
});
document.addEventListener('keydown', event => {
  if (event.isComposing) return;
  if (event.target.id === 'question' && event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); generate('chat'); return; }
  if (event.target.matches('[role=tab]') && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
    event.preventDefault(); const tabs = ['chat', 'summary', 'quiz', 'flashcards']; const index = tabs.indexOf(state.tab);
    state.tab = event.key === 'Home' ? tabs[0] : event.key === 'End' ? tabs.at(-1) : tabs[(index + (event.key === 'ArrowRight' ? 1 : 3)) % 4];
    render(); $(`#tab-${state.tab}`).focus(); return;
  }
  if (state.tab !== 'flashcards' || !workspace().cards.length || document.querySelector('dialog[open]') || event.target.closest('input,textarea,select,a,[contenteditable]')) return;
  // Arrow shortcuts also work while the card is focused. Space on the card
  // already triggers its native button click, so avoid flipping it twice.
  const focusedCard = event.target.closest('.flashcard');
  if (event.target.closest('button') && !focusedCard) return;
  const w = workspace();
  if (event.key === ' ' && !focusedCard) { event.preventDefault(); w.flipped = !w.flipped; render(); }
  else if (['ArrowLeft', 'ArrowRight'].includes(event.key)) {
    event.preventDefault(); w.cardIndex = Math.max(0, Math.min(w.cards.length - 1, w.cardIndex + (event.key === 'ArrowRight' ? 1 : -1))); w.flipped = false; render();
    if (focusedCard) $('.flashcard')?.focus({ preventScroll: true });
  }
});
$('#source-search').addEventListener('input', renderSources);
$('#settings-button').addEventListener('click', showSettings);
$('#refresh-button').addEventListener('click', loadSources);
$('#upload-button').addEventListener('click', () => $('#file-input').click());
$('#file-input').addEventListener('change', event => upload([...event.target.files]));
$('#sources-button').addEventListener('click', showSources);
$('#sources-dialog').addEventListener('close', () => $('#app-shell').prepend($('#sources')));
matchMedia('(min-width: 901px)').addEventListener('change', event => { if (event.matches && $('#sources-dialog').open) $('#sources-dialog').close(); });
loadSources();
