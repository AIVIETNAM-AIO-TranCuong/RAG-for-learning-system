// Explicitly fictional fixtures for UI exploration. They never enter Qdrant.
export const sampleDocuments = [
  { document_id: 'sample-alignment', filename: 'Căn chỉnh mô hình ngôn ngữ.pdf', total_chunks: 40, max_page: 17 },
  { document_id: 'sample-rag', filename: 'Retrieval-Augmented Generation.pdf', total_chunks: 32, max_page: 14 },
  { document_id: 'sample-notes', filename: 'Ghi chú học tập · Tuần 03.md', total_chunks: 12, max_page: 1 },
];

export const openingQuestion = 'RLHF và DPO khác nhau như thế nào?';
export const sampleAnswer = {
  answer: 'Cả **RLHF** và **DPO** đều hướng mô hình đến những câu trả lời phù hợp hơn với sở thích của con người. Điểm khác biệt nằm ở cách tối ưu.\n\n### RLHF: học qua một mô hình phần thưởng\nQuy trình thường gồm ba bước: tinh chỉnh có giám sát, huấn luyện mô hình phần thưởng từ các cặp câu trả lời được đánh giá, rồi tối ưu chính sách bằng học tăng cường. [S1]\n\n### DPO: tối ưu trực tiếp từ sở thích\nDPO dùng trực tiếp các cặp phản hồi được ưu tiên và không được ưu tiên để tinh chỉnh mô hình. Cách này không cần huấn luyện một mô hình phần thưởng riêng hay chạy vòng lặp học tăng cường như quy trình RLHF truyền thống. [S2]\n\n**Điều cần nhớ:** RLHF có thêm bước học phần thưởng và tối ưu chính sách; DPO đơn giản hóa quy trình bằng một mục tiêu tối ưu trực tiếp.',
  citations: [
    { marker: 'S1', chunk: { content: 'Ví dụ minh họa: Một quy trình RLHF phổ biến bắt đầu bằng tinh chỉnh có giám sát, sau đó học mô hình phần thưởng từ đánh giá sở thích và tối ưu chính sách bằng học tăng cường.', metadata: { filename: sampleDocuments[0].filename, document_id: 'sample-alignment', page: 6 }, score: .91 } },
    { marker: 'S2', chunk: { content: 'Ví dụ minh họa: DPO biểu diễn việc học từ sở thích bằng một mục tiêu tối ưu trực tiếp trên các cặp phản hồi, tránh yêu cầu mô hình phần thưởng riêng trong vòng lặp huấn luyện.', metadata: { filename: sampleDocuments[0].filename, document_id: 'sample-alignment', page: 9 }, score: .88 } },
  ],
};

const quizPool = [
  ['Mục tiêu chính của việc căn chỉnh mô hình ngôn ngữ là gì?', ['A. Làm cho đầu ra phù hợp hơn với mong muốn của con người', 'B. Tăng kích thước tệp mô hình', 'C. Loại bỏ toàn bộ dữ liệu huấn luyện', 'D. Thay thế mọi bước đánh giá'], 'A', 'Căn chỉnh hướng mô hình đến hành vi hữu ích và phù hợp hơn với mong muốn của con người.'],
  ['Trong quy trình RLHF phổ biến, mô hình phần thưởng học từ đâu?', ['A. Dung lượng GPU', 'B. Các cặp phản hồi được con người đánh giá', 'C. Tên tài liệu', 'D. Tốc độ kết nối mạng'], 'B', 'Mô hình phần thưởng thường được huấn luyện bằng dữ liệu sở thích: phản hồi được chọn và phản hồi bị từ chối.'],
  ['Điểm đơn giản hóa của DPO so với RLHF truyền thống là gì?', ['A. Không cần dữ liệu', 'B. Không cần mô hình ngôn ngữ', 'C. Không cần một mô hình phần thưởng riêng trong vòng lặp tối ưu', 'D. Luôn cho kết quả chính xác tuyệt đối'], 'C', 'DPO tối ưu trực tiếp từ các cặp sở thích, đơn giản hóa quy trình huấn luyện.'],
  ['RAG bổ sung thông tin cho mô hình chủ yếu bằng cách nào?', ['A. Tăng độ dài tên mô hình', 'B. Đổi màu giao diện', 'C. Xóa lịch sử truy xuất', 'D. Truy xuất các đoạn tài liệu liên quan làm ngữ cảnh'], 'D', 'RAG truy xuất ngữ cảnh phù hợp rồi đưa vào prompt để hỗ trợ sinh câu trả lời.'],
  ['Trích dẫn nguồn giúp người học làm gì?', ['A. Đối chiếu câu trả lời với tài liệu gốc', 'B. Bỏ qua việc kiểm chứng', 'C. Tự động thay đổi tài liệu', 'D. Đảm bảo mọi kết luận đều đúng'], 'A', 'Trích dẫn giúp truy về đoạn nguồn và kiểm tra tính phù hợp của câu trả lời.'],
  ['Fine-tuning khác retrieval ở điểm nào?', ['A. Hai khái niệm hoàn toàn giống nhau', 'B. Fine-tuning cập nhật trọng số, retrieval bổ sung ngữ cảnh', 'C. Retrieval luôn cập nhật trọng số', 'D. Fine-tuning chỉ đổi tên tệp'], 'B', 'Fine-tuning làm thay đổi trọng số mô hình; retrieval tìm dữ liệu để bổ sung vào ngữ cảnh khi suy luận.'],
  ['Vì sao cần tách tài liệu thành các chunk?', ['A. Để xóa nội dung', 'B. Để thay thế câu hỏi', 'C. Để truy xuất những đoạn có dung lượng phù hợp', 'D. Để tránh mọi phép đo'], 'C', 'Chunking tạo các đoạn vừa đủ nhỏ để truy xuất và đưa vào cửa sổ ngữ cảnh.'],
  ['Reranker thường làm gì sau retrieval?', ['A. Huấn luyện lại toàn bộ LLM', 'B. Nén toàn bộ tệp PDF', 'C. Xóa câu hỏi', 'D. Sắp xếp lại các đoạn theo độ liên quan'], 'D', 'Reranker đánh giá lại các ứng viên được truy xuất để chọn các đoạn phù hợp hơn với câu hỏi.'],
  ['Một cặp dữ liệu sở thích thường có thành phần nào?', ['A. Cùng prompt, phản hồi được chọn và phản hồi bị từ chối', 'B. Chỉ có tên mô hình', 'C. Chỉ có trang đầu PDF', 'D. Hai tệp âm thanh ngẫu nhiên'], 'A', 'Dữ liệu sở thích thường so sánh hai phản hồi cho cùng một prompt.'],
  ['Vì sao vẫn cần đánh giá sau khi căn chỉnh?', ['A. Để bỏ hết dữ liệu kiểm thử', 'B. Để đo chất lượng và phát hiện hành vi chưa mong muốn', 'C. Để đổi loại tài liệu', 'D. Vì mọi phản hồi đều như nhau'], 'B', 'Đánh giá giúp xác định mức cải thiện và phát hiện các giới hạn còn lại.'],
];
const cardPool = [
  ['RLHF là gì?', 'Reinforcement Learning from Human Feedback: học tăng cường dựa trên phản hồi của con người.'],
  ['DPO là gì?', 'Direct Preference Optimization: tối ưu trực tiếp từ dữ liệu sở thích.'],
  ['RLHF và DPO khác nhau ở đâu?', 'RLHF truyền thống học mô hình phần thưởng rồi tối ưu chính sách; DPO tối ưu trực tiếp từ các cặp sở thích.'],
  ['SFT là gì?', 'Supervised Fine-Tuning: tinh chỉnh mô hình trên các ví dụ đầu vào và đầu ra được gán nhãn.'],
  ['Dữ liệu sở thích gồm những gì?', 'Một prompt cùng phản hồi được ưu tiên và phản hồi bị từ chối.'],
  ['Mô hình phần thưởng làm gì?', 'Ước lượng mức độ phù hợp của một phản hồi theo dữ liệu đánh giá đã học.'],
  ['RAG là gì?', 'Retrieval-Augmented Generation: kết hợp truy xuất tài liệu và sinh câu trả lời.'],
  ['Embedding dùng để làm gì?', 'Biểu diễn nội dung thành vector để so sánh ngữ nghĩa và tìm kiếm.'],
  ['Chunking là gì?', 'Chia tài liệu thành các đoạn nhỏ phù hợp để truy xuất và dựng ngữ cảnh.'],
  ['Reranking là gì?', 'Đánh giá lại độ liên quan để sắp xếp các kết quả truy xuất.'],
  ['Vì sao câu trả lời cần trích dẫn?', 'Để người đọc có thể kiểm tra kết luận trong ngữ cảnh tài liệu nguồn.'],
  ['Retrieval có thay đổi trọng số mô hình không?', 'Thông thường không. Retrieval bổ sung thông tin vào ngữ cảnh khi suy luận.'],
  ['Fine-tuning thay đổi điều gì?', 'Fine-tuning cập nhật một phần hoặc toàn bộ trọng số của mô hình.'],
  ['Cửa sổ ngữ cảnh là gì?', 'Lượng token mô hình có thể nhận và sử dụng trong một lượt xử lý.'],
  ['Hallucination là gì?', 'Nội dung được mô hình sinh ra nhưng không có căn cứ hoặc không chính xác.'],
];

const delay = value => new Promise(resolve => setTimeout(() => resolve(structuredClone(value)), 450));
export function createSampleApi() {
  return {
    health: () => delay({ status: 'sample', llm_configured: true }),
    documents: () => delay(sampleDocuments),
    ask: (_question, documentId) => {
      const filename = sampleDocuments.find(doc => doc.document_id === documentId)?.filename || sampleDocuments[0].filename;
      const result = structuredClone(sampleAnswer);
      result.citations.forEach(citation => { citation.chunk.metadata.filename = filename; });
      return delay(result);
    },
    summarize: documentId => delay({ document_id: documentId, method: 'single', total_chunks: 40,
      content: '## Căn chỉnh mô hình ngôn ngữ\n\nTài liệu minh họa giới thiệu cách hướng hành vi của mô hình ngôn ngữ đến những phản hồi hữu ích và phù hợp hơn với con người.\n\n### 1. Nền tảng: tinh chỉnh có giám sát\nMô hình học cách làm theo chỉ dẫn từ các ví dụ đầu vào và phản hồi được lựa chọn. Đây thường là bước khởi đầu trước khi tối ưu theo sở thích.\n\n### 2. Hai hướng học từ sở thích\n- **RLHF:** học mô hình phần thưởng, sau đó tối ưu chính sách bằng học tăng cường.\n- **DPO:** học trực tiếp từ các cặp phản hồi được ưu tiên và bị từ chối.\n\n### 3. Kết nối với hệ thống RAG\nRAG bổ sung các đoạn tài liệu liên quan vào ngữ cảnh khi trả lời. Retrieval chọn ứng viên, reranker xếp lại độ liên quan, còn trích dẫn giúp kiểm chứng đầu ra.\n\n### Ghi nhớ\nCăn chỉnh cải thiện hành vi; retrieval bổ sung kiến thức vào ngữ cảnh. Cả hai vẫn cần được đánh giá bằng những bài kiểm tra phù hợp.' }),
    quiz: (documentId, topic, count) => delay({ document_id: documentId, topic,
      items: quizPool.slice(0, count).map(([question, options, answer, explanation]) => ({ question, options, answer, explanation })) }),
    flashcards: (documentId, topic, count) => delay({ document_id: documentId, topic,
      cards: cardPool.slice(0, count).map(([front, back]) => ({ front, back })) }),
    upload: () => Promise.reject(new Error('Chuyển sang Kết nối API để tải tài liệu thật. Dữ liệu mẫu không ghi vào backend.')),
  };
}
