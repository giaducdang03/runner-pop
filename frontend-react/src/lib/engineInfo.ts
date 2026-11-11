/** Tên ngắn + mô tả dễ hiểu cho từng thuật toán tách nền; engine lạ (backend thêm mới) dùng nhãn backend gửi. */
export interface EngineInfo {
  title: string;
  desc: string;
  badge?: string;
}

const INFO: Record<string, EngineInfo> = {
  birefnet: {
    title: "BiRefNet",
    badge: "Khuyên dùng",
    desc: "Dùng AI để tách người khỏi nền, viền tóc và vải mượt. Chậm nhất, hợp hầu hết các ảnh.",
  },
  cf: {
    title: "GrabCut + Closed-form",
    badge: "Nhanh",
    desc: "Tách nhanh, viền khá mượt. Hợp ảnh có nền đơn giản, người nổi rõ.",
  },
  knn: {
    title: "GrabCut + KNN",
    desc: "Giống Closed-form nhưng làm mịn viền theo cách khác, chậm hơn một chút.",
  },
  focus: {
    title: "Mặt phẳng nét",
    badge: "Ảnh xoá phông",
    desc: "Giữ màu vùng nét, vùng mờ về đen trắng. Chỉ hợp ảnh chụp xoá phông.",
  },
};

export const engineInfo = (name: string, fallbackLabel?: string): EngineInfo =>
  INFO[name] ?? { title: fallbackLabel ?? name, desc: "" };
