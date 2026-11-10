// Mọi lời gọi backend đi qua /api (Vite proxy khi dev, nginx khi chạy docker).
export const API = "/api";

type Params = Record<string, string | number | boolean>;

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const qs = (params?: Params) =>
  params ? "?" + new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])) : "";

async function fail(r: Response): Promise<never> {
  let msg = r.statusText;
  try {
    const j = await r.json();
    msg = j.detail ?? JSON.stringify(j);
  } catch {
    /* giữ statusText */
  }
  throw new ApiError(r.status, msg);
}

export async function getJSON<T>(path: string, params?: Params): Promise<T> {
  const r = await fetch(`${API}${path}${qs(params)}`);
  if (!r.ok) await fail(r);
  return r.json();
}

export async function postJSON<T>(path: string, params?: Params, body?: FormData): Promise<T> {
  const r = await fetch(`${API}${path}${qs(params)}`, { method: "POST", body });
  if (!r.ok) await fail(r);
  return r.json();
}

/** URL ảnh của phiên; thêm `v` (đổi mỗi lần xử lý xong) để trình duyệt không dùng ảnh cache cũ. */
export const sessionUrl = (sid: string, file: string, params: Params = {}) =>
  `${API}/session/${sid}/${file}${qs(params)}`;
