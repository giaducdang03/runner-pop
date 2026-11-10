import { create } from "zustand";

export type PreMode = "auto" | "none" | "custom";

export interface Settings {
  auto: boolean;
  preMode: PreMode;
  preSteps: string[];
  findPoles: boolean;
  method: string; // rỗng tới khi /options tải xong, rồi lấy engine mặc định của backend
  band: number;
  focus: number;
  device: string;
  set: (patch: Partial<Omit<Settings, "set">>) => void;
}

export const useSettings = create<Settings>((set) => ({
  auto: true,
  preMode: "auto",
  preSteps: [],
  findPoles: false,
  method: "",
  band: 0.02,
  focus: 0.3,
  device: "auto",
  set: (patch) => set(patch),
}));

/** Giá trị `preprocess` gửi cho POST /detect (giống pre_spec của bản Streamlit). */
export const preSpec = (s: Pick<Settings, "preMode" | "preSteps">) =>
  s.preMode === "custom" ? s.preSteps.join(",") || "none" : s.preMode;
