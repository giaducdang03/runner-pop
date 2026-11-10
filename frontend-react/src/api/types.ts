export interface Engine {
  name: string;
  label: string;
}
export interface Options {
  engines: Engine[];
  preprocess: { name: string; label: string }[];
  devices?: string[];
  default_engine?: string;
  bg_filters?: { name: string; label: string }[];
  focus_conf_min?: number;
}
export interface Health {
  yolo_available: boolean;
  warmup?: "idle" | "running" | "done" | "error";
  warmup_detail?: string;
  device?: { name?: string; backend?: string; runtime?: string; dtype?: string };
  available_backends?: Record<string, string>;
}
export interface Prepare {
  state?: "idle" | "running" | "done" | "error";
  detail?: string;
  used_device?: string;
}

/** Một bước pipeline (backend/pipeline/trace.py::Trace.to_list); `image` là data URI JPEG hoặc null. */
export interface Stage {
  key: string;
  label: string;
  seconds: number;
  note: string;
  device: string;
  image: string | null;
}
export interface JobStatus<R> {
  state: "running" | "done" | "error";
  error: string | null;
  status: number;
  running: string | null;
  stages: Stage[];
  result: R | null;
}
export interface Person {
  idx: number;
  box: [number, number, number, number];
  conf: number;
}
export interface DetectResult {
  session_id: string;
  width: number;
  height: number;
  persons: Person[];
  quality: { flags?: string[]; brightness: number; contrast: number; sharpness: number; noise: number };
  preprocess: string[];
  attempt: number;
  suggested: number;
}
