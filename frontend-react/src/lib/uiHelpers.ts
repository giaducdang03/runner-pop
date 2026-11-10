// Port từ frontend/ui_helpers.py (hàm thuần, test bằng vitest).
export type PrepState = "idle" | "running" | "done" | "error" | undefined | null;

export const isWarming = (s?: string | null) => s === "running";
export const shouldPoll = isWarming;
export const engineReady = (s: PrepState) => s !== "idle" && s !== "running";
export const shouldPollPrepare = (s: PrepState) => s === "idle" || s === "running";
export const shouldPostPrepare = (s: PrepState) => s === "idle";

export function pickDefaultEngine(opts: { default_engine?: string }, names: string[]): string {
  return opts.default_engine && names.includes(opts.default_engine) ? opts.default_engine : names[0];
}

export function shouldAutoRun(
  auto: boolean,
  hasTarget: boolean,
  sig: string,
  last: string | null,
  warming: boolean,
  ready = true,
): boolean {
  return auto && hasTarget && !warming && ready && last !== sig;
}

export function prepareNote(deviceMode: string, used: string): string {
  return deviceMode === "gpu" && used.trim().toUpperCase().startsWith("CPU")
    ? `Đã dùng CPU thay vì GPU (${used}).`
    : "";
}
