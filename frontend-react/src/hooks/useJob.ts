import { useEffect, useState } from "react";
import { getJSON } from "../api/client";
import type { JobStatus, Stage } from "../api/types";

export interface JobView<R> {
  state: "idle" | "running" | "done" | "error";
  stages: Stage[];
  running: string | null;
  result: R | null;
  error: string | null;
}

const IDLE: JobView<never> = { state: "idle", stages: [], running: null, result: null, error: null };

/**
 * Theo dõi job nền /session/{sid}/job/{phase}: poll 250 ms, gom dần các bước mới (tham số `since`).
 * `sid` đổi (hoặc `runId` đổi, vd chạy lại tách nền trên cùng phiên) -> bắt đầu lại từ đầu.
 */
export function useJob<R>(sid: string | null, phase: "detect" | "segment", runId: number | string = 0): JobView<R> {
  const [view, setView] = useState<JobView<R>>(IDLE);

  useEffect(() => {
    if (!sid) {
      setView(IDLE);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    let stages: Stage[] = [];
    setView({ ...IDLE, state: "running" });

    const tick = async () => {
      try {
        const r = await getJSON<JobStatus<R>>(`/session/${sid}/job/${phase}`, { since: stages.length });
        if (cancelled) return;
        stages = [...stages, ...r.stages];
        setView({ state: r.state, stages, running: r.running, result: r.result, error: r.error });
        if (r.state === "running") timer = setTimeout(tick, 250);
      } catch (e) {
        if (!cancelled) setView({ state: "error", stages, running: null, result: null, error: (e as Error).message });
      }
    };
    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [sid, phase, runId]);

  return view;
}
