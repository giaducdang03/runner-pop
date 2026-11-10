import { useEffect, useState } from "react";
import { postJSON } from "../api/client";
import type { DetectResult } from "../api/types";
import { preSpec, useSettings } from "../store/settings";
import { useJob } from "./useJob";

/**
 * Tải ảnh -> POST /detect?background=true -> theo dõi job "detect".
 * Phát hiện lại khi đổi ảnh, cấu hình tiền xử lý hoặc công tắc gậy (không phải khi đổi device/engine),
 * và chỉ bắt đầu khi `canStart` (model gậy đã nạp xong nếu đang bật).
 */
export function useDetect(file: File | null, canStart: boolean) {
  const pre = useSettings(preSpec);
  const findPoles = useSettings((s) => s.findPoles);
  const [sid, setSid] = useState<string | null>(null);
  const [startError, setStartError] = useState<string | null>(null);

  useEffect(() => {
    setSid(null);
    setStartError(null);
    if (!file || !canStart) return;
    let cancelled = false;
    const body = new FormData();
    body.append("file", file);
    const { device } = useSettings.getState();
    postJSON<{ session_id: string }>("/detect", { preprocess: pre, device, background: true, poles: findPoles }, body)
      .then((r) => !cancelled && setSid(r.session_id))
      .catch((e: Error) => !cancelled && setStartError(e.message));
    return () => {
      cancelled = true;
    };
  }, [file, canStart, pre, findPoles]);

  const job = useJob<DetectResult>(sid, "detect");
  const state = startError ? "error" : sid || !file ? job.state : canStart ? "running" : "idle";
  return { sid, job, state, error: startError ?? job.error, det: job.result };
}
