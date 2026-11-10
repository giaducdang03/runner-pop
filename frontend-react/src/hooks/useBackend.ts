import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getJSON } from "../api/client";
import type { Health, Options } from "../api/types";
import { pickDefaultEngine } from "../lib/uiHelpers";
import { useSettings } from "../store/settings";

const FALLBACK: Options = { engines: [{ name: "cf", label: "GrabCut + Closed-form" }], preprocess: [] };

/** Engine / bước tiền xử lý lấy từ backend: thêm bước mới ở backend là giao diện tự có. */
export function useOptions() {
  const q = useQuery({ queryKey: ["options"], queryFn: () => getJSON<Options>("/options"), staleTime: 60_000 });
  const options = q.data ?? FALLBACK; // backend chưa sẵn sàng -> danh sách dự phòng
  const method = useSettings((s) => s.method);
  const set = useSettings((s) => s.set);

  useEffect(() => {
    const names = options.engines.map((e) => e.name);
    if (!names.includes(method)) set({ method: pickDefaultEngine(options, names) });
  }, [options, method, set]);

  return { options, isError: q.isError };
}

/** /health: poll 2 s khi đang warm-up, 15 s khi đã xong. */
export function useHealth() {
  const q = useQuery({
    queryKey: ["health"],
    queryFn: () => getJSON<Health>("/health"),
    refetchInterval: (query) => (query.state.data?.warmup === "running" ? 2000 : 15_000),
  });
  return { health: q.data, isError: q.isError, warming: q.data?.warmup === "running" };
}
