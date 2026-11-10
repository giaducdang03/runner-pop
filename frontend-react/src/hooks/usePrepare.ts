import { useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getJSON, postJSON } from "../api/client";
import type { Prepare } from "../api/types";
import { engineReady, shouldPollPrepare, shouldPostPrepare } from "../lib/uiHelpers";

/**
 * Trạng thái nạp model của một cặp (engine, device): GET /prepare, poll 2 s khi idle/running.
 * Chỉ POST khi idle (lần đầu hoặc backend vừa khởi động lại); lỗi chỉ thử lại khi gọi `retry()`.
 * `engine` = "poles" cho model tìm gậy. `enabled=false` -> coi như sẵn sàng.
 */
export function usePrepare(engine: string | undefined, device: string, enabled = true) {
  const qc = useQueryClient();
  const on = enabled && !!engine;
  const key = ["prepare", engine, device];

  const q = useQuery({
    queryKey: key,
    enabled: on,
    // lỗi mạng / thiếu endpoint -> {} (không chặn giao diện), giống prepare_state bản Streamlit
    queryFn: () => getJSON<Prepare>("/prepare", { engine: engine!, device }).catch(() => ({}) as Prepare),
    refetchInterval: (query) => (shouldPollPrepare(query.state.data?.state) ? 2000 : false),
  });

  const post = useMutation({
    mutationFn: () => postJSON<Prepare>("/prepare", { engine: engine!, device }),
    onSuccess: (d) => qc.setQueryData(key, d),
  });

  const state = q.data?.state;
  useEffect(() => {
    if (on && shouldPostPrepare(state) && !post.isPending) post.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [on, state, engine, device]);

  return {
    state,
    detail: q.data?.detail ?? "",
    usedDevice: q.data?.used_device ?? "",
    polling: on && shouldPollPrepare(state),
    ready: !on || (!q.isPending && engineReady(state)),
    retry: () => post.mutate(),
  };
}
