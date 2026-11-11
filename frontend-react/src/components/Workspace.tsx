import { useEffect, useRef, useState } from "react";
import { Alert, Button, Loader, SegmentedControl } from "@mantine/core";
import { postJSON, sessionUrl } from "../api/client";
import type { DetectResult, Stage } from "../api/types";
import { useHealth, useOptions } from "../hooks/useBackend";
import { useJob } from "../hooks/useJob";
import { shouldAutoRun } from "../lib/uiHelpers";
import { engineInfo } from "../lib/engineInfo";
import { useSettings } from "../store/settings";
import { BgFilterGrid } from "./BgFilterGrid";
import { BoxDrawer, type BoxT } from "./BoxDrawer";
import { CompareViewer } from "./CompareViewer";
import { BibTag, cleanBib, ExportActions } from "./ResultControls";
import { StepsPanel } from "./StepsPanel";

const PEOPLE_LIMIT = 5;

export interface ProcessResult {
  session_id: string;
  person: number;
  method: string;
  elapsed_s: number;
  unknown_px: number;
  bib: string;
  focus_confidence: number | null;
}

interface Props {
  sid: string;
  det: DetectResult;
  detectStages: Stage[];
  ready: boolean; // engine (và model gậy) đã nạp xong
}

/** Từ khi có kết quả phát hiện: chọn người, tách nền, chỉnh hiệu ứng, tải về. Mount với key={sid} để state reset theo ảnh. */
export function Workspace({ sid, det, detectStages, ready }: Props) {
  const { options } = useOptions();
  const { warming } = useHealth();
  const { auto, method, band, focus, device } = useSettings();
  const { persons, width: W, height: H } = det;

  const [choice, setChoice] = useState(Math.min(det.suggested, Math.max(0, persons.length - 1)));
  const [drawing, setDrawing] = useState(persons.length === 0);
  const [box, setBox] = useState<BoxT | null>(null);
  const manualBox = drawing && box ? box.join(",") : null;
  const hasTarget = drawing ? !!box : persons.length > 0;

  const filters = options.bg_filters?.length ? options.bg_filters : [{ name: "bw", label: "Nền đen trắng" }];
  const [bgFilterRaw, setBgFilter] = useState("bw");
  const bgFilter = filters.some((f) => f.name === bgFilterRaw) ? bgFilterRaw : "bw";
  const [bgColor, setBgColor] = useState("#808080");
  const [intensity, setIntensity] = useState(100);
  const [bibValue, setBibValue] = useState("");

  const [run, setRun] = useState(0); // tăng mỗi khi một lượt /process được nhận -> useJob theo dõi lượt mới
  const [startError, setStartError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const seg = useJob<ProcessResult>(run > 0 ? sid : null, "segment", run);
  const busy = starting || seg.state === "running";

  // Giữ kết quả gần nhất trong lúc chạy lại, để cột phải không nhảy; `ver` chỉ đổi khi cutout mới đã có ở backend
  const [done, setDone] = useState<{ res: ProcessResult; ver: number } | null>(null);
  const [view, setView] = useState<"select" | "result">("select");
  useEffect(() => {
    if (seg.state === "done" && seg.result) {
      setDone({ res: seg.result, ver: run });
      setBibValue(seg.result.bib);
      setView("result");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seg.state]);

  const start = async () => {
    setStartError(null);
    setStarting(true);
    const params: Record<string, string | number | boolean> = { person: choice, method, band, focus, device, background: true };
    if (manualBox) params.box = manualBox;
    try {
      await postJSON(`/process/${sid}`, params);
      setRun((n) => n + 1);
    } catch (e) {
      setStartError((e as Error).message);
    } finally {
      setStarting(false);
    }
  };

  // Chế độ tự động: chạy 1 lần cho mỗi tổ hợp chưa xử lý; đang có lượt chạy thì chờ xong mới chạy tổ hợp mới nhất.
  const sig = [sid, choice, manualBox, method, band, method === "focus" ? focus : null, device].join("|");
  const lastSig = useRef<string | null>(null);
  useEffect(() => {
    if (shouldAutoRun(auto, hasTarget, sig, lastSig.current, warming || busy, ready && !!method)) {
      lastSig.current = sig; // đặt trước để lỗi không làm lặp vô hạn
      void start();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig, auto, warming, busy, ready, hasTarget]);

  const engineLabel = engineInfo(method, options.engines.find((e) => e.name === method)?.label).title;
  const segErr = startError ?? (seg.state === "error" ? seg.error : null);
  const showResult = view === "result" && !!done && !busy;
  const lowFocus = done?.res.focus_confidence != null && done.res.focus_confidence < (options.focus_conf_min ?? 1.3);

  // Ảnh đông người: chỉ hiện vài người đầu danh sách (và người đang chọn), những người còn lại bấm trực tiếp trên ảnh được
  const [allPeople, setAllPeople] = useState(false);
  const shownPersons = allPeople ? persons : persons.filter((p, i) => i < PEOPLE_LIMIT || p.idx === choice);

  const pickPerson = (i: number) => {
    setDrawing(false);
    setChoice(i);
    setView("select");
  };

  return (
    <div className="layout">
      <div className="main">
        {segErr && <Alert color="red">{segErr}</Alert>}
        {lowFocus && !busy && (
          <Alert color="yellow" p="xs">Ảnh này không có phông mờ rõ nên kết quả có thể kém chính xác. Hãy thử thuật toán khác trong Cài đặt.</Alert>
        )}

        <div className="preview-row">
          <div className="preview">
            {done && (
              <div className="preview-head">
                {!busy ? (
                  <SegmentedControl size="xs" value={view} onChange={(v) => setView(v as typeof view)}
                                    data={[{ value: "result", label: "Kết quả" }, { value: "select", label: "Chọn người" }]} />
                ) : <span />}
                <div className={busy ? "is-stale" : undefined}>
                  <BibTag value={bibValue} detected={!!done.res.bib} onChange={setBibValue} />
                </div>
              </div>
            )}
            {/* Viewer giữ nguyên trong DOM để không mất chế độ xem khi chạy lại */}
            {done && (
              <div className="preview-fill" hidden={!showResult}>
                <CompareViewer sid={sid} ver={done.ver} filter={bgFilter} color={bgColor} intensity={intensity} onIntensityCommit={setIntensity} />
              </div>
            )}
            {!showResult && (
              <div className="stage">
                <BoxDrawer src={sessionUrl(sid, "original.jpg")} width={W} height={H} persons={persons} choice={choice} onPick={pickPerson}
                           drawing={drawing} box={box} onCommit={setBox} dim={busy}>
                  {busy && (
                    <div className="busy">
                      <Loader size={14} color="ink" />
                      {seg.running ?? "Đang tách nền"}
                    </div>
                  )}
                </BoxDrawer>
              </div>
            )}
          </div>
          <StepsPanel detect={detectStages} segment={seg.stages} segmentRunning={busy ? seg.running : null} engineLabel={engineLabel}
                      quality={det.quality} applied={det.preprocess} debug={done && !busy ? { sid, ver: done.ver } : null} />
        </div>
      </div>

      <aside className="rail">
        <div className="rail-body">
          <div>
            <h3 className="rail-title">Người chạy</h3>
            <div className="chips">
              {shownPersons.map((p) => (
                <button key={p.idx} type="button" className="chip" data-on={!drawing && p.idx === choice} aria-pressed={!drawing && p.idx === choice}
                        onClick={() => pickPerson(p.idx)}>
                  Người {p.idx + 1}<small>{Math.round(p.conf * 100)}%</small>
                </button>
              ))}
              {persons.length > PEOPLE_LIMIT && (
                <button type="button" className="chip" aria-expanded={allPeople} onClick={() => setAllPeople((v) => !v)}>
                  {allPeople ? "Thu gọn" : `Thêm ${persons.length - shownPersons.length} người`}
                </button>
              )}
              <button type="button" className="chip" data-on={drawing} aria-pressed={drawing} onClick={() => { setDrawing(true); setView("select"); }}>
                Tự vẽ khung
              </button>
            </div>
            {persons.length === 0 && <p className="rail-note" style={{ marginTop: 10 }}>Không tìm thấy người trong ảnh. Hãy kéo chuột quanh người chạy để vẽ khung.</p>}
            <Button mt="sm" size="xs" variant={auto ? "default" : "filled"} onClick={start} loading={busy} disabled={!ready || !method || !hasTarget}>
              {auto ? "Tách nền lại" : "Tách nền"}
            </Button>
          </div>

          {done && (
            <div className={busy ? "rail-scroll is-stale" : "rail-scroll"}>
              <BgFilterGrid sid={sid} ver={done.ver} filters={filters} active={bgFilter} color={bgColor}
                            onSelect={setBgFilter} onColorCommit={setBgColor} />
            </div>
          )}
        </div>

        {done && (
          <div className="rail-foot">
            <ExportActions sid={sid} ver={done.ver} bib={cleanBib(bibValue)} filter={bgFilter} color={bgColor} intensity={intensity} disabled={busy} />
          </div>
        )}
      </aside>
    </div>
  );
}
