import { useRef, useState, type ReactNode } from "react";
import type { Person } from "../api/types";

export type BoxT = [number, number, number, number];

const MIN_SIDE = 20; // backend từ chối box nhỏ hơn

interface Props {
  src: string;
  width: number; // kích thước ảnh làm việc (px): mọi box tính theo hệ toạ độ này
  height: number;
  persons: Person[];
  choice: number;
  onPick: (idx: number) => void;
  drawing: boolean; // true -> kéo chuột để vẽ khung thay vì chọn người
  box: BoxT | null;
  onCommit: (b: BoxT) => void; // chỉ gọi khi thả chuột, để chế độ tự động không chạy theo từng lần di chuột
  dim?: boolean; // đang xử lý: ảnh về xám
  children?: ReactNode;
}

const pct = (b: readonly number[], W: number, H: number) => ({
  left: `${(b[0] / W) * 100}%`,
  top: `${(b[1] / H) * 100}%`,
  width: `${((b[2] - b[0]) / W) * 100}%`,
  height: `${((b[3] - b[1]) / H) * 100}%`,
});

/** Ảnh kèm khung người chọn được bằng cách bấm trực tiếp, hoặc kéo chuột để tự vẽ khung. */
export function BoxDrawer({ src, width, height, persons, choice, onPick, drawing, box, onCommit, dim, children }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [start, setStart] = useState<[number, number] | null>(null);
  const [draft, setDraft] = useState<BoxT | null>(null);

  const pos = (e: React.PointerEvent): [number, number] => {
    const r = ref.current!.getBoundingClientRect();
    return [
      Math.round(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * width),
      Math.round(Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) * height),
    ];
  };
  const toBox = (a: [number, number], b: [number, number]): BoxT => [
    Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1]),
  ];

  const shown = draft ?? box;
  return (
    <div
      ref={ref}
      className="frame"
      data-dim={dim}
      style={drawing ? { cursor: "crosshair", touchAction: "none" } : undefined}
      onPointerDown={(e) => {
        if (!drawing) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        const p = pos(e);
        setStart(p);
        setDraft(toBox(p, p));
      }}
      onPointerMove={(e) => start && setDraft(toBox(start, pos(e)))}
      onPointerUp={() => {
        if (draft && draft[2] - draft[0] >= MIN_SIDE && draft[3] - draft[1] >= MIN_SIDE) onCommit(draft);
        setStart(null);
        setDraft(null);
      }}
    >
      <img src={src} draggable={false} alt="Ảnh đã tải lên, kèm khung những người phát hiện được" />
      {!drawing &&
        persons.map((p) => (
          <button key={p.idx} type="button" className="person" data-on={p.idx === choice} style={pct(p.box, width, height)}
                  onClick={() => onPick(p.idx)} aria-label={`Chọn người ${p.idx + 1}`}>
            <span>{p.idx + 1}</span>
          </button>
        ))}
      {drawing && shown && <div className="drawn" style={pct(shown, width, height)} />}
      {drawing && !shown && <div className="hint">Kéo chuột quanh người chạy để vẽ khung</div>}
      {children}
    </div>
  );
}
