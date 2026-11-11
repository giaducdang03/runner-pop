import { useEffect, useState } from "react";
import { Loader, Modal, Text, UnstyledButton } from "@mantine/core";
import type { Stage } from "../api/types";

interface Props {
  title: string;
  stages: Stage[];
  /** nhãn bước đang chạy (job còn running) -> thêm ô "đang chạy" cuối hàng */
  running?: string | null;
}

/** Một pipeline thành timeline dọc: mỗi bước là một nút tròn nối bằng đường thẳng; bấm ảnh nhỏ để phóng to (←/→ đổi bước, Esc đóng). */
export function PipelineStrip({ title, stages, running }: Props) {
  const [open, setOpen] = useState<number | null>(null);
  const total = stages.reduce((a, s) => a + s.seconds, 0);

  const step = (d: number) =>
    setOpen((i) => {
      if (i === null) return i;
      let j = i;
      do j = (j + d + stages.length) % stages.length;
      while (!stages[j].image && j !== i);
      return j;
    });

  useEffect(() => {
    if (open === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") step(1);
      else if (e.key === "ArrowLeft") step(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, stages.length]);

  if (!stages.length && !running) return null;
  const cur = open !== null ? stages[open] : null;

  return (
    <div className="tl-group">
      <p className="strip-head">
        {title}
        <span>{stages.length} bước, {total.toFixed(1)} giây</span>
      </p>
      <ol className="tl">
        {stages.map((s, i) => (
          <li className="tl-item" key={`${s.key}-${i}`}>
            <span className="tl-dot" aria-hidden="true">{i + 1}</span>
            <div className="tl-body">
              <span className="tl-name">{s.label}</span>
              <span className="tl-time">{s.seconds.toFixed(2)} giây{s.device ? `, ${s.device}` : ""}</span>
              {s.note && <span className="tl-note">{s.note}</span>}
            </div>
            {s.image && (
              <button type="button" className="tl-thumb" onClick={() => setOpen(i)} aria-label={`Phóng to bước ${i + 1}: ${s.label}`}>
                <img src={s.image} alt="" />
              </button>
            )}
          </li>
        ))}
        {running && (
          <li className="tl-item" data-running="true">
            <span className="tl-dot" aria-hidden="true"><Loader size={12} color="ink" /></span>
            <div className="tl-body">
              <span className="tl-name">{running}</span>
              <span className="tl-time">Đang chạy</span>
            </div>
          </li>
        )}
      </ol>

      <Modal opened={cur !== null} onClose={() => setOpen(null)} size="auto" centered withCloseButton={false}
             overlayProps={{ backgroundOpacity: 0.85 }} padding="md">
        {cur && open !== null && (
          <div className="light">
            <Text fw={600} ta="center">{open + 1}. {cur.label}</Text>
            {cur.image && <img src={cur.image} alt={cur.label} />}
            {cur.note && <Text size="sm" c="dimmed" ta="center" maw="80vw">{cur.note}</Text>}
            <div className="light-nav">
              <UnstyledButton fz="xl" px="sm" onClick={() => step(-1)} aria-label="Bước trước">‹</UnstyledButton>
              Phím mũi tên để chuyển bước, Esc để đóng
              <UnstyledButton fz="xl" px="sm" onClick={() => step(1)} aria-label="Bước sau">›</UnstyledButton>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
