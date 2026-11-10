import { useRef, useState } from "react";
import { Alert, Button, Group } from "@mantine/core";
import { API, sessionUrl } from "../api/client";

export const cleanBib = (v: string) => v.replace(/[^0-9a-zA-Z]/g, "");

/** Số BIB in như tấm số áo; bấm vào để sửa. */
export function BibTag({ value, detected, onChange }: { value: string; detected: boolean; onChange: (v: string) => void }) {
  return (
    <label className="bib">
      <span className="bib-label">{detected ? "Số BIB, bấm để sửa" : "Không đọc được số BIB, hãy nhập tay"}</span>
      <input value={value} maxLength={8} placeholder="0000" inputMode="text" onChange={(e) => onChange(e.currentTarget.value)} />
    </label>
  );
}

interface ExportProps {
  sid: string;
  ver: number;
  bib: string; // đã lọc ký tự
  filter: string;
  color: string;
  intensity: number; // 0..100
  disabled?: boolean;
}

function save(url: string, name: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
}

/** Tải ảnh PNG / JPEG và báo cáo PDF, đúng hiệu ứng, cường độ, số BIB đang chọn. */
export function ExportActions({ sid, ver, bib, filter, color, intensity, disabled }: ExportProps) {
  const prefix = bib ? `BIB_${bib}` : "";
  const k = filter === "none" ? 1 : intensity / 100;
  const img = (fmt: "png" | "jpeg") => sessionUrl(sid, "compose", { intensity: k, filter, color, fmt });
  const imgName = (ext: string) => `${prefix || "result"}.${ext}`;

  const cache = useRef<{ key: string; url: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const downloadPdf = async () => {
    const name = prefix ? `${prefix}_report.pdf` : "color_pop_runner_report.pdf";
    const key = [sid, ver, filter, color, k, bib].join("|"); // PDF chỉ đúng với đúng các lựa chọn lúc tạo
    if (cache.current?.key === key) return save(cache.current.url, name);
    setBusy(true);
    setError(null);
    try {
      const qs = new URLSearchParams({ intensity: String(k), filter, color, bib });
      const r = await fetch(`${API}/session/${sid}/report.pdf?${qs}`);
      if (!r.ok) {
        let msg = r.statusText;
        try { msg = (await r.json()).detail ?? msg; } catch { /* giữ statusText */ }
        throw new Error(msg);
      }
      if (cache.current) URL.revokeObjectURL(cache.current.url);
      cache.current = { key, url: URL.createObjectURL(await r.blob()) };
      save(cache.current.url, name);
    } catch (e) {
      setError(`Không tạo được báo cáo PDF: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const off = disabled ? { "data-disabled": true, onClick: (e: React.MouseEvent) => e.preventDefault() } : {};
  return (
    <>
      {error && <Alert color="red" p="xs" mb="xs">{error}</Alert>}
      <Group gap={8} grow wrap="nowrap">
        <Button component="a" color="bib" href={img("png")} download={imgName("png")} {...off}>Tải PNG</Button>
        <Button component="a" variant="default" href={img("jpeg")} download={imgName("jpg")} {...off}>Tải JPEG</Button>
        <Button variant="default" loading={busy} disabled={disabled} onClick={downloadPdf}>Báo cáo</Button>
      </Group>
    </>
  );
}
