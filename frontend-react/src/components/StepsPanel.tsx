import { Image, SimpleGrid, Stack, Text } from "@mantine/core";
import { useLocalStorage } from "@mantine/hooks";
import { IconChevronRight } from "@tabler/icons-react";
import { sessionUrl } from "../api/client";
import type { DetectResult, Stage } from "../api/types";
import { PipelineStrip } from "./PipelineStrip";

interface Props {
  detect: Stage[];
  detectRunning?: string | null;
  segment?: Stage[];
  segmentRunning?: string | null;
  engineLabel?: string;
  quality?: DetectResult["quality"];
  applied?: string[];
  debug?: { sid: string; ver: number } | null;
}

/** Các bước xử lý: cột dọc cạnh ảnh preview, mở mặc định; nhớ trạng thái mở theo trình duyệt. */
export function StepsPanel({ detect, detectRunning, segment = [], segmentRunning, engineLabel, quality, applied, debug }: Props) {
  const [open, setOpen] = useLocalStorage({ key: "runnerpop-steps-side-open", defaultValue: true });
  const count = detect.length + segment.length;
  const total = [...detect, ...segment].reduce((a, s) => a + s.seconds, 0);

  return (
    <section className="steps" data-open={open}>
      <button type="button" className="steps-toggle" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <IconChevronRight size={16} />
        Các bước xử lý
        {count > 0 && <span>{count} bước, {total.toFixed(1)} giây</span>}
      </button>
      {open && (
        <div className="steps-body">
          {quality && (
            <Text size="xs" c="dimmed">
              Chất lượng ảnh: {quality.flags?.join(", ") || "tốt"}, độ sáng {quality.brightness.toFixed(0)}, tương phản {quality.contrast.toFixed(0)},
              độ nét {quality.sharpness.toFixed(0)}, nhiễu {quality.noise.toFixed(1)}. Tiền xử lý: {applied?.join(", ") || "không"}.
            </Text>
          )}
          <PipelineStrip title="Phát hiện người" stages={detect} running={detectRunning} />
          <PipelineStrip title={engineLabel ? `Tách nền bằng ${engineLabel}` : "Tách nền"} stages={segment} running={segmentRunning} />
          {debug && (
            <SimpleGrid cols={2} spacing="xs">
              <Stack gap={4}>
                <Image src={sessionUrl(debug.sid, "alpha.png", { v: debug.ver })} radius={4} alt="Alpha matte" />
                <Text size="xs" c="dimmed">Alpha matte</Text>
              </Stack>
              <Stack gap={4}>
                <Image src={sessionUrl(debug.sid, "trimap.png", { v: debug.ver })} radius={4} alt="Trimap" />
                <Text size="xs" c="dimmed">Trimap (xám là vùng chưa chắc chắn)</Text>
              </Stack>
            </SimpleGrid>
          )}
        </div>
      )}
    </section>
  );
}
