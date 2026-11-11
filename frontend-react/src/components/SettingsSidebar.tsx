import { useEffect, useState } from "react";
import { Alert, Badge, Divider, Group, MultiSelect, Select, SegmentedControl, Slider, Stack, Switch, Text } from "@mantine/core";
import { useHealth, useOptions } from "../hooks/useBackend";
import { usePrepare } from "../hooks/usePrepare";
import { engineInfo } from "../lib/engineInfo";
import { useSettings } from "../store/settings";
import { PrepareStatus } from "./PrepareStatus";

const PRE_MODES = [
  { value: "auto", label: "Tự động theo chất lượng ảnh" },
  { value: "none", label: "Không" },
  { value: "custom", label: "Tự chọn các bước" },
];

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <Stack gap={6}>
      <Text size="sm" fw={500}>{label}</Text>
      {children}
      {hint && <Text size="xs" c="dimmed">{hint}</Text>}
    </Stack>
  );
}

/** Slider chỉ ghi vào store khi thả tay, để chế độ tự động không tách nền lại theo từng nấc kéo. */
function CommitSlider({ value, onCommit, ...rest }: { value: number; onCommit: (v: number) => void; min: number; max: number; step: number; disabled?: boolean }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return <Slider {...rest} value={v} onChange={setV} onChangeEnd={onCommit} />;
}

/** Nội dung ngăn Cài đặt: các tuỳ chọn kỹ thuật, không cần đụng tới trong lúc dùng bình thường. */
export function SettingsSidebar() {
  const s = useSettings();
  const { options } = useOptions();
  const { health, isError, warming } = useHealth();

  const infoOf = (n: string) => engineInfo(n, options.engines.find((e) => e.name === n)?.label);
  const engineLabel = (n: string) => infoOf(n).title;
  const prep = usePrepare(s.method || undefined, s.device);
  const poles = usePrepare("poles", s.device, s.findPoles);
  const devices = options.devices ?? ["auto", "cpu", "gpu"];

  return (
    <Stack gap="lg">
      <Switch
        label="Tự động tách nền"
        description="Tự chọn người chạy chính và tách nền ngay sau khi tải ảnh."
        checked={s.auto}
        onChange={(e) => s.set({ auto: e.currentTarget.checked })}
      />

      <Field label="Cách tách nền" hint={infoOf(s.method).desc}>
        <Select
          data={options.engines.map((e) => ({ value: e.name, label: infoOf(e.name).title }))}
          value={s.method || null}
          allowDeselect={false}
          onChange={(v) => v && s.set({ method: v })}
          renderOption={({ option, checked }) => {
            const i = infoOf(option.value);
            return (
              <Stack gap={2} style={{ flex: 1 }}>
                <Group gap={8} wrap="nowrap">
                  <Text size="sm" fw={checked ? 600 : 500}>{i.title}</Text>
                  {i.badge && <Badge size="xs" variant="light" color="gray">{i.badge}</Badge>}
                </Group>
                {i.desc && <Text size="xs" c="dimmed">{i.desc}</Text>}
              </Stack>
            );
          }}
        />
      </Field>
      <PrepareStatus prep={prep} label={engineLabel(s.method)} device={s.device} />

      <Field label={`Độ rộng viền: ${s.band}`} hint="Tăng nếu tóc hoặc vải mỏng bị cắt, giảm nếu viền còn dính nền.">
        <CommitSlider min={0.005} max={0.08} step={0.005} value={s.band} onCommit={(v) => s.set({ band: v })} />
      </Field>

      {s.method === "focus" && (
        <Field label={`Ngưỡng độ nét: ${s.focus}`} hint="Tăng nếu giữ màu thừa cả vùng hơi mờ, giảm nếu mất vùng nét.">
          <CommitSlider min={0.1} max={1} step={0.05} value={s.focus} onCommit={(v) => s.set({ focus: v })} />
        </Field>
      )}

      <Divider />

      <Field label="Làm rõ ảnh trước khi tìm người" hint="Chỉ ảnh dùng để tìm người được chỉnh, ảnh kết quả luôn giữ màu gốc.">
        <Select data={PRE_MODES} value={s.preMode} allowDeselect={false} onChange={(v) => v && s.set({ preMode: v as typeof s.preMode })} />
        {s.preMode === "custom" && (
          <MultiSelect placeholder="Các bước, theo thứ tự chọn" data={options.preprocess.map((p) => ({ value: p.name, label: p.label }))}
                       value={s.preSteps} onChange={(v) => s.set({ preSteps: v })} />
        )}
      </Field>

      <Switch
        label="Giữ màu cả gậy cầm tay"
        description="Dò gậy leo núi của người đang chọn. Chậm hơn vài giây, đổi công tắc sẽ tìm người lại."
        checked={s.findPoles}
        onChange={(e) => s.set({ findPoles: e.currentTarget.checked })}
      />
      {s.findPoles && <PrepareStatus prep={poles} label="Tìm gậy (YOLO-pose + MobileSAM)" device={s.device} />}

      <Divider />

      <Field label="Thiết bị xử lý" hint="auto dùng GPU nếu có, không thì CPU.">
        <SegmentedControl data={devices} value={s.device} onChange={(v) => s.set({ device: v })} fullWidth />
      </Field>

      {isError && <Alert color="red" p="xs">Không kết nối được máy chủ. Hãy chạy: uvicorn backend.main:app --port 8000</Alert>}
      {health && (
        <Stack gap={4}>
          {!health.yolo_available && <Alert color="yellow" p="xs">Máy chủ chưa cài ultralytics nên không tìm được người.</Alert>}
          {health.device && (
            <Text size="xs" c="dimmed">
              Đang dùng {health.device.name ?? "?"} ({health.device.backend ?? "?"}, {health.device.dtype ?? "?"}).
            </Text>
          )}
          {health.available_backends && (
            <Text size="xs" c="dimmed">
              Khả dụng: {Object.entries(health.available_backends).map(([k, v]) => `${k} (${v})`).join(", ") || "chỉ CPU"}. Muốn đổi, hãy khởi động lại máy chủ với biến ACCEL_BACKEND.
            </Text>
          )}
          {warming && <Text size="xs" c="dimmed">Đang chuẩn bị model.</Text>}
          {health.warmup === "error" && <Alert color="yellow" p="xs">Chuẩn bị model lỗi, lần chạy đầu sẽ chậm: {health.warmup_detail}</Alert>}
        </Stack>
      )}
    </Stack>
  );
}
