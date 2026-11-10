import { Alert, Button, Loader, Group, Text } from "@mantine/core";
import { usePrepare } from "../hooks/usePrepare";
import { prepareNote } from "../lib/uiHelpers";

interface Props {
  prep: ReturnType<typeof usePrepare>;
  label: string;
  device: string;
}

/** Thông báo nạp model của một engine (hoặc model gậy): đang nạp / lỗi + thử lại / ghi chú fallback CPU. */
export function PrepareStatus({ prep, label, device }: Props) {
  if (prep.polling) {
    return (
      <Alert color="blue" p="xs">
        <Group gap="xs" wrap="nowrap" align="flex-start">
          <Loader size="xs" mt={2} />
          <Text size="xs">
            Đang nạp model cho '{label}' trên {device}… (lần đầu có thể mất vài phút do biên dịch)
          </Text>
        </Group>
      </Alert>
    );
  }
  if (prep.state === "error") {
    return (
      <Alert color="yellow" p="xs" title="Nạp model gặp lỗi">
        <Text size="xs">{prep.detail}. Vẫn có thể thử xử lý.</Text>
        <Button size="compact-xs" mt="xs" variant="light" onClick={prep.retry}>
          Thử nạp lại
        </Button>
      </Alert>
    );
  }
  if (prep.state === "done") {
    const note = prepareNote(device, prep.usedDevice);
    const text = note || (prep.usedDevice ? `Model '${label}' đã sẵn sàng (${prep.usedDevice}).` : "");
    return text ? <Text size="xs" c="dimmed">{text}</Text> : null;
  }
  return null;
}
