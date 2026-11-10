import { ColorInput } from "@mantine/core";
import { useQueries } from "@tanstack/react-query";
import { getJSON } from "../api/client";

export interface BgFilter {
  name: string;
  label: string;
}

interface Props {
  sid: string;
  ver: number;
  filters: BgFilter[];
  active: string;
  color: string; // màu đã chốt (dùng cho ảnh xem trước)
  onSelect: (name: string) => void;
  onColorCommit: (c: string) => void; // thả bảng màu mới gọi lại backend
}

interface PreviewItem {
  name: string;
  image: string; // data URI
}

/** Lưới hiệu ứng nền: ảnh xem trước tải song song (cache theo phiên, lần tách, màu), bấm để chọn. */
export function BgFilterGrid({ sid, ver, filters, active, color, onSelect, onColorCommit }: Props) {
  const previews = useQueries({
    queries: filters.map((f) => ({
      queryKey: ["bgpreview", sid, ver, color, f.name],
      queryFn: async () => {
        const r = await getJSON<PreviewItem[]>(`/session/${sid}/bgfilter_previews`, { color, names: f.name });
        return r.find((p) => p.name === f.name)?.image ?? "";
      },
      staleTime: Infinity,
      retry: false, // lỗi -> ô chỉ còn nhãn, không thử lại mãi
    })),
  });

  return (
    <div>
      <h3 className="rail-title">Hiệu ứng nền</h3>
      <div className="tiles">
        {filters.map((f, i) => (
          <button key={f.name} type="button" className="tile" data-on={f.name === active} aria-pressed={f.name === active} onClick={() => onSelect(f.name)}>
            {previews[i]?.data ? <img src={previews[i].data} alt="" /> : <div className="tile-blank" />}
            <span className="tile-label">{f.label}</span>
          </button>
        ))}
      </div>
      {active === "solid" && <ColorInput mt="sm" label="Màu nền" defaultValue={color} onChangeEnd={onColorCommit} />}
    </div>
  );
}
