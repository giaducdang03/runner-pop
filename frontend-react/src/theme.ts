import { createTheme, type MantineColorsTuple } from "@mantine/core";

const ink: MantineColorsTuple = [
  "#F3F4F5", "#E3E4E6", "#C9CBCF", "#A9ACB1", "#85888E", "#5F6267", "#1C1D1F", "#151617", "#0F1011", "#08090A",
];
const bib: MantineColorsTuple = [
  "#FFFBE0", "#FFF5B3", "#FFEE80", "#FFE64D", "#FFDF26", "#FFDA14", "#FFD60A", "#E6C000", "#C7A600", "#A38800",
];

// Giao diện chỉ có xám; vàng số áo dành riêng cho thứ đang được chọn và nút tải (giống ảnh: chỉ người được chọn có màu).
export const theme = createTheme({
  primaryColor: "ink",
  primaryShade: 6,
  colors: { ink, bib },
  autoContrast: true,
  fontFamily: "'Be Vietnam Pro', system-ui, sans-serif",
  defaultRadius: "md",
  headings: { fontFamily: "'Be Vietnam Pro', system-ui, sans-serif", fontWeight: "600" },
});
