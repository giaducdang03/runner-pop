import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, SegmentedControl, Slider } from "@mantine/core";
import { sessionUrl } from "../api/client";

// Xử lý ở độ phân giải vừa đủ để hiển thị (canvas chỉ hiện tối đa ~1100 px) -> ít pixel hơn, kéo mượt hơn
const MAX_SIDE = 1100;

const load = (src: string) =>
  new Promise<HTMLImageElement>((res, rej) => {
    const i = new Image();
    i.onload = () => res(i);
    i.onerror = () => rej(new Error("Không tải được ảnh từ máy chủ. Phiên có thể đã hết hạn, hãy tải ảnh lại."));
    i.src = src;
  });

const px = (i: CanvasImageSource, w: number, h: number) => {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const x = c.getContext("2d", { willReadFrequently: true })!;
  x.drawImage(i, 0, 0, w, h);
  return x.getImageData(0, 0, w, h).data;
};

interface Buf {
  io: HTMLImageElement;
  W: number;
  H: number;
  N: number;
  Bp: Uint8ClampedArray;
  AL: Float32Array;
  C0: Float32Array; // a*F + (1-a)*B
  CD: Float32Array; // (1-a)*(Bf - B)
  out: ImageData;
}

type Mode = "split" | "side" | "after" | "before";
const MODES = [
  { value: "split", label: "So sánh" },
  { value: "side", label: "Cạnh nhau" },
  { value: "after", label: "Kết quả" },
  { value: "before", label: "Ảnh gốc" },
];

interface Props {
  sid: string;
  ver: number; // đổi mỗi lần tách nền xong -> nạp lại toàn bộ ảnh
  filter: string;
  color: string;
  intensity: number; // 0..100
  onIntensityCommit: (v: number) => void; // thả thanh -> báo cha (để tải ảnh / PDF dùng đúng cường độ)
}

/**
 * So sánh trước/sau ngay trong trình duyệt, không gọi lại backend khi kéo thanh.
 * out = a*F + (1-a)*((1-s)*B + s*Bf) = C0 + s*CD (cùng công thức với backend): mỗi pixel chỉ còn một phép nhân-cộng mỗi kênh.
 */
export function CompareViewer({ sid, ver, filter, color, intensity, onIntensityCommit }: Props) {
  const cvRef = useRef<HTMLCanvasElement>(null);
  const bufRef = useRef<Buf | null>(null);
  const revealed = useRef<number | null>(null);
  const [baseVer, setBaseVer] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("split");
  const [split, setSplit] = useState(0.5);
  const [s, setS] = useState(intensity);
  const noFilter = filter === "none";

  const live = useRef({ mode, split, s: 100 });
  live.current = { mode, split, s: noFilter ? 100 : s };

  const draw = useCallback(() => {
    const b = bufRef.current;
    const cv = cvRef.current;
    if (!b || !cv) return;
    const ctx = cv.getContext("2d")!;
    const { mode: m, split: sp } = live.current;
    const wantW = m === "side" ? 2 * b.W : b.W;
    if (cv.width !== wantW) cv.width = wantW; // đổi kích thước canvas sẽ xoá nội dung
    if (cv.height !== b.H) cv.height = b.H;
    if (m === "side") {
      ctx.drawImage(b.io, 0, 0, b.W, b.H);
      ctx.putImageData(b.out, b.W, 0);
    } else if (m === "after") ctx.putImageData(b.out, 0, 0);
    else if (m === "before") ctx.drawImage(b.io, 0, 0, b.W, b.H);
    else {
      ctx.putImageData(b.out, 0, 0);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, Math.round(sp * b.W), b.H);
      ctx.clip();
      ctx.drawImage(b.io, 0, 0, b.W, b.H);
      ctx.restore();
    }
  }, []);

  const render = useCallback(() => {
    const b = bufRef.current;
    if (!b) return;
    const k = live.current.s / 100;
    const D = b.out.data; // Uint8ClampedArray: tự làm tròn và kẹp 0..255
    for (let i = 0, n = b.N; i < n; i++) {
      const p = 4 * i;
      const j = 3 * i;
      D[p] = b.C0[j] + k * b.CD[j];
      D[p + 1] = b.C0[j + 1] + k * b.CD[j + 1];
      D[p + 2] = b.C0[j + 2] + k * b.CD[j + 2];
    }
    draw();
  }, [draw]);

  // Ảnh mới (tách nền lại / phiên mới) -> nạp lại ảnh gốc, tiền cảnh, nền, alpha
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all(["original.jpg", "foreground.png", "background.jpg", "alpha.png"].map((f) => load(sessionUrl(sid, f, { v: ver }))))
      .then(([i0, i1, i2, i3]) => {
        if (cancelled) return;
        const k = Math.min(1, MAX_SIDE / Math.max(i0.width, i0.height));
        const W = Math.max(1, Math.round(i0.width * k));
        const H = Math.max(1, Math.round(i0.height * k));
        const N = W * H;
        const Fp = px(i1, W, H);
        const Bp = px(i2, W, H);
        const Ap = px(i3, W, H);
        const AL = new Float32Array(N);
        const C0 = new Float32Array(3 * N);
        for (let i = 0; i < N; i++) {
          const a = Ap[4 * i] / 255;
          AL[i] = a;
          for (let c = 0; c < 3; c++) C0[3 * i + c] = a * Fp[4 * i + c] + (1 - a) * Bp[4 * i + c];
        }
        const out = new ImageData(W, H);
        for (let i = 0; i < N; i++) out.data[4 * i + 3] = 255;
        bufRef.current = { io: i0, W, H, N, Bp, AL, C0, CD: new Float32Array(3 * N), out };
        setBaseVer((n) => n + 1);
      })
      .catch((e: Error) => !cancelled && (setError(e.message), setLoading(false)));
    return () => {
      cancelled = true;
    };
  }, [sid, ver]);

  // Chỉ đổi hiệu ứng / màu -> chỉ nạp lại nền đã lọc
  useEffect(() => {
    if (baseVer === 0) return;
    let cancelled = false;
    let raf = 0;
    setLoading(true);
    load(sessionUrl(sid, "bgfilter.jpg", { filter, color, v: ver }))
      .then((img) => {
        const b = bufRef.current;
        if (cancelled || !b) return;
        const BF = px(img, b.W, b.H);
        for (let i = 0; i < b.N; i++) {
          const w = 1 - b.AL[i];
          for (let c = 0; c < 3; c++) b.CD[3 * i + c] = w * (BF[4 * i + c] - b.Bp[4 * i + c]);
        }
        render();
        setLoading(false);

        // Lần đầu có kết quả mới: thanh so sánh quét từ phải vào giữa, đúng một lần cho mỗi lượt tách nền
        if (revealed.current === ver) return;
        revealed.current = ver;
        if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
        const t0 = performance.now();
        const tick = (t: number) => {
          const p = Math.min(1, (t - t0) / 700);
          setSplit(1 - 0.5 * (1 - (1 - p) ** 3));
          if (p < 1) raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
      })
      .catch((e: Error) => !cancelled && (setError(e.message), setLoading(false)));
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [baseVer, sid, ver, filter, color, render]);

  useEffect(() => setS(intensity), [intensity]);

  // kéo thanh cường độ: vẽ lại tối đa một lần mỗi khung hình
  useEffect(() => {
    const id = requestAnimationFrame(render);
    return () => cancelAnimationFrame(id);
  }, [s, noFilter, render]);
  useEffect(() => draw(), [mode, split, draw]);

  const dragging = useRef(false);
  const move = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    setSplit(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)));
  };
  const showSplit = mode === "split";

  return (
    <div className="viewer">
      {error && <Alert color="red" mb="xs">{error}</Alert>}
      <div className="stage">
        <div
          className="frame"
          style={{ touchAction: "none", cursor: showSplit ? "ew-resize" : undefined }}
          onPointerDown={(e) => {
            if (!showSplit) return;
            dragging.current = true;
            e.currentTarget.setPointerCapture(e.pointerId);
            move(e);
          }}
          onPointerMove={(e) => dragging.current && move(e)}
          onPointerUp={() => (dragging.current = false)}
        >
          {loading && <div className="loadbar" />}
          <canvas ref={cvRef} />
          {(showSplit || mode === "side") && (
            <>
              <span className="tag" style={{ left: 10 }}>Trước</span>
              <span className="tag" style={{ right: 10 }}>Sau</span>
            </>
          )}
          {showSplit && (
            <>
              <div className="split-line" style={{ left: `${split * 100}%` }} />
              <div className="split-knob" style={{ left: `${split * 100}%` }} />
            </>
          )}
        </div>
      </div>
      <div className="viewer-tools">
        <SegmentedControl size="xs" data={MODES} value={mode} onChange={(v) => setMode(v as Mode)} />
        {!noFilter && (
          <div className="intensity">
            Cường độ
            <Slider flex={1} min={0} max={100} step={5} value={s} label={(v) => `${v}%`} onChange={setS} onChangeEnd={onIntensityCommit} />
            <b>{s}%</b>
          </div>
        )}
      </div>
    </div>
  );
}
