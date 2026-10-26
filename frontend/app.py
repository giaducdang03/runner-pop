"""
Streamlit frontend. Chạy:  streamlit run frontend/app.py

Luồng:
  1. Tải ảnh  -> POST /detect (YOLO) -> hiện box đánh số, chọn người
  2. Bấm Xử lý -> POST /process (GrabCut + trimap + matting + khử nhiễm màu)
  3. Slider cường độ + so sánh trước/sau chạy REALTIME trong trình duyệt
     (canvas JS, không gọi lại backend). Ảnh tải về dùng /compose của backend.
"""
import base64
import os

import requests
import streamlit as st
import streamlit.components.v1 as components

API = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Runner – nền đen trắng", layout="wide")
st.title("Giữ màu người chạy – nền đen trắng")


def api_get(path, **kw):
    r = requests.get(f"{API}{path}", timeout=120, **kw)
    r.raise_for_status()
    return r


def err_msg(r: requests.Response) -> str:
    try:
        return r.json().get("detail", r.text)
    except Exception:
        return r.text


@st.cache_data(ttl=60, show_spinner=False)
def load_options():
    """Engine và bước tiền xử lý lấy từ backend -> thêm bước mới ở backend là giao diện tự có."""
    try:
        return api_get("/options").json()
    except Exception:
        return {"engines": [{"name": "cf", "label": "GrabCut + Closed-form"}], "preprocess": []}


OPTS = load_options()
ENGINE_LABELS = {e["name"]: e["label"] for e in OPTS["engines"]}
PRE_LABELS = {p["name"]: p["label"] for p in OPTS["preprocess"]}
ENGINE_DEFAULT = "birefnet" if "birefnet" in ENGINE_LABELS else next(iter(ENGINE_LABELS))

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Cài đặt")
    auto = st.toggle("Chế độ tự động", value=True,
                     help="Tự chọn người chạy chính và tách nền ngay sau khi tải ảnh; "
                          "tự xử lý lại khi đổi người hoặc cài đặt.")
    pre_mode = st.selectbox("Tiền xử lý để phát hiện người", ["auto", "none", "custom"],
                            format_func={"auto": "Tự động theo chất lượng ảnh", "none": "Không",
                                         "custom": "Tự chọn các bước"}.get,
                            help="Chỉ ảnh dùng cho YOLO được chỉnh; ảnh kết quả luôn giữ màu gốc.")
    if pre_mode == "custom":
        picked = st.multiselect("Các bước (theo thứ tự chọn)", list(PRE_LABELS),
                                format_func=PRE_LABELS.get)
        pre_spec = ",".join(picked) or "none"
    else:
        pre_spec = pre_mode
    method = st.radio("Thuật toán tách nền", list(ENGINE_LABELS),
                      index=list(ENGINE_LABELS).index(ENGINE_DEFAULT),
                      format_func=ENGINE_LABELS.get)
    band = st.slider("Độ rộng vùng viền (unknown)", 0.005, 0.08, 0.02, 0.005,
                     help="Tăng nếu có tóc/vải mỏng bị cắt; giảm nếu viền còn dính nền.")
    focus = st.slider("Ngưỡng độ nét", 0.1, 1.0, 0.3, 0.05, disabled=method != "focus",
                      help="Chỉ cho chế độ 'Mặt phẳng nét'. Tăng nếu giữ màu thừa cả vùng hơi mờ; giảm nếu mất vùng nét.")
    st.caption(f"Backend: {API}")
    try:
        h = api_get("/health").json()
        st.success("Backend OK" + ("" if h["yolo_available"] else " – chưa có ultralytics"))
    except Exception:
        st.error("Không kết nối được backend.\nChạy: uvicorn backend.main:app --port 8000")

# ---------------------------------------------------------------- upload
up = st.file_uploader("Tải ảnh người chạy", type=["jpg", "jpeg", "png", "webp"])
if up is None:
    st.info("Tải ảnh lên để bắt đầu.")
    st.stop()

file_key = f"{up.name}-{up.size}-{pre_spec}"
if st.session_state.get("file_key") != file_key:
    st.session_state.clear()
    st.session_state["file_key"] = file_key
    with st.spinner("YOLO đang phát hiện người..."):
        r = requests.post(f"{API}/detect", params={"preprocess": pre_spec},
                          files={"file": (up.name, up.getvalue(), up.type)}, timeout=300)
    if r.status_code != 200:
        st.error(err_msg(r))
        st.stop()
    st.session_state["det"] = r.json()

det = st.session_state["det"]
sid = det["session_id"]
persons = det["persons"]

q = det.get("quality") or {}
if q:
    flags = ", ".join(q.get("flags", [])) or "tốt"
    steps = " → ".join(det.get("preprocess", [])) or "không"
    retry = f" · phát hiện sau {det['attempt']} lần thử lại" if det.get("attempt") else ""
    st.caption(f"Chất lượng ảnh: {flags} (sáng {q['brightness']:.0f}, tương phản {q['contrast']:.0f}, "
               f"nét {q['sharpness']:.0f}, nhiễu {q['noise']:.1f}) · tiền xử lý: {steps}{retry}")

# ---------------------------------------------------------------- chọn người
c1, c2 = st.columns([3, 2])
with c1:
    st.image(f"{API}/session/{sid}/preview.jpg", caption="Người phát hiện được", use_container_width=True)

manual_box = None
with c2:
    if persons:
        choice = st.selectbox(
            "Chọn người chạy",
            options=[p["idx"] for p in persons],
            index=min(det.get("suggested", 0), len(persons) - 1),
            format_func=lambda i: f"#{i}  (conf {persons[i]['conf']:.2f})"
            + (" – tự chọn" if i == det.get("suggested", 0) else ""),
        )
    else:
        st.warning("YOLO không phát hiện người nào. Nhập box thủ công bên dưới.")
        choice = 0
    with st.expander("Nhập box thủ công (tuỳ chọn)", expanded=not persons):
        W, H = det["width"], det["height"]
        use_manual = st.checkbox("Dùng box thủ công", value=not persons)
        x1, x2 = st.slider("x", 0, W, (int(W * 0.3), int(W * 0.7)))
        y1, y2 = st.slider("y", 0, H, (int(H * 0.1), int(H * 0.9)))
        if use_manual:
            manual_box = f"{x1},{y1},{x2},{y2}"
    go = st.button("Xử lý lại" if auto else "Xử lý", type="primary", use_container_width=True)

# Chế độ tự động: chạy 1 lần cho mỗi tổ hợp (người, engine, tham số) chưa xử lý.
sig = (file_key, choice, manual_box, method, band, focus if method == "focus" else None)
auto_run = auto and (persons or manual_box) and st.session_state.get("auto_sig") != sig
if auto_run:
    st.session_state["auto_sig"] = sig          # đặt trước để lỗi không làm lặp vô hạn

if go or auto_run:
    params = {"person": choice, "method": method, "band": band, "focus": focus}
    if manual_box:
        params["box"] = manual_box
    with st.spinner("Đang tách người và matting (CPU, có thể mất 10–40 giây)..."):
        r = requests.post(f"{API}/process/{sid}", params=params, timeout=900)
    if r.status_code != 200:
        st.error(err_msg(r))
    else:
        info = r.json()
        st.session_state["done"] = info
        st.session_state["assets"] = {
            "orig": base64.b64encode(api_get(f"/session/{sid}/original.jpg").content).decode(),
            "fg": base64.b64encode(api_get(f"/session/{sid}/foreground.png").content).decode(),
            "bg": base64.b64encode(api_get(f"/session/{sid}/background.jpg").content).decode(),
            "alpha": base64.b64encode(api_get(f"/session/{sid}/alpha.png").content).decode(),
        }

if "assets" not in st.session_state:
    st.stop()

# ---------------------------------------------------------------- viewer
info = st.session_state["done"]
a = st.session_state["assets"]
st.success(f"Xong trong {info['elapsed_s']} s.")

VIEWER = """
<style>
  :root { color-scheme: light dark; }
  body { margin:0; font-family: system-ui, sans-serif; }
  #bar { display:flex; gap:18px; align-items:center; flex-wrap:wrap; margin:0 0 8px; font-size:14px; }
  #bar label { display:flex; gap:8px; align-items:center; }
  #stage { text-align:center; }
  /* wrap ôm sát canvas để toạ độ kéo thanh luôn khớp ảnh; ảnh dọc bị giới hạn chiều cao để không tràn iframe */
  #wrap { position:relative; display:inline-block; max-width:100%; user-select:none; touch-action:none; }
  canvas { display:block; width:auto; height:auto; max-width:100%; max-height:740px; border-radius:6px; }
  #tag { position:absolute; top:8px; padding:2px 8px; border-radius:4px; font-size:12px;
         background:rgba(0,0,0,.55); color:#fff; pointer-events:none; }
  #line { position:absolute; top:0; bottom:0; width:2px; background:#fff; box-shadow:0 0 4px #0008; pointer-events:none; }
  #knob { position:absolute; top:50%; width:28px; height:28px; margin:-14px 0 0 -13px; border-radius:50%;
          background:#fff; box-shadow:0 0 6px #0008; pointer-events:none; }
</style>
<div id="bar">
  <label>Cường độ nền đen trắng
    <input id="s" type="range" min="0" max="100" value="100" style="width:200px"><b id="sv">100%</b></label>
  <label>Chế độ
    <select id="mode">
      <option value="split">So sánh (kéo thanh)</option>
      <option value="side">Cạnh nhau (trái: trước, phải: sau)</option>
      <option value="after">Chỉ kết quả</option>
      <option value="before">Chỉ ảnh gốc</option>
    </select></label>
</div>
<div id="stage"><div id="wrap">
  <canvas id="cv"></canvas>
  <div id="tag" style="left:8px">Trước</div>
  <div id="tag2" class="tag" style="position:absolute;top:8px;right:8px;padding:2px 8px;border-radius:4px;font-size:12px;background:rgba(0,0,0,.55);color:#fff;pointer-events:none">Sau</div>
  <div id="line"></div><div id="knob"></div>
</div></div>
<script>
const load = src => new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = src; });
const px = i => { const c = document.createElement('canvas'); c.width = i.width; c.height = i.height;
  const x = c.getContext('2d', {willReadFrequently:true}); x.drawImage(i,0,0); return x.getImageData(0,0,i.width,i.height); };

(async () => {
  const [io, ifg, ibg, ia] = await Promise.all([
    load("data:image/jpeg;base64,__ORIG__"), load("data:image/png;base64,__FG__"),
    load("data:image/jpeg;base64,__BG__"), load("data:image/png;base64,__ALPHA__")]);
  const W = io.width, H = io.height, N = W*H;
  const F = px(ifg).data, B = px(ibg).data, A = px(ia).data;
  const gray = new Float32Array(N);
  for (let i=0;i<N;i++) gray[i] = 0.2126*B[4*i] + 0.7152*B[4*i+1] + 0.0722*B[4*i+2];

  const cv = document.getElementById('cv'); cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  const out = ctx.createImageData(W, H), D = out.data;
  const sl = document.getElementById('s'), sv = document.getElementById('sv'), mode = document.getElementById('mode');
  const line = document.getElementById('line'), knob = document.getElementById('knob'), wrap = document.getElementById('wrap');
  const tag = document.getElementById('tag'), tag2 = document.getElementById('tag2');
  let split = 0.5, s = 1.0;

  function render() {
    // cùng công thức với backend: out = a*F + (1-a)*((1-s)*B + s*gray(B))
    for (let i=0;i<N;i++) {
      const a = A[4*i]/255, g = gray[i], k = 4*i;
      D[k]   = a*F[k]   + (1-a)*((1-s)*B[k]   + s*g);
      D[k+1] = a*F[k+1] + (1-a)*((1-s)*B[k+1] + s*g);
      D[k+2] = a*F[k+2] + (1-a)*((1-s)*B[k+2] + s*g);
      D[k+3] = 255;
    }
    draw();
  }
  function draw() {
    const m = mode.value;
    const wantW = m === 'side' ? 2*W : W;
    if (cv.width !== wantW) cv.width = wantW;          // đổi kích thước canvas sẽ xoá nội dung
    if (m === 'side') { ctx.drawImage(io, 0, 0); ctx.putImageData(out, W, 0); }
    else if (m === 'after') ctx.putImageData(out, 0, 0);
    else if (m === 'before') ctx.drawImage(io, 0, 0);
    else {
      ctx.putImageData(out, 0, 0);
      const x = Math.round(split*W);
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, x, H); ctx.clip(); ctx.drawImage(io, 0, 0); ctx.restore();
    }
    const showSplit = m === 'split';
    line.style.display = knob.style.display = showSplit ? '' : 'none';
    tag.style.display = tag2.style.display = (showSplit || m === 'side') ? '' : 'none';
    line.style.left = knob.style.left = (split*100) + '%';
  }
  sl.addEventListener('input', () => { s = sl.value/100; sv.textContent = sl.value + '%'; render(); });
  mode.addEventListener('change', draw);
  const move = e => { const r = wrap.getBoundingClientRect(); split = Math.min(1, Math.max(0, (e.clientX - r.left)/r.width)); draw(); };
  let drag = false;
  wrap.addEventListener('pointerdown', e => { drag = true; wrap.setPointerCapture(e.pointerId); move(e); });
  wrap.addEventListener('pointermove', e => { if (drag) move(e); });
  wrap.addEventListener('pointerup', () => drag = false);
  render();
})();
</script>
"""
html = (VIEWER.replace("__ORIG__", a["orig"]).replace("__FG__", a["fg"])
        .replace("__BG__", a["bg"]).replace("__ALPHA__", a["alpha"]))
components.html(html, height=780, scrolling=False)

# ---------------------------------------------------------------- tải về + debug
d1, d2 = st.columns(2)
with d1:
    dl_int = st.slider("Cường độ khi tải về", 0, 100, 100, 5, format="%d%%") / 100
    fmt = st.radio("Định dạng", ["png", "jpeg"], horizontal=True)
    img = api_get(f"/session/{sid}/compose", params={"intensity": dl_int, "fmt": fmt}).content
    st.download_button("Tải ảnh kết quả", img, file_name=f"result.{'jpg' if fmt == 'jpeg' else 'png'}",
                       mime=f"image/{fmt}", use_container_width=True)
with d2:
    with st.expander("Debug: alpha & trimap"):
        st.image(f"{API}/session/{sid}/alpha.png", caption="Alpha matte", use_container_width=True)
        st.image(f"{API}/session/{sid}/trimap.png", caption="Trimap (xám = vùng unknown)", use_container_width=True)
