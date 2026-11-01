"""
Streamlit frontend. Chạy:  streamlit run frontend/app.py

Luồng:
  1. Tải ảnh  -> POST /detect (YOLO) -> hiện box đánh số, chọn người
  2. Bấm Xử lý -> POST /process (GrabCut + trimap + matting + khử nhiễm màu)
  3. Slider cường độ + so sánh trước/sau chạy REALTIME trong trình duyệt
     (canvas JS, không gọi lại backend). Ảnh tải về dùng /compose của backend.
"""
import base64
import json
import os
import threading
import time
from html import escape

import requests
import streamlit as st
import streamlit.components.v1 as components

from ui_helpers import (device_line, is_warming, pick_default_engine, prepare_message,
                        prepare_note, should_auto_run, should_poll, sidebar_prepare_plan, fragment_action)

API = os.environ.get("API_URL", "http://127.0.0.1:8000")

APP_TITLE = "RunnerPop – Image Preprocessing Optimization for AI-based Runner Segmentation"
st.set_page_config(page_title=APP_TITLE, page_icon="🏃", layout="wide")
st.title(APP_TITLE)
st.caption("Giữ màu người chạy, đưa nền về đen trắng · YOLO-seg → tách nền → matting → ghép (chạy trên CPU, hoặc GPU Intel / NVIDIA / Apple khi có)")


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
ENGINE_DEFAULT = pick_default_engine(OPTS, list(ENGINE_LABELS))
DEVICE_CHOICES = OPTS.get("devices") or ["auto", "cpu", "gpu"]

PIPELINE_TMPL = """
<style>
  :root { color-scheme: light dark; }
  body { margin:0; font-family: system-ui, sans-serif; }
  #row { display:flex; gap:8px; overflow-x:scroll; padding:4px 2px 12px; }
  .card { flex:0 0 220px; border:1px solid rgba(128,128,128,.35); border-radius:10px; padding:8px; box-sizing:border-box; }
  .card .n { font-size:11px; opacity:.65; }
  .card .t { font-weight:600; font-size:13px; margin:2px 0 6px; min-height:34px; }
  /* ảnh dọc (ROI người) không được làm thẻ cao hơn iframe 440px, nếu không thanh cuộn ngang bị cắt mất */
  .card img { width:100%; max-height:240px; object-fit:contain; border-radius:6px; display:block; cursor:zoom-in; transition:transform .15s; }
  .card img:hover { transform:scale(1.02); }
  .card .none { height:70px; border-radius:6px; background:rgba(128,128,128,.12); display:flex;
                align-items:center; justify-content:center; font-size:11px; opacity:.6; }
  .card .d { font-size:11px; opacity:.75; margin-top:6px; line-height:1.3; }
  .arrow { flex:0 0 auto; align-self:center; font-size:22px; opacity:.5; }
</style>
<div id="row"></div>
<script>
const stages = __DATA__;
const row = document.getElementById('row');
// Modal gắn vào trang cha (iframe của Streamlit cùng origin) để phủ toàn màn hình; không được thì dùng iframe.
let P = window;
try { void window.parent.document.body; P = window.parent; } catch (e) {}
const D = P.document;
let idx = 0;

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

stages.forEach((s, i) => {
  if (i) { const a = document.createElement('div'); a.className = 'arrow'; a.textContent = '➜'; row.appendChild(a); }
  const c = document.createElement('div'); c.className = 'card';
  c.innerHTML = `<div class="n">BƯỚC ${i+1} · ${s.seconds.toFixed(2)}s</div><div class="t">${esc(s.label)}</div>`
    + (s.image ? `<img src="${s.image}" title="Bấm để phóng to">` : `<div class="none">không có ảnh</div>`)
    + `<div class="d">${s.note ? esc(s.note) : '&nbsp;'}</div>`;
  const img = c.querySelector('img'); if (img) img.addEventListener('click', () => openModal(i));
  row.appendChild(c);
});

function closeModal() {
  const o = D.getElementById('pl-modal'); if (o) o.remove();
  if (P.__plKey) { D.removeEventListener('keydown', P.__plKey); P.__plKey = null; }
}
function step(d) {
  let j = idx; do { j = (j + d + stages.length) % stages.length; } while (!stages[j].image && j !== idx);
  openModal(j);
}
function openModal(i) {
  closeModal(); idx = i;
  const s = stages[i];
  const o = D.createElement('div'); o.id = 'pl-modal';
  o.style.cssText = 'position:fixed;inset:0;z-index:999999;background:rgba(0,0,0,.88);display:flex;flex-direction:column;'
    + 'align-items:center;justify-content:center;gap:12px;font-family:system-ui,sans-serif;color:#fff;cursor:zoom-out';
  o.innerHTML = `<div style="font-size:14px;opacity:.8">BƯỚC ${i+1}/${stages.length} · ${s.seconds.toFixed(2)}s</div>`
    + `<div style="font-size:18px;font-weight:600;text-align:center;padding:0 60px">${esc(s.label)}</div>`
    + `<img src="${s.image}" style="max-width:92vw;max-height:76vh;border-radius:8px;box-shadow:0 8px 40px #000a;background:#222">`
    + `<div style="font-size:13px;opacity:.75;max-width:80vw;text-align:center">${esc(s.note || '')}</div>`
    + `<div style="font-size:12px;opacity:.55">← → chuyển bước · Esc hoặc bấm để đóng</div>`;
  const nav = (txt, side, d) => { const b = D.createElement('div'); b.textContent = txt;
    b.style.cssText = `position:absolute;top:50%;${side}:18px;font-size:38px;padding:8px 14px;cursor:pointer;user-select:none;opacity:.7`;
    b.addEventListener('click', e => { e.stopPropagation(); step(d); }); return b; };
  o.appendChild(nav('‹', 'left', -1)); o.appendChild(nav('›', 'right', 1));
  o.addEventListener('click', closeModal);
  D.body.appendChild(o);
  P.__plKey = e => { if (e.key === 'Escape') closeModal(); else if (e.key === 'ArrowRight') step(1); else if (e.key === 'ArrowLeft') step(-1); };
  D.addEventListener('keydown', P.__plKey);
}
</script>
"""


def render_pipeline(title: str, stages: list) -> None:
    """Vẽ các bước đã chạy thành sơ đồ pipeline nằm ngang; bấm vào ảnh để phóng to (modal, ←/→ đổi bước)."""
    if not stages:
        return
    total = sum(s["seconds"] for s in stages)
    st.markdown(f"**{title}** · {len(stages)} bước · {total:.1f}s")
    data = json.dumps(stages).replace("</", "<\\/")
    components.html(PIPELINE_TMPL.replace("__DATA__", data), height=440, scrolling=False)


def live_cards_html(title: str, stages: list, running: str = None, state: str = "running") -> str:
    """Sơ đồ tĩnh (không JS) cho lúc đang chạy: các bước đã xong + thẻ 'đang chạy'. Xong rồi mới chuyển sang bản zoom được."""
    card = ("flex:0 0 200px;border:1px solid rgba(128,128,128,.35);border-radius:10px;"
            "padding:8px;box-sizing:border-box")
    cards = []
    for i, s in enumerate(stages, 1):
        img = (f'<img src="{s["image"]}" style="width:100%;border-radius:6px;display:block"/>' if s["image"]
               else '<div style="height:60px;border-radius:6px;background:rgba(128,128,128,.12)"></div>')
        cards.append(
            f'<div style="{card}"><div style="font-size:11px;opacity:.65">BƯỚC {i} · {s["seconds"]:.2f}s ✓</div>'
            f'<div style="font-weight:600;font-size:13px;margin:2px 0 6px;min-height:34px">{escape(s["label"])}</div>'
            f'{img}<div style="font-size:11px;opacity:.75;margin-top:6px">{escape(s["note"]) or "&nbsp;"}</div></div>')
    if running and state == "running":
        cards.append(
            f'<div style="{card};border:2px dashed #f5a623">'
            f'<div style="font-size:11px;color:#f5a623">BƯỚC {len(stages) + 1} · ĐANG CHẠY…</div>'
            f'<div style="font-weight:600;font-size:13px;margin:2px 0 6px;min-height:34px">{escape(running)}</div>'
            '<div style="height:110px;border-radius:6px;background:rgba(245,166,35,.12);display:flex;'
            'align-items:center;justify-content:center;font-size:30px">⏳</div></div>')
    arrow = '<div style="flex:0 0 auto;align-self:center;font-size:22px;opacity:.5">➜</div>'
    head = f"<b>{escape(title)}</b> · {len(stages)} bước · {sum(s['seconds'] for s in stages):.1f}s"
    return (f'<div style="margin-bottom:4px">{head}</div>'
            '<div style="display:flex;gap:8px;overflow-x:auto;padding:4px 2px 12px">' + arrow.join(cards) + "</div>")


def follow_job(slot, sid: str, phase: str, title: str):
    """Theo dõi job nền, vẽ pipeline realtime trong `slot`. Trả về (job_status, danh_sách_stage)."""
    stages: list = []
    with slot:
        ph = st.empty()
    while True:
        r = api_get(f"/session/{sid}/job/{phase}", params={"since": len(stages)}).json()
        stages += r["stages"]
        ph.html(live_cards_html(title, stages, r["running"], r["state"]))
        if r["state"] != "running":
            break
        time.sleep(0.25)
    ph.empty()
    return r, stages


@st.fragment(run_every=2)
def warm_status():
    """Hiện trạng thái warm-up và hỏi lại /health mỗi 2 s; hết "running" thì chạy lại cả script để auto-run tiếp tục."""
    try:
        state = api_get("/health").json().get("warmup")
    except Exception:
        state = None
    if should_poll(state):
        st.info("Đang chuẩn bị model…")
    else:
        st.rerun()


def prepare_state(engine: str, device_mode: str, post: bool = False) -> dict:
    """POST (idempotent) hoặc GET /prepare. Lỗi mạng / thiếu endpoint -> {} (coi như không chặn)."""
    try:
        fn = requests.post if post else requests.get
        r = fn(f"{API}/prepare", params={"engine": engine, "device": device_mode}, timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


@st.fragment(run_every=2)
def prepare_status(engine: str, device_mode: str, label: str):
    """Hiện thông báo đang nạp model và hỏi lại GET /prepare mỗi 2 s; hết "running" thì chạy lại cả script."""
    state = prepare_state(engine, device_mode).get("state")
    act = fragment_action(state)
    if act == "post":                       # backend khởi động lại giữa chừng: yêu cầu nạp lại đúng một lần
        prepare_state(engine, device_mode, post=True)
    if act == "rerun":
        st.rerun()
    else:
        st.info(prepare_message(label, device_mode))


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
    device = st.selectbox("Thiết bị", DEVICE_CHOICES,
                          help="auto: dùng GPU nếu có, không thì CPU. Gửi kèm mọi yêu cầu phát hiện / tách nền.")
    prep = prepare_state(method, device)                # GET trước; chỉ POST khi chưa từng yêu cầu (idle)
    plan = sidebar_prepare_plan(prep.get("state"))
    if plan["post"]:
        prep = prepare_state(method, device, post=True) or prep
        plan = sidebar_prepare_plan(prep.get("state"))
    ready = plan["ready"]
    if plan["polling"]:
        prepare_status(method, device, ENGINE_LABELS[method])
    elif plan["show_error"]:
        st.warning(f"Nạp model gặp lỗi: {prep.get('detail', '')}. Vẫn có thể thử xử lý.")
        if st.button("Thử nạp lại"):                    # chỉ nút này mới thử lại cặp đang lỗi
            prepare_state(method, device, post=True)
            st.rerun()
    elif prep.get("state") == "done" and prepare_note(device, prep.get("used_device", "")):
        st.caption(prepare_note(device, prep.get("used_device", "")))
    st.caption(f"Backend: {API}")
    warming = False
    try:
        h = api_get("/health").json()
        st.success("Backend OK" + ("" if h["yolo_available"] else " – chưa có ultralytics"))
        if h.get("device"):
            st.caption(device_line(h["device"]))
            avail = h.get("available_backends") or {}
            st.caption("Backend khả dụng: " + (", ".join(f"{k} ({v})" for k, v in avail.items()) or "chỉ CPU"))
            st.caption("Đổi backend cần khởi động lại backend với biến môi trường ACCEL_BACKEND.")
        warming = is_warming(h.get("warmup"))
        if should_poll(h.get("warmup")):
            warm_status()
        elif h.get("warmup") == "error":
            st.warning(f"Chuẩn bị model lỗi, lần chạy đầu sẽ chậm: {h.get('warmup_detail', '')}")
    except Exception:
        st.error("Không kết nối được backend.\nChạy: uvicorn backend.main:app --port 8000")

# ---------------------------------------------------------------- upload
up = st.file_uploader("Tải ảnh người chạy", type=["jpg", "jpeg", "png", "webp"], disabled=not ready)
if up is None:
    st.info("Tải ảnh lên để bắt đầu.")
    st.stop()

file_key = f"{up.name}-{up.size}-{pre_spec}"
if st.session_state.get("file_key") != file_key:
    st.session_state.clear()
    st.session_state["file_key"] = file_key
    r = requests.post(f"{API}/detect", params={"preprocess": pre_spec, "device": device, "background": "true"},
                      files={"file": (up.name, up.getvalue(), up.type)}, timeout=300)
    if r.status_code != 200:
        st.error(err_msg(r))
        st.stop()
    job, stages = follow_job(st.container(), r.json()["session_id"], "detect",
                             "Pipeline 1 · Tiền xử lý & phát hiện người (đang chạy)")
    if job["state"] == "error":
        st.error(job["error"])
        st.stop()
    st.session_state["det"] = job["result"]
    st.session_state["trace"] = {"detect": stages, "segment": []}

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

render_pipeline("Pipeline 1 · Tiền xử lý & phát hiện người", st.session_state["trace"]["detect"])

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
    go = st.button("Xử lý lại" if auto else "Xử lý", type="primary", use_container_width=True,
                   disabled=not ready)

# Chế độ tự động: chạy 1 lần cho mỗi tổ hợp (người, engine, tham số) chưa xử lý.
sig = (file_key, choice, manual_box, method, band, focus if method == "focus" else None, device)
auto_run = should_auto_run(auto, bool(persons or manual_box), sig, st.session_state.get("auto_sig"), warming, ready)
if auto_run:
    st.session_state["auto_sig"] = sig          # đặt trước để lỗi không làm lặp vô hạn

if go or auto_run:
    params = {"person": choice, "method": method, "band": band, "focus": focus, "device": device}
    if manual_box:
        params["box"] = manual_box
    r = requests.post(f"{API}/process/{sid}", params={**params, "background": "true"}, timeout=60)
    job = None
    if r.status_code != 200:
        st.error(err_msg(r))
    else:
        job, stages = follow_job(st.container(), sid, "segment",
                                 f"Pipeline 2 · Tách nền ({ENGINE_LABELS.get(method, method)}) (đang chạy)")
        if job["state"] == "error":
            st.error(job["error"])
            job = None
    if job is not None:
        info = job["result"]
        st.session_state["done"] = info
        st.session_state["seg_ver"] = st.session_state.get("seg_ver", 0) + 1     # làm mới cache ảnh xem trước / nền đã lọc
        st.session_state["trace"]["segment"] = stages
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
render_pipeline(f"Pipeline 2 · Tách nền ({ENGINE_LABELS.get(info['method'], info['method'])})",
                st.session_state["trace"]["segment"])
st.success(f"Xong trong {info['elapsed_s']} s.")
_det_bib = info.get("bib", "")
if st.session_state.get("bib_src") != (sid, _det_bib):       # mỗi lần xử lý xong -> nạp lại giá trị đọc được
    st.session_state["bib_src"] = (sid, _det_bib)
    st.session_state["bib_value"] = _det_bib
st.text_input("Số BIB" + ("" if _det_bib else " (không đọc được tự động — nhập tay nếu cần)"), key="bib_value")
_bib = "".join(ch for ch in st.session_state.get("bib_value", "") if ch.isalnum())
_bib_prefix = f"BIB_{_bib}" if _bib else ""
_fc = info.get("focus_confidence")
if _fc is not None and _fc < OPTS.get("focus_conf_min", 1.3):
    st.warning("Ảnh này không có phông mờ rõ — kết quả 'mặt phẳng nét' có thể kém tin cậy. Thử engine khác.")

# ---------------------------------------------------------------- trạng thái bộ lọc nền (đọc trước viewer, lưới đặt dưới viewer)
BG_FILTERS = OPTS.get("bg_filters") or [{"name": "bw", "label": "Nền đen trắng"}]
if st.session_state.get("bg_filter") not in {f["name"] for f in BG_FILTERS}:
    st.session_state["bg_filter"] = "bw"
bg_filter = st.session_state["bg_filter"]
bg_color = st.session_state.get("bg_color", "#808080")
bg_int = 1.0 if bg_filter == "none" else st.session_state.get("bg_intensity", 100) / 100
_ver = st.session_state.get("seg_ver", 0)

_left, _right = st.columns([5, 3], gap="medium")      # trái: viewer, phải: lưới hiệu ứng
if st.session_state.get("bgf_key") != (sid, bg_filter, bg_color, _ver):
    with _left:
        _ld = st.progress(0.15, text="Đang áp dụng hiệu ứng…")      # thanh chờ khi lấy nền đã lọc từ backend
        st.session_state["bgf_img"] = base64.b64encode(api_get(
            f"/session/{sid}/bgfilter.jpg", params={"filter": bg_filter, "color": bg_color}).content).decode()
        st.session_state["bgf_key"] = (sid, bg_filter, bg_color, _ver)
        _ld.progress(1.0, text="Đang ghép ảnh…")
        _ld.empty()

VIEWER = """
<style>
  :root { color-scheme: light dark; }
  body { margin:0; font-family: system-ui, sans-serif; }
  #bar { display:flex; gap:18px; align-items:center; flex-wrap:wrap; margin:0 0 8px; font-size:14px; }
  #bar label { display:flex; gap:8px; align-items:center; }
  #ld { height:4px; border-radius:2px; background:rgba(128,128,128,.25); overflow:hidden; margin:0 0 8px; }
  #ld div { height:100%; width:35%; border-radius:2px; background:#f5a623; animation:ld 1s ease-in-out infinite; }
  @keyframes ld { 0% { margin-left:-35%; } 100% { margin-left:100%; } }
  #stage { text-align:center; }
  /* wrap ôm sát canvas để toạ độ kéo thanh luôn khớp ảnh; ảnh dọc bị giới hạn chiều cao để không tràn iframe */
  #wrap { position:relative; display:inline-block; max-width:100%; user-select:none; touch-action:none; }
  canvas { display:block; width:auto; height:auto; max-width:100%; max-height:540px; border-radius:6px; }
  #tag { position:absolute; top:8px; padding:2px 8px; border-radius:4px; font-size:12px;
         background:rgba(0,0,0,.55); color:#fff; pointer-events:none; }
  #line { position:absolute; top:0; bottom:0; width:2px; background:#fff; box-shadow:0 0 4px #0008; pointer-events:none; }
  #knob { position:absolute; top:50%; width:28px; height:28px; margin:-14px 0 0 -13px; border-radius:50%;
          background:#fff; box-shadow:0 0 6px #0008; pointer-events:none; }
</style>
<div id="ld"><div></div></div>
<div id="bar">
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
  const [io, ifg, ibg, ia, ibf] = await Promise.all([
    load("data:image/jpeg;base64,__ORIG__"), load("data:image/png;base64,__FG__"),
    load("data:image/jpeg;base64,__BG__"), load("data:image/png;base64,__ALPHA__"),
    load("data:image/jpeg;base64,__BF__")]);
  const W = io.width, H = io.height, N = W*H;
  const F = px(ifg).data, B = px(ibg).data, A = px(ia).data, BF = px(ibf).data;   // BF: nền đã lọc (bộ lọc chọn ở lưới)

  const cv = document.getElementById('cv'); cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  const out = ctx.createImageData(W, H), D = out.data;
  const mode = document.getElementById('mode');
  const line = document.getElementById('line'), knob = document.getElementById('knob'), wrap = document.getElementById('wrap');
  const tag = document.getElementById('tag'), tag2 = document.getElementById('tag2');
  let split = 0.5, s = __S__;

  function render() {
    // cùng công thức với backend: out = a*F + (1-a)*((1-s)*B + s*Bf)
    for (let i=0;i<N;i++) {
      const a = A[4*i]/255, k = 4*i;
      D[k]   = a*F[k]   + (1-a)*((1-s)*B[k]   + s*BF[k]);
      D[k+1] = a*F[k+1] + (1-a)*((1-s)*B[k+1] + s*BF[k+1]);
      D[k+2] = a*F[k+2] + (1-a)*((1-s)*B[k+2] + s*BF[k+2]);
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
  mode.addEventListener('change', draw);
  const move = e => { const r = wrap.getBoundingClientRect(); split = Math.min(1, Math.max(0, (e.clientX - r.left)/r.width)); draw(); };
  let drag = false;
  wrap.addEventListener('pointerdown', e => { drag = true; wrap.setPointerCapture(e.pointerId); move(e); });
  wrap.addEventListener('pointermove', e => { if (drag) move(e); });
  wrap.addEventListener('pointerup', () => drag = false);
  render();
  document.getElementById('ld').style.display = 'none';
})();
</script>
"""
html = (VIEWER.replace("__ORIG__", a["orig"]).replace("__FG__", a["fg"])
        .replace("__BG__", a["bg"]).replace("__ALPHA__", a["alpha"])
        .replace("__BF__", st.session_state["bgf_img"]).replace("__S__", f"{bg_int:.3f}"))
with _left:
    components.html(html, height=620, scrolling=False)

# ---------------------------------------------------------------- chỉnh màu nền (cột phải, cuộn trong khung cố định)
st.markdown(
    "<style>[class*='st-key-bgf_'] button{padding:.05rem .2rem;min-height:1.6rem;font-size:.74rem}"
    "[class*='st-key-bgf_'] button p{font-size:.74rem}</style>", unsafe_allow_html=True)
with _right:
    st.markdown("**Chỉnh màu nền**")
    if st.session_state.get("bgf_prev_key") != (sid, bg_color, _ver):
        st.session_state["bgf_prev"] = {}
        st.session_state["bgf_prev_key"] = (sid, bg_color, _ver)
    _prev = st.session_state["bgf_prev"]
    _todo = [f["name"] for f in BG_FILTERS if f["name"] not in _prev]
    if _todo:                                             # tải từng ảnh xem trước một, có thanh tiến độ
        _bar = st.progress(0.0, text="Đang tải ảnh xem trước…")
        for _k, _n in enumerate(_todo, 1):
            try:
                _pv = api_get(f"/session/{sid}/bgfilter_previews", params={"color": bg_color, "names": _n}).json()
                _it = next((p for p in _pv if p["name"] == _n), None)
                _prev[_n] = base64.b64decode(_it["image"].split(",", 1)[1]) if _it else b""
            except Exception:
                _prev[_n] = b""                           # lỗi -> ô chỉ còn nút, không có ảnh (không thử lại mãi)
            _bar.progress(_k / len(_todo), text=f"Đang tải ảnh xem trước… {_k}/{len(_todo)}")
        _bar.empty()
    _N = 3
    with st.container(height=470, border=False):          # khung cuộn riêng: trang không dài thêm khi có nhiều hiệu ứng
        for _row in range(0, len(BG_FILTERS), _N):
            _cols = st.columns(_N, gap="small")
            for _col, _f in zip(_cols, BG_FILTERS[_row:_row + _N]):
                with _col:
                    if _prev.get(_f["name"]):
                        st.image(_prev[_f["name"]], use_container_width=True)
                    if st.button(_f["label"], key=f"bgf_{_f['name']}", use_container_width=True,
                                 type="primary" if _f["name"] == bg_filter else "secondary"):
                        st.session_state["bg_filter"] = _f["name"]
                        st.rerun()
    if bg_filter != "none":
        st.slider("Cường độ hiệu ứng", 0, 100, 100, 5, format="%d%%", key="bg_intensity")
    if bg_filter == "solid":
        st.color_picker("Màu nền", bg_color, key="bg_color")

# ---------------------------------------------------------------- tải về + debug
d1, d2 = st.columns(2)
with d1:
    dl_int = bg_int                                   # cùng một thanh cường độ với viewer
    dl_args = {"intensity": dl_int, "filter": bg_filter, "color": bg_color}
    fmt = st.radio("Định dạng", ["png", "jpeg"], horizontal=True)
    img = api_get(f"/session/{sid}/compose", params={**dl_args, "fmt": fmt}).content
    st.download_button("Tải ảnh kết quả", img, file_name=f"{_bib_prefix or 'result'}.{'jpg' if fmt == 'jpeg' else 'png'}",
                       mime=f"image/{fmt}", use_container_width=True)
with d2:
    st.markdown("**Báo cáo PDF**")
    if st.button("Tạo báo cáo PDF (toàn bộ thông tin + pipeline)", use_container_width=True):
        # backend không báo tiến độ -> chạy request ở thread phụ, thanh chạy theo thời gian (tiệm cận 95%) tới khi xong
        _box = {}

        def _fetch_pdf():
            try:
                _box["r"] = requests.get(f"{API}/session/{sid}/report.pdf", params=dl_args, timeout=120)
            except requests.RequestException as e:
                _box["err"] = str(e)

        _th = threading.Thread(target=_fetch_pdf, daemon=True)
        _bar = st.progress(0.0, text="Đang tạo báo cáo PDF…")
        _t0 = time.time()
        _th.start()
        while _th.is_alive():
            _el = time.time() - _t0
            _bar.progress(min(0.95, 1 - 0.5 ** (_el / 3)), text=f"Đang tạo báo cáo PDF… {_el:.0f}s")
            time.sleep(0.2)
        _bar.progress(1.0, text="Xong")
        _bar.empty()
        rr = _box.get("r")
        if rr is None:
            st.error(f"Không tạo được PDF: {_box.get('err', 'không rõ lỗi')}")
        elif rr.status_code == 200:
            st.session_state["pdf"] = rr.content
            st.session_state["pdf_key"] = (sid, _ver, bg_filter, bg_color, bg_int)     # PDF chỉ đúng với hiệu ứng lúc tạo
        else:
            st.error(err_msg(rr))
    if st.session_state.get("pdf") and st.session_state.get("pdf_key") != (sid, _ver, bg_filter, bg_color, bg_int):
        st.caption("Hiệu ứng đã đổi — bấm tạo lại báo cáo để PDF khớp.")
    elif st.session_state.get("pdf"):
        st.download_button("Tải báo cáo PDF", st.session_state["pdf"], file_name=f"{_bib_prefix}_report.pdf" if _bib_prefix else "color_pop_runner_report.pdf",
                           mime="application/pdf", use_container_width=True)
    with st.expander("Debug: alpha & trimap"):
        st.image(f"{API}/session/{sid}/alpha.png", caption="Alpha matte", use_container_width=True)
        st.image(f"{API}/session/{sid}/trimap.png", caption="Trimap (xám = vùng unknown)", use_container_width=True)
