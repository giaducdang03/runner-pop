# Giữ đồ vật cầm tay (gậy trekking, chai, điện thoại…) trong mask người – Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Khi tách runner, đồ vật người đó đang cầm (trước hết là **gậy trekking**, sau đó là chai, điện thoại và vật COCO khác) được giữ trọn trong mask và alpha, không bị cắt ngang sát bàn tay.

**Architecture:** Sau YOLO-seg, chạy thêm YOLO-pose để lấy **tâm nắm tay** của từng người. Từ mỗi nắm tay, **quét tia hướng xuống** trên ảnh màu, kiểm tra mặt cắt ngang dạng "thanh" để tìm gậy. Gậy tìm được, cùng các vật COCO nằm gần nắm tay, được gộp vào `Person.mask` và `Person.box` ngay ở bước detect, nên mọi engine tự giữ chúng. Gậy mảnh sẽ mất khi thu về độ phân giải làm việc 640 px, nên alpha của gậy được tính lại ở **độ phân giải gốc** trong một dải hẹp và đè lên alpha của engine.

**Tech Stack:** Python 3.13, ultralytics 8.4 (YOLOv8-seg + YOLOv8-pose), OpenCV, NumPy, pymatting, pytest.

**Spec:** Không có file spec. Yêu cầu lấy từ trao đổi ngày 2026-11-08: "tách được những đồ vật mà người đó đang cầm nắm", bài toán ảnh giải chạy, ảnh test `photo_5394198.jpg` (Mù Cang Chải Ultra Trail 2026).

**Nguyên nhân gốc (đã kiểm chứng bằng code và probe):**
- [detect.py:111](../../../backend/pipeline/detect.py) chỉ predict `classes=[PERSON_CLASS]`. Trên ảnh test, YOLO COCO không phát hiện được cây gậy nào ở mọi lớp (chỉ ra `sports ball` và `handbag` sai).
- [gate.py:22-33](../../../backend/pipeline/segment/gate.py), [grabcut.py:39-49](../../../backend/pipeline/segment/grabcut.py), [focus.py:411-415](../../../backend/pipeline/segment/focus.py) đều chặn kết quả trong khoảng 1–6% box quanh mask YOLO.
- `WORK_MAX_SIDE = 640`: runner cao khoảng 1700 px (ảnh 2560) chạy ở scale khoảng 0.32. Gậy rộng 10–25 px chỉ còn 3–8 px, và gậy mảnh của runner áo đỏ gần như biến mất.

**Kết quả probe trên ảnh test** (ảnh resize về cạnh dài 2560, script ở scratchpad, không nằm trong repo):

| Phương pháp | Kết quả |
|---|---|
| LSD, đầu đoạn sát cổ tay | Thất bại: cổ tay (keypoint) cách gậy 80–100 px; LSD bỏ phần gậy đen trên nền tối và gậy mảnh bị nhoè |
| Quét tia + ridge Hessian, mọi góc | Thất bại một nửa: tia bám viền tay áo, viền quần |
| **Quét tia + mặt cắt dạng thanh + góc 45–135°** | Runner chính: tay phải 87.5° / 0.92 ✅, tay trái 79° / 0.71 ✅. Runner áo đỏ: 75° / 0.59 ✅. Người không cầm gậy: 0.15 ✅. Người áo trắng: nhận nhầm gậy của runner chính (79° / 0.74) ❌ |

## Global Constraints

- Comment/docstring trong code viết tiếng Việt, theo phong cách hiện có.
- Mọi ngưỡng mới đặt trong `backend/pipeline/config.py`, nhóm `# --- đồ vật cầm trên tay ---`, với đúng các giá trị nêu trong từng task.
- Ngưỡng hình học tính theo **chiều cao người `h`** (box thân, ảnh làm việc), không dùng px tuyệt đối, để chạy được với mọi `MAX_SIDE`.
- Người không cầm gì, hoặc khi `HELD_ENABLE=0`, phải cho kết quả **giống hệt** trước khi sửa. Toàn bộ test hiện có phải vẫn pass, trừ assertion `classes` được sửa có chủ đích ở Task 5.
- Pose hoặc dò gậy lỗi (thiếu weights, exception) thì ghi chú vào trace và chạy tiếp như cũ; tuyệt đối không làm hỏng request.
- Thư mục không phải git repo nên bỏ các bước commit. Nếu sau này `git init` thì commit message viết bằng tiếng Anh.
- Chạy test từ `background_blur/`: `.venv/Scripts/python.exe -m pytest tests -q`.

## Review Focus

1. **Hai nắm tay sát nhau quanh một gậy** (runner chính và người áo trắng trong ảnh test): mỗi gậy chỉ thuộc **một** người, là người mà mask bao quanh chỗ nắm. Người còn lại tìm lại gậy của mình, hoặc không có gậy. Test `test_pole_owned_by_hand_whose_mask_wraps_grip` (Task 3) và `test_real_image_poles` (Task 3).
2. **Viền vật thể lớn hoặc đường thẳng ở nền** (viền tay áo, quần, vạch đường, hàng rào): không được nhận là gậy. Test `test_edge_of_large_region_is_not_a_pole` và `test_horizontal_bar_rejected_by_angle_prior` (Task 3).
3. **Gậy mảnh mất khi thu về 640 px:** alpha gậy ở ảnh đầy đủ phải còn ≥ 0.8 dọc trục gậy với mọi engine. Test `test_thin_pole_alpha_survives_downscale` (Task 6).
4. **Vật thò xuống dưới chân** (đầu gậy, vali): thứ tự trước/sau vẫn tính theo chân của thân người. Test `test_occlusion_uses_body_foot_not_held_object` (Task 1).
5. **Pose không thấy cổ tay** (tay vung ra sau thân, confidence thấp): không dò gậy cho tay đó, không crash, người vẫn được tách như cũ. Test `test_no_pole_search_when_wrist_conf_low` (Task 3).

---

### Task 1: Dữ liệu người mở rộng + truyền qua ROI + occlusion theo chân thân người

**Files:**
- Modify: `backend/pipeline/schema.py` (`Person`)
- Modify: `backend/pipeline/segment/base.py` (`work_scene.conv`)
- Modify: `backend/pipeline/occlusion.py` (`exclusion_mask`)
- Test: `tests/test_held_schema.py`

**Interfaces:**
- Produces (trên `Person`, mọi field có default để code cũ không đổi):
  - `body_box: Optional[Tuple[int,int,int,int]] = None`: box thân người trước khi gộp vật; `None` nghĩa là trùng `box`.
  - `hands: List[Tuple[float,float]] = []`: tâm nắm tay (toạ độ ảnh làm việc), do Task 2 điền.
  - `held: List[str] = []`: nhãn vật đã gộp, `"pole"` hoặc tên lớp COCO.
  - `poles: List["Pole"] = []`: gậy đã gộp (Task 3 định nghĩa `Pole`), dùng ở Task 6.
  - `foot_y -> int` (property): `(body_box or box)[3]`.
  - `body_h -> int` (property): chiều cao của `(body_box or box)`.
- `work_scene(...)`: `Person` ở hệ ROI giữ `body_box` (qua `box_to_work`), `hands` (đổi toạ độ `(x - roi.x1) * roi.scale`), `held`, `poles` (đổi toạ độ bằng `Pole.to_work(roi)`, định nghĩa ở Task 3; trước Task 3 thì là list rỗng).

- [ ] **Step 1: Viết test thất bại:**
  - `test_foot_y_prefers_body_box`: `Person(idx=0, box=(0,0,10,50), conf=1, body_box=(0,0,10,40)).foot_y == 40`; `body_box=None` thì bằng `50`.
  - `test_work_scene_keeps_body_box_hands_held`: ROI có scale 0.5. Kỳ vọng: `person_w.body_box == box_to_work(body_box, roi)`, `person_w.hands[0] == ((hx - roi.x1) * 0.5, (hy - roi.y1) * 0.5)`, `person_w.held == ["pole"]`.
  - `test_occlusion_uses_body_foot_not_held_object`: A có `body_box` y2 = 150, `box` y2 = 190; B có y2 = 152, mask hai người chồng nhau. Kỳ vọng: vùng chồng nhau **không** nằm trong `exclusion_mask(scene, A, A.mask)`.
- [ ] **Step 2:** `.venv/Scripts/python.exe -m pytest tests/test_held_schema.py -q`. Kỳ vọng: FAIL.
- [ ] **Step 3: Cài đặt** field và property. Sửa `conv()` trong `work_scene`. Occlusion: `ref_h = person.body_h`, `other_nearer = (o.foot_y - person.foot_y) > NEAR_TOL * ref_h`.
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest tests -q`. Kỳ vọng: toàn bộ PASS.

---

### Task 2: YOLO-pose → tâm nắm tay của từng người

**Files:**
- Create: `backend/pipeline/pose.py`
- Modify: `backend/pipeline/detect.py`: `_build_model` nhận `task`; gọi pose trong `detect_scene` sau stage `yolo`
- Modify: `backend/pipeline/config.py`
- Test: `tests/test_pose_hands.py`

**Interfaces:**
- Consumes: `Person.hands` (Task 1); `get_yolo(weights, device)` hiện có.
- Produces:
  - `pose.hand_points(kp: np.ndarray) -> List[Tuple[float,float]]`. `kp` có dạng 17×3 (COCO keypoints). Với mỗi cặp (khuỷu, cổ tay) là (7,9) và (8,10), cả hai conf ≥ `HAND_MIN_CONF` thì `hand = wrist + HAND_EXTRAP * (wrist - elbow)`.
  - `pose.attach_hands(persons: List[Person], pose_boxes: np.ndarray, pose_kps: np.ndarray) -> None`: ghép mỗi pose với person có IoU box lớn nhất (≥ 0.5), rồi điền `person.hands`.
  - `pose.detect_hands(rgb, persons, device) -> None`: chạy `get_yolo(C.POSE_WEIGHTS, device).predict(rgb, imgsz=C.POSE_IMGSZ, conf=0.25)` rồi `attach_hands`. Exception thì log warning và để `hands` rỗng.
- Config:
  ```python
  HELD_ENABLE = os.environ.get("HELD_ENABLE", "1") != "0"
  POSE_WEIGHTS = os.environ.get("POSE_WEIGHTS", "yolov8s-pose.pt")
  POSE_IMGSZ = 960
  HAND_MIN_CONF = 0.3
  HAND_EXTRAP = 0.35          # nắm tay = cổ tay + 0.35 * (cổ tay − khuỷu)
  ```
- `_build_model(weights, device)`: với openvino, `task = "pose" if "pose" in Path(weights).stem else "segment"`.

- [ ] **Step 1: Viết test thất bại:**
  - `test_hand_point_extrapolates_from_elbow`: khuỷu (100,100), cổ tay (100,200), conf 0.9. Kỳ vọng: tay = (100, 235).
  - `test_no_pole_search_when_wrist_conf_low`: cổ tay conf 0.1. Kỳ vọng: `hand_points` không có tay đó.
  - `test_attach_hands_matches_by_iou`: 2 person, 2 pose theo thứ tự đảo. Kỳ vọng: mỗi person nhận đúng tay của mình.
  - `test_detect_hands_failure_keeps_pipeline`: `get_yolo` giả ném exception. Kỳ vọng: không raise, `hands == []`.
- [ ] **Step 2:** `.venv/Scripts/python.exe -m pytest tests/test_pose_hands.py -q`. Kỳ vọng: FAIL.
- [ ] **Step 3: Cài đặt.** Trong `detect_scene`, khi `persons` không rỗng và `C.HELD_ENABLE`, mở stage `trace.stage("pose", "YOLO-pose: vị trí nắm tay")`, gọi `detect_hands(det_img, persons, device)`, rồi `s.show(...)` ảnh vẽ chấm tại các tay.
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest tests -q`. Kỳ vọng: PASS.

---

### Task 3: Dò gậy từ nắm tay (quét tia + mặt cắt dạng thanh)

**Files:**
- Create: `backend/pipeline/poles.py`
- Create: `tests/data/photo_5394198.jpg` (chép từ `../dataset/mu_cang_chai_2026_resized/photo_5394198.jpg`)
- Modify: `pytest.ini`: đăng ký marker `real` ("cần weights YOLO thật, chậm")
- Modify: `backend/pipeline/config.py`
- Test: `tests/test_poles.py`

**Interfaces:**
- Consumes: `Person.hands`, `Person.mask`, `Person.body_h` (Task 1, 2).
- Produces:
  - `@dataclass Pole: top: (x,y); tip: (x,y); half_w: float; score: float; owner: int` (idx người). Thêm `to_work(roi) -> Pole` (đổi toạ độ, nhân `half_w` theo `roi.scale`) và `polygon() -> np.ndarray` (4 đỉnh dải quanh trục, rộng `2 * half_w + 2`).
  - `poles.find_poles(rgb: np.ndarray, persons: List[Person]) -> List[Pole]`.
- Config (giá trị lấy từ probe):
  ```python
  POLE_MIN_SCORE = 0.5
  POLE_ANGLES = (45.0, 135.0)          # độ, 90 = thẳng xuống; 0.5° mỗi bước
  POLE_SEARCH = (0.06, 0.55)           # đoạn tia [r0, r1] * h tính từ nắm tay
  POLE_HAND_TOL = 0.03                 # dịch tia ±0.03h theo pháp tuyến, bước 6 px
  POLE_HALF_W = (0.002, 0.003, 0.005, 0.008, 0.012)   # nửa bề rộng gậy * h (0.012 để bắt gậy tre)
  POLE_BAR_CONTRAST = 0.05             # |Lab(tâm) − TB(Lab hai bên)| (Lab chia 255, ảnh blur σ=1.2)
  POLE_SIDE_RATIO = 0.7                # |Lab(trái) − Lab(phải)| < 0.7 * contrast (loại viền)
  POLE_GAP = 0.05                      # khoảng hụt tối đa (* h) khi dò đầu gậy
  POLE_GRIP_UP = 0.12                  # dò ngược lên trên nắm tay tối đa 0.12h (tay cầm)
  ```

- [ ] **Step 1: Viết test thất bại.** Dùng ảnh tổng hợp 600×400 nền xám 128, người là khối màu `[100:500, 150:250]`, `hands=[(260,250)]`, `h=400`:
  - `test_finds_dark_pole_from_hand`: thanh đen rộng 6 px từ (262,240) đến (300,560). Kỳ vọng: 1 `Pole`, góc trong ±3° của góc thật, `score ≥ 0.5`, `tip[1] ≥ 540`.
  - `test_edge_of_large_region_is_not_a_pole`: không có gậy, nhưng nửa phải ảnh tối (một viền thẳng đứng chạy qua tay). Kỳ vọng: `[]`.
  - `test_horizontal_bar_rejected_by_angle_prior`: thanh nằm ngang từ tay sang phải. Kỳ vọng: `[]`.
  - `test_pole_owned_by_hand_whose_mask_wraps_grip`: hai người, tay cách nhau 40 px, một gậy. Mask của A phủ vùng đĩa quanh chỗ nắm trên trục gậy; mask của B thì không. Kỳ vọng: 1 gậy, `owner == A.idx`, B không có gậy.
  - `@pytest.mark.real test_real_image_poles` (skip nếu `yolov8s-pose.pt` hoặc `yolov8s-seg.pt` không nạp được): ảnh test resize cạnh dài 2560, `detect_scene` + `find_poles`. Kỳ vọng:
    - Người có box giao `(1225,5,2345,1703)` với IoU ≥ 0.7 có **2** gậy, góc trong khoảng `87.5 ± 4` và `79 ± 4`.
    - Không người nào khác sở hữu gậy có góc `79 ± 4` với `top` nằm cách gậy đó < 50 px.
    - Người box khoảng `(315,187,1071,1526)` (áo đỏ) có gậy góc `75 ± 6`.
- [ ] **Step 2:** `.venv/Scripts/python.exe -m pytest tests/test_poles.py -q`. Kỳ vọng: FAIL.
- [ ] **Step 3: Cài đặt `find_poles`**, theo đúng thuật toán đã probe:
  ```
  lab = Lab(GaussianBlur(rgb, σ=1.2)) / 255
  với mỗi người p, mỗi hand trong p.hands, h = p.body_h:
    với góc a trong POLE_ANGLES (bước 0.5°), d = (cos a, sin a), n = (−sin a, cos a):
      với ox trong [−TOL*h, +TOL*h] bước 6 px:  o = hand + ox*n
        mẫu P = o + r*d, r trong [r0*h, r1*h] bước 3 px (bỏ mẫu ngoài ảnh; < 30 mẫu thì bỏ)
        sup[i] = OR theo hw trong POLE_HALF_W*h:
                 c=lab(P), l=lab(P−(hw+4)n), r=lab(P+(hw+4)n)
                 |c−(l+r)/2| > CONTRAST  và  |l−r| < SIDE_RATIO*|c−(l+r)/2|
        score = mean(sup); ghi nhớ hw cho nhiều sup nhất
    ứng viên của tay = (score, a, ox, hw) tốt nhất; bỏ nếu score < POLE_MIN_SCORE
  ```
  - **Đầu gậy:** đi dọc tia từ `r0*h`; `tip` là mẫu `sup` cuối cùng trước khi gặp một khoảng hụt dài hơn `POLE_GAP*h`, hoặc tại mép ảnh.
  - **Đầu trên:** đi ngược từ nắm tay theo `−d` tối đa `POLE_GRIP_UP*h`, cùng quy tắc hụt, ra `top`.
  - **Chủ của gậy:** hai ứng viên được coi là cùng một gậy khi khoảng cách vuông góc trung bình giữa hai trục < `2*hw + 6` px. Khi đó gậy thuộc người có **tỉ lệ pixel mask cao nhất** trong đĩa bán kính `0.04h` quanh điểm trên trục gần nắm tay nhất. Hoà thì xét tay gần trục hơn. Người thua chạy lại việc quét, **loại** các góc trong ±3° của gậy đã bị lấy, và nhận ứng viên mới nếu có.
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest tests/test_poles.py -q -m "not real"` phải PASS; sau đó chạy `-m real`, kỳ vọng PASS.

---

### Task 4: Gộp gậy vào mask người + stage trace

**Files:**
- Create: `backend/pipeline/held.py`
- Modify: `backend/pipeline/detect.py` (`detect_scene`, sau stage `pose`)
- Test: `tests/test_held_merge.py`

**Interfaces:**
- Consumes: `find_poles`, `Pole.polygon()` (Task 3).
- Produces:
  - `held.merge_pole(person: Person, pole: Pole, shape) -> None`: lần gộp đầu thì set `body_box = box`. Sau đó `fillConvexPoly(mask, pole.polygon(), 255)`, `box = hợp(box, bbox polygon)` (cắt vào ảnh), `held.append("pole")`, `poles.append(pole)`.
  - Trong `detect_scene`: stage `trace.stage("held", "Đồ vật cầm tay (gậy, …)")`, gọi `find_poles` rồi `merge_pole`. Note: `"{n} gậy"`. Overlay: mask người được chọn đã gộp.
- Đặt trước `suggest_person`, để điểm chọn người dùng mask đã gộp. Thứ tự và idx người không đổi.

- [ ] **Step 1: Viết test thất bại:**
  - `test_merge_pole_extends_mask_and_box`: kỳ vọng mask có 255 ở điểm giữa trục gậy, `box` bao `tip`, `body_box` là box cũ, `held == ["pole"]`.
  - `test_detect_scene_without_hands_unchanged`: `get_yolo` giả cho pose trả rỗng. Kỳ vọng: mask, box của người giống hệt trước, có stage `held` với note `"0 gậy"`.
- [ ] **Step 2:** chạy, kỳ vọng FAIL. **Step 3:** cài đặt. **Step 4:** `.venv/Scripts/python.exe -m pytest tests -q -m "not real"`, kỳ vọng PASS.

---

### Task 5: Vật COCO gần nắm tay (chai, điện thoại, túi…)

**Files:**
- Modify: `backend/pipeline/held.py`, `backend/pipeline/detect.py` (`run()` và `_persons_from_result`), `backend/pipeline/config.py`
- Modify: `tests/test_detect_device.py:86` (assertion `classes`) và `_Res` (thêm `boxes.cls`)
- Modify: `README.md`: biến môi trường `HELD_ENABLE`, `HELD_CLASSES`, `POSE_WEIGHTS`
- Test: `tests/test_held_merge.py` (thêm)

**Interfaces:**
- Produces:
  - `held.HeldObject(cls: int, name: str, box, conf: float, mask: np.ndarray)`.
  - `held.attach_objects(persons, objects) -> int`. Một vật được gộp vào người p khi: `conf ≥ HELD_MIN_CONF`; diện tích ≤ `HELD_MAX_AREA` × diện tích mask p; và **(p có `hands`)** box vật cách một tay của p ≤ `HELD_HAND_DIST * p.body_h`, **hoặc (p không có `hands`)** mask vật chạm `dilate(p.mask, HELD_TOUCH_FRAC * cạnh dài box)`. Vật gần nhiều người thì thuộc người có tay gần nhất (hoặc tiếp xúc nhiều nhất). `held.append(name)`.
  - `_persons_from_result` đọc `getattr(res.boxes, "cls", None)`; nếu `None` thì coi mọi box là người (tương thích fake/kết quả cũ). Sort và đánh `idx` người **trước**, rồi trả `(persons, objects)` cho `detect_scene`. Gộp vật diễn ra trong stage `held` **sau** khi đã có tay. `detect_persons(...)` (API công khai, test cũ dùng) vẫn trả `List[Person]`, bỏ qua `objects`.
- Config:
  ```python
  _HELD_DEFAULT = ("24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,"
                   "50,51,52,53,54,55,63,64,65,66,67,73,74,75,76,77,78,79")
  HELD_CLASSES = [int(c) for c in os.environ.get("HELD_CLASSES", _HELD_DEFAULT).replace(" ", "").split(",") if c]
  HELD_MIN_CONF = 0.25
  HELD_HAND_DIST = 0.08
  HELD_TOUCH_FRAC = 0.03
  HELD_MAX_AREA = 1.5
  ```
  `run()`: `classes = [C.PERSON_CLASS] + (C.HELD_CLASSES if C.HELD_ENABLE else [])`.

- [ ] **Step 1: Viết/sửa test:**
  - Sửa `test_detect_persons_passes_device_kwargs_to_model`: `kw["classes"] == [C.PERSON_CLASS] + C.HELD_CLASSES`.
  - `test_result_without_cls_treated_as_person`: fake boxes không có `cls`, trả 1 người.
  - `test_object_near_hand_attached`: chai (cls 39) cách tay 0.03h. Kỳ vọng: gộp, `held == ["bottle"]`.
  - `test_object_touching_shoulder_but_far_from_hands_not_attached`: vật chạm vai, cách tay 0.3h. Kỳ vọng: không gộp.
  - `test_low_conf_object_not_attached`: conf 0.12. Kỳ vọng: không gộp.
  - `test_held_disabled`: `HELD_ENABLE=False`. Kỳ vọng: `classes == [PERSON_CLASS]`, không chạy pose (pose giả ghi số lần gọi = 0).
- [ ] **Step 2:** chạy, kỳ vọng FAIL. **Step 3:** cài đặt. **Step 4:** `.venv/Scripts/python.exe -m pytest tests -q -m "not real"`, kỳ vọng PASS.

---

### Task 6: Alpha của gậy ở độ phân giải gốc

**Files:**
- Create: `backend/pipeline/thin.py`
- Modify: `backend/pipeline/segment/__init__.py` (`build_cutout`: gọi sau engine, trước stage `compose`)
- Modify: `backend/pipeline/config.py`
- Test: `tests/test_thin_alpha.py`

**Interfaces:**
- Consumes: `Person.poles` (toạ độ ảnh đầy đủ), `Scene.others`, `exclusion_mask`, `make_trimap`/`matte` (matting.py).
- Produces: `thin.overlay_poles(cutout: Cutout, scene: Scene, person: Person, trace) -> Cutout`. Với mỗi gậy:
  1. Lấy dải hình chữ nhật xoay quanh trục, rộng `2 * (half_w + THIN_BAND_PX + 4)`, rồi crop bbox của dải ở **ảnh đầy đủ**.
  2. Trimap: trong polygon gậy co lại `THIN_BAND_PX` là 255; trong polygon nở `THIN_BAND_PX` mà không thuộc phần co là 128; còn lại 0.
  3. `matte(crop, trimap, "cf")` ra `a_pole`.
  4. Ghép `alpha = max(alpha, a_pole)` trong crop, trừ vùng `exclusion_mask` của người khác (tính ở ảnh đầy đủ).
  5. Pixel có alpha tăng: `foreground = rgb` (gậy không cần khử nhiễm phức tạp), `background` giữ nguyên.

  Stage trace `"thin", "Alpha gậy ở độ phân giải gốc"`. Person không có `poles` thì trả `cutout` y nguyên.
- Config: `THIN_BAND_PX = 2`.
- Áp dụng cho **mọi engine** (cf/knn/birefnet/focus). Không sửa engine, nên các khoá cache của engine không bị ảnh hưởng.

- [ ] **Step 1: Viết test thất bại:**
  - `test_thin_pole_alpha_survives_downscale`: ảnh 2400×1600, người lớn (ROI scale < 0.4), gậy đen rộng 8 px, engine `birefnet` với `birefnet_alpha` giả chỉ trả thân người (không có gậy). Kỳ vọng: sau `build_cutout`, `alpha ≥ 0.8` tại 20 điểm đều nhau dọc trục gậy (ngoài thân); `alpha ≤ 0.1` cách trục 20 px.
  - `test_overlay_respects_other_person_exclusion`: gậy đi qua mask của người khác đứng gần hơn. Kỳ vọng: alpha ở phần giao không tăng.
  - `test_no_poles_cutout_unchanged`: `np.array_equal` alpha trước và sau.
- [ ] **Step 2:** chạy, kỳ vọng FAIL. **Step 3:** cài đặt. **Step 4:** `.venv/Scripts/python.exe -m pytest tests -q -m "not real"`, kỳ vọng PASS.
- [ ] **Step 5: Chạy thử app** (`uvicorn backend.main:app` + `streamlit run frontend/app.py`) với `photo_5394198.jpg`, chọn runner áo hồng. Kiểm tra bằng mắt:
  - Stage `pose` có chấm ở hai nắm tay; stage `held` báo `"2 gậy"`.
  - Ở cả engine `birefnet` và `cf`, hai gậy giữ màu đến mép dưới ảnh.
  - Gậy tre của người áo trắng **không** giữ màu.

---

### Task 7 (tuỳ chọn, làm sau): BiRefNet giữ mảng dính liền người cho vật lạ

Dành cho bình mềm, khăn, cờ cầm tay: những vật không có trong COCO và không có dạng gậy.
- `gate_to_person(alpha, init, box, exclude=None, extend: Optional[bool]=None)`: giữ thành phần `(alpha > 0.5) & ngoài gate_bin & ~exclude` chạm mép gate, với diện tích ≤ `HELD_EXT_MAX_AREA = 0.35` × diện tích mask và bbox nằm trong box nở `HELD_EXT_REACH = 0.35` × cạnh dài. Gate được nâng bằng `dilate(kept, 2r+1)`.
- Nếu đã có `hands`, chỉ giữ thành phần nằm cách một tay ≤ `0.1h`.
- Test: `test_gate_keeps_attached_object_beyond_gate`, `test_gate_extension_rejects_oversized_region`, `test_gate_extension_respects_exclude`, `test_gate_extension_off`. Các test gate cũ phải vẫn pass.

---

## Giới hạn đã biết / để giai đoạn sau

- **Cổ tay bị che** (tay ra sau thân, pose conf < 0.3) thì không dò được gậy cho tay đó.
- **Gậy cong hoặc gậy gấp** (dạng chữ Z) không phải đường thẳng nên không được bắt. Gậy gắn trên áo vest thì đã nằm trong mask thân.
- **Đầu gậy nhoè** do chuyển động có thể bị ngắn hơn thực tế.
- **SAM/MobileSAM** (gợi ý bằng điểm dọc trục gậy hoặc box vật) cho viền sát hơn và xử lý được vật lạ. Đây là bước nâng cấp sau khi Task 1–6 chạy ổn; khi đó thay bước 2 của Task 6 (trimap theo polygon) bằng mask SAM.
- Ngưỡng hiện chỉ được chỉnh trên **một** ảnh. Cần thêm vài ảnh khác của cùng giải (có gậy, không gậy, ngược sáng) để chỉnh lại `POLE_MIN_SCORE` và `POLE_BAR_CONTRAST`.
