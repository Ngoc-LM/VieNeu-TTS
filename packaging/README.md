# 📦 Đóng gói VieNeu Voice Clone (app desktop Windows)

App desktop tối giản cho **VieNeu-TTS v3 Turbo**, chỉ làm một việc: **nhân bản
giọng nói** từ một clip mẫu 3–8 giây, rồi dùng chính giọng đó đọc text.

Chạy hoàn toàn trên **CPU qua ONNX Runtime** (torch-free) — không cần GPU,
không cần cài Python trên máy người dùng cuối.

| | |
|---|---|
| Mã nguồn app | [`apps/voice_clone_app.py`](../apps/voice_clone_app.py) |
| Spec PyInstaller | [`vieneu_clone.spec`](vieneu_clone.spec) |
| Script build | [`build_windows.ps1`](build_windows.ps1) |
| CI build | [`.github/workflows/build-windows-app.yml`](../.github/workflows/build-windows-app.yml) |
| Hướng dẫn cho người dùng cuối | [`DOC_NGUOI_DUNG.txt`](DOC_NGUOI_DUNG.txt) |

---

## 1. Chạy thử từ source (không cần đóng gói)

```bash
uv sync            # cài phụ thuộc torch-free
uv run vieneu-clone
```

Mở http://127.0.0.1:7861. Muốn tự mở trình duyệt: đặt `VIENEU_OPEN_BROWSER=1`.

## 2. Build app Windows

> ⚠️ **PyInstaller không cross-compile.** Bản `.exe` cho Windows **bắt buộc**
> phải build trên máy Windows (hoặc runner `windows-latest` của GitHub Actions).
> Build trên Linux/macOS chỉ tạo ra binary của chính hệ đó.

### Cách A — build trên máy Windows

Yêu cầu: Windows x64, Python 3.10–3.13 trong PATH.

```powershell
git clone https://github.com/pnnbao97/VieNeu-TTS.git
cd VieNeu-TTS
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -Zip
```

Script sẽ tạo venv build riêng, cài phụ thuộc torch-free, chạy PyInstaller.

Kết quả:

```
packaging\dist\VieNeuVoiceClone\VieNeuVoiceClone.exe   ← bấm đúp để chạy
packaging\dist\VieNeuVoiceClone-win64.zip              ← bản để phát hành (-Zip)
```

Tuỳ chọn: `-Clean` xoá venv/build cũ trước khi dựng lại từ đầu.

### Cách B — để GitHub Actions build

1. Vào tab **Actions** → **Build Windows app (VieNeu Voice Clone)** → **Run workflow**.
2. Tải artifact `VieNeuVoiceClone-win64` khi job xong.

Workflow cũng tự chạy khi push tag dạng `app-v*` (ví dụ `app-v1.0.0`).
Trước khi nén, CI khởi động chính file `.exe` vừa dựng và gọi HTTP vào nó để
chắc chắn bundle không thiếu module — bundle hỏng sẽ làm job fail thay vì lọt
ra tay người dùng.

---

## 3. Bản đóng gói hoạt động thế nào

**Model không nằm trong bundle.** Bundle chỉ chứa code + runtime (~1 GB, chủ yếu
là `onnxruntime` và `gradio`). Lần chạy đầu tiên app tải trọng số v3 Turbo từ
HuggingFace về máy; từ lần sau chạy offline.

Thư mục dữ liệu (ghi được, tồn tại qua các lần cập nhật app):

```
%LOCALAPPDATA%\VieNeuVoiceClone\models    ← cache model (HF_HOME)
%LOCALAPPDATA%\VieNeuVoiceClone\outputs   ← file WAV đã tạo
```

Đây là lý do app phải tự đặt `HF_HOME`: bản đóng gói thường nằm trong thư mục
chỉ đọc, còn PyInstaller giải nén vào thư mục tạm bị xoá sau mỗi lần chạy — để
mặc định thì mỗi lần mở app là tải lại model từ đầu.

Khi khởi động, app nạp model ở **luồng nền** nên giao diện mở ra ngay; nếu bấm
"Tạo giọng nói" trước khi model xong thì thao tác chỉ chờ chứ không lỗi.

### Vì sao `onedir` chứ không `onefile`

Bundle ~1 GB. Bản `onefile` phải giải nén toàn bộ ra thư mục tạm ở **mỗi lần
mở app**, làm thời gian khởi động tăng vọt. Muốn người dùng chỉ thấy một file
duy nhất thì bọc thư mục `onedir` bằng installer (Inno Setup / NSIS).

### Vì sao loại torch ra khỏi bundle

App ép `backend="onnx"`, nên toàn bộ nhánh PyTorch là trọng lượng chết — giữ
lại sẽ làm bundle phình thêm khoảng 2 GB. Spec liệt kê `torch`, `torchaudio`,
`transformers`, `neucodec`… trong `excludes`, và script build còn gỡ torch khỏi
venv build một lần nữa cho chắc.

---

## 4. Xử lý sự cố khi build

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| App chạy báo `ModuleNotFoundError: gradio...` | Gradio gọi `inspect.getsource()` lên code của chính nó. Spec đã xử lý bằng `module_collection_mode = {"gradio": "py"}` — đừng bỏ dòng này. |
| `FileNotFoundError` một file `.json`/`.js` của gradio | Thiếu data files. Kiểm tra `collect_data_files(..., include_py_files=True)` trong spec. |
| Lỗi DLL của onnxruntime | Kiểm tra `collect_dynamic_libs("onnxruntime")` còn trong spec. |
| Bundle to bất thường (>2 GB) | torch lọt vào venv build. Chạy lại với `-Clean`. |
| Cảnh báo `Failed to collect submodules for 'vieneu.v3_turbo_serve' ... No module named 'torch'` | **Bình thường, bỏ qua.** Đó là server GPU, cố ý không đóng gói. |
| `TypeError: Audio.__init__() got an unexpected keyword argument ...` | Khác biệt API giữa gradio 5 và 6. App đã hỗ trợ cả hai qua `_gradio_major()`; nếu thêm tham số mới cho component thì nhớ kiểm trên cả hai phiên bản. |
| Antivirus chặn `.exe` | Thường gặp với binary PyInstaller chưa ký. Muốn phát hành rộng nên ký code (code signing certificate). |

---

## 5. Giới hạn đã biết

- **Chỉ CPU.** App ép đường ONNX. Tốc độ khoảng RTF 0.5 (~2× real-time) trên
  CPU phổ thông. Cần tốc độ GPU thì dùng `uv sync --extra cuda` với Web UI đầy
  đủ (`vieneu-web`) thay vì app này.
- **Cần Internet ở lần chạy đầu** để tải model.
- **Giới hạn 3000 ký tự** mỗi lần tạo; văn bản dài hơn cần chia nhỏ.
- Bản `.exe` **chưa được ký số**, nên SmartScreen/antivirus có thể cảnh báo.
- App chạy được trên **cả gradio 5 và gradio 6** (`pyproject` khai báo
  `gradio>=5.49.1`, nên `uv sync` theo lockfile ra bản 5 còn `pip install`
  mới ra bản 6). Sửa giao diện thì kiểm trên cả hai.

## 6. Lưu ý sử dụng

Chỉ nhân bản giọng của chính bạn hoặc giọng bạn có sự đồng ý rõ ràng của chủ
giọng. Audio do app tạo ra được đóng dấu chìm (watermark) theo mặc định của SDK.
