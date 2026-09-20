"""VieNeu Voice Clone — app desktop tối giản cho VieNeu-TTS v3 Turbo.

Chỉ một việc: lấy một clip giọng mẫu 3–8 giây, rồi đọc text bằng chính giọng đó.

Chạy từ source:      uv run vieneu-clone
Chạy bản đóng gói:   VieNeuVoiceClone.exe   (xem packaging/README.md)

App chạy hoàn toàn trên CPU qua ONNX Runtime (torch-free). Lần chạy đầu tiên
tải model v3 Turbo về cache rồi dùng lại offline cho các lần sau.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Đặt trước khi import gradio: bản đóng gói không gọi về server analytics.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ── Thư mục dữ liệu ────────────────────────────────────────────────────────────
def is_frozen() -> bool:
    """True khi đang chạy trong bản PyInstaller đã đóng gói."""
    return bool(getattr(sys, "frozen", False))


def app_data_dir() -> Path:
    """Thư mục ghi được, tồn tại qua các lần cập nhật app.

    Windows  → %LOCALAPPDATA%\\VieNeuVoiceClone
    macOS    → ~/Library/Application Support/VieNeuVoiceClone
    Linux    → ~/.local/share/VieNeuVoiceClone
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    d = Path(base) / "VieNeuVoiceClone"
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_cache_dirs() -> Path:
    """Trỏ cache HuggingFace vào thư mục dữ liệu của app.

    Bản đóng gói nằm trong Program Files (chỉ đọc) và PyInstaller giải nén vào
    thư mục tạm bị xoá sau mỗi lần chạy — model ~vài trăm MB phải nằm ngoài cả
    hai chỗ đó, nếu không mỗi lần mở app là tải lại.
    """
    models = app_data_dir() / "models"
    models.mkdir(parents=True, exist_ok=True)
    # Chỉ đặt khi người dùng chưa tự cấu hình HF_HOME.
    os.environ.setdefault("HF_HOME", str(models))
    return models


OUTPUT_DIR = app_data_dir() / "outputs"


# ── Nạp model (lười, chạy nền) ────────────────────────────────────────────────
_tts = None
_tts_lock = threading.Lock()


def get_tts():
    """Trả về instance Vieneu đã sẵn sàng; nạp một lần duy nhất, thread-safe.

    Chỉ gán ``_tts`` khi nạp thành công, nên một lần thất bại (rớt mạng lúc
    tải model) không bị nhớ lại — lần gọi sau vẫn thử lại được.
    """
    global _tts
    if _tts is not None:
        return _tts
    with _tts_lock:
        if _tts is not None:
            return _tts
        from vieneu import Vieneu

        # backend="onnx": ép đường CPU torch-free kể cả khi máy có sẵn torch,
        # để bản đóng gói luôn chạy đúng như lúc build.
        _tts = Vieneu(mode="v3turbo", backend="onnx")
        return _tts


def preload_in_background() -> None:
    """Tải/nạp model ngay khi app mở, để lần bấm 'Tạo giọng nói' đầu không phải chờ.

    Chỉ được gọi SAU khi main thread đã dựng xong UI — xem ghi chú trong ``main()``.
    """

    def _run():
        try:
            get_tts()
            print("✅ Model đã sẵn sàng.", flush=True)
        except Exception as exc:  # noqa: BLE001 — chỉ log; app vẫn chạy, thử lại khi bấm nút
            print(f"❌ Chưa nạp được model: {exc}", flush=True)
            print("   App vẫn mở; bấm 'Tạo giọng nói' để thử lại.", flush=True)

    threading.Thread(target=_run, daemon=True).start()


# ── Sinh audio ────────────────────────────────────────────────────────────────
MAX_CHARS = 3000


def synthesize(ref_audio: Optional[str], text: str, denoise: bool):
    """Clone giọng từ ``ref_audio`` rồi đọc ``text``.

    Trả về ``(audio_cho_player, dòng_trạng_thái)``.
    """
    import gradio as gr

    if not ref_audio:
        raise gr.Error("Hãy tải lên hoặc ghi âm một clip giọng mẫu 3–8 giây trước.")
    text = (text or "").strip()
    if not text:
        raise gr.Error("Hãy nhập đoạn text cần đọc.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text dài {len(text)} ký tự, vượt giới hạn {MAX_CHARS}. Hãy chia nhỏ.")

    started = time.time()
    # Cố nạp lại thay vì nhớ lỗi cũ: thất bại lúc khởi động
    # thường chỉ là rớt mạng khi tải model, và người dùng bấm lại sau khi có
    # mạng thì phải chạy được, không bắt họ khởi động lại app.
    try:
        tts = get_tts()
    except Exception as exc:  # noqa: BLE001 — đổi thành thông báo đọc được trong UI
        raise gr.Error(
            f"Không nạp được model: {exc}\n"
            "Kiểm tra kết nối Internet (lần chạy đầu cần tải model) rồi bấm lại."
        ) from exc
    audio = tts.infer(text, ref_audio=ref_audio, denoise=bool(denoise))
    elapsed = time.time() - started

    sr = tts.sample_rate
    duration = len(audio) / sr if sr else 0.0
    rtf = elapsed / duration if duration else 0.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"vieneu_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    tts.save(audio, out_path)

    status = (
        f"✅ Xong sau {elapsed:.1f}s · audio {duration:.1f}s · RTF {rtf:.2f} "
        f"({1 / rtf:.1f}× real-time)\n📁 Đã lưu: {out_path}"
        if rtf
        else f"✅ Xong sau {elapsed:.1f}s\n📁 Đã lưu: {out_path}"
    )
    return str(out_path), status


# ── Giao diện ─────────────────────────────────────────────────────────────────
SAMPLE_TEXT = (
    "Xin chào, đây là giọng nói được nhân bản bằng VieNeu-TTS phiên bản ba Turbo. "
    "Bạn chỉ cần một đoạn ghi âm ngắn là có thể tạo ra giọng đọc của riêng mình."
)

CSS = """
.vieneu-header { text-align: center; margin-bottom: 0.5rem; }
.vieneu-header h1 { margin-bottom: 0.2rem; }
footer { display: none !important; }
"""


# pyproject khai báo `gradio>=5.49.1`, nên máy này cài gradio 5 mà máy khác cài
# gradio 6 là chuyện bình thường — app phải chạy được trên cả hai. Khác biệt ở
# những chỗ app đụng tới:
#   - Audio(show_download_button=...): gradio 6 bỏ hẳn (mặc định đã có nút tải).
#   - css/theme: gradio 5 nhận ở Blocks(), gradio 6 chuyển sang launch().
#   - launch(show_api=...): gradio 6 bỏ.
def _gradio_major() -> int:
    """Số major của gradio đang cài; mặc định 5 nếu không đọc được."""
    import gradio as gr

    try:
        return int(str(gr.__version__).split(".")[0])
    except (ValueError, IndexError):
        return 5


def _style_kwargs() -> dict:
    """css + theme, để gắn vào Blocks() (gradio 5) hoặc launch() (gradio 6)."""
    import gradio as gr

    return {"css": CSS, "theme": gr.themes.Soft()}


def build_ui():
    import gradio as gr

    blocks_kwargs: dict = {"title": "VieNeu Voice Clone"}
    if _gradio_major() < 6:
        blocks_kwargs.update(_style_kwargs())

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(
            """
            <div class="vieneu-header">
              <h1>🦜 VieNeu Voice Clone</h1>
              <p>Nhân bản giọng nói tiếng Việt từ một clip 3–8 giây — VieNeu-TTS v3 Turbo, 48 kHz, chạy hoàn toàn trên máy bạn.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                ref_audio = gr.Audio(
                    label="1️⃣ Giọng mẫu (3–8 giây, nói rõ, ít tiếng ồn)",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                denoise = gr.Checkbox(
                    value=True,
                    label="Tự động khử nhiễu clip mẫu",
                    info="Nên bật. Tắt nếu clip đã rất sạch và bạn muốn giữ nguyên chất giọng.",
                )
                text = gr.Textbox(
                    label="2️⃣ Text cần đọc",
                    value=SAMPLE_TEXT,
                    lines=7,
                    max_lines=20,
                    placeholder="Nhập đoạn văn bản tiếng Việt…",
                )
                generate_btn = gr.Button("🎙️ Tạo giọng nói", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="3️⃣ Kết quả (bấm ⬇ để tải file WAV)",
                    type="filepath",
                    autoplay=False,
                )
                status = gr.Markdown("⏳ Đang chuẩn bị model… lần chạy đầu tiên cần tải khoảng vài trăm MB.")
                gr.Markdown(
                    "**Mẹo dùng tốt hơn**\n"
                    "- Clip mẫu 3–8 giây, một người nói, không nhạc nền.\n"
                    "- Ghi âm ở nơi yên tĩnh cho độ giống cao nhất.\n"
                    "- Chèn `[cười]`, `[thở dài]`, `[hắng giọng]` vào text để thêm sắc thái *(thử nghiệm)*.\n"
                    "- File WAV được lưu tự động vào thư mục dữ liệu của app."
                )

        generate_btn.click(
            fn=synthesize,
            inputs=[ref_audio, text, denoise],
            outputs=[output_audio, status],
            api_name="synthesize",
        )

        # Người dùng chỉ được quyền dùng giọng mình có quyền sử dụng — nói rõ trong UI.
        gr.Markdown(
            "---\n"
            "⚠️ Chỉ nhân bản giọng của chính bạn hoặc giọng bạn có sự đồng ý rõ ràng của chủ giọng. "
            "Audio tạo ra được đóng dấu chìm (watermark)."
        )

    return demo


# ── Điểm vào ──────────────────────────────────────────────────────────────────
def main() -> None:
    setup_cache_dirs()
    print("🦜 VieNeu Voice Clone đang khởi động…", flush=True)
    print(f"📂 Dữ liệu app: {app_data_dir()}", flush=True)

    # Dựng UI trước, rồi mới mở luồng nạp model: như vậy các package nặng
    # (numpy, gradio) được import xong trên main thread, thread nền chỉ lấy lại
    # từ sys.modules thay vì cùng lúc import lần đầu. Không bắt buộc, nhưng
    # tránh hẳn một lớp lỗi import khó tái hiện trong bản đóng gói.
    demo = build_ui()
    preload_in_background()

    # inbrowser: bản đóng gói tự mở trình duyệt để người dùng chỉ cần bấm một lần.
    launch_kwargs: dict = {
        "server_name": os.environ.get("VIENEU_HOST", "127.0.0.1"),
        "server_port": int(os.environ.get("VIENEU_PORT", "7861")),
        "inbrowser": is_frozen() or os.environ.get("VIENEU_OPEN_BROWSER") == "1",
        "quiet": False,
    }
    if _gradio_major() >= 6:
        launch_kwargs.update(_style_kwargs())
    else:
        launch_kwargs["show_api"] = False

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
