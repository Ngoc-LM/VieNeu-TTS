"""VieNeu Voice Clone — app desktop tối giản cho VieNeu-TTS v3 Turbo.

Chỉ một việc: lấy một clip giọng mẫu 3–8 giây, rồi đọc text bằng chính giọng đó.

Chạy từ source:      uv run vieneu-clone
Chạy bản đóng gói:   VieNeuVoiceClone.exe   (xem packaging/README.md)

App chạy hoàn toàn trên CPU qua ONNX Runtime (torch-free). Lần chạy đầu tiên
tải model v3 Turbo về cache rồi dùng lại offline cho các lần sau.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

# Đặt trước khi import gradio: bản đóng gói không gọi về server analytics.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ── Thư mục dữ liệu ────────────────────────────────────────────────────────────
def is_frozen() -> bool:
    """True khi đang chạy trong bản PyInstaller đã đóng gói."""
    return bool(getattr(sys, "frozen", False))


def should_open_browser() -> bool:
    """Có tự mở trình duyệt khi khởi động không.

    Mặc định: có, nếu đang chạy bản đóng gói (người dùng bấm đúp .exe thì mong
    app tự hiện ra). Biến VIENEU_OPEN_BROWSER được đặt thì nó quyết định, kể cả
    khi đang đóng gói — CI chạy chính file .exe đó để smoke test và không muốn
    runner mọc ra vài cửa sổ trình duyệt.
    """
    raw = os.environ.get("VIENEU_OPEN_BROWSER")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return is_frozen()


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


# ── Cắt văn bản dài thành đoạn ────────────────────────────────────────────────
# Số ký tự tối đa mỗi lần gọi infer(). KHÔNG phải giới hạn độ dài văn bản: văn
# bản dài bao nhiêu cũng được, app tự cắt rồi ghép lại.
#
# Vì sao không ném cả 100k ký tự vào một lần infer()? infer() tự cắt chunk 256
# ký tự và ghép, nên về mặt kết quả thì chạy được — nhưng nó giữ TOÀN BỘ audio
# trong RAM rồi mới ghép: hai tiếng audio 48 kHz float32 là ~1,5 GB, cộng thêm
# một bản sao lúc ghép nữa. Cắt đoạn rồi ghi dần xuống đĩa giữ RAM ở mức một
# đoạn, và cho phép báo tiến độ thay vì treo hàng giờ không dấu hiệu gì.
SEGMENT_CHARS = 2000

_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
# Ranh giới câu tiếng Việt: kết thúc bằng . ! ? … rồi tới khoảng trắng.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def _split_on_words(sentence: str, max_chars: int) -> List[str]:
    """Cắt một câu quá dài thành các mảnh <= ``max_chars`` ở ranh giới TỪ.

    Lối thoát cuối cùng cho văn bản không có dấu câu. Một từ dài hơn cả
    ``max_chars`` (chuỗi rác, URL khổng lồ) thì cắt cứng theo ký tự.
    """
    pieces: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for word in sentence.split():
        while len(word) > max_chars:      # từ đơn lẻ dài hơn cả giới hạn
            if buf:
                pieces.append(" ".join(buf))
                buf, buf_len = [], 0
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if buf and buf_len + len(word) + 1 > max_chars:
            pieces.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(word)
        buf_len += len(word) + 1
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def split_into_segments(text: str, max_chars: int = SEGMENT_CHARS) -> List[Tuple[str, str]]:
    """Cắt ``text`` thành các đoạn <= ``max_chars`` ký tự.

    Trả về list ``(đoạn, loại_ranh_giới_sau_đoạn)`` với loại thuộc
    {"para", "sentence", "minor"} — dùng để tính khoảng nghỉ khi ghép, đúng theo
    bảng V3_GAP_SILENCE của SDK. Phần tử cuối mang loại "" (không có ranh giới sau).

    Ưu tiên cắt ở ranh giới ĐOẠN, vì khoảng nghỉ giữa hai đoạn (0,70 s) đúng
    bằng thứ SDK sẽ chèn nếu xử lý cả bài một lần — cắt ở đó thì kết quả ghép
    lại không khác gì. Đoạn đơn lẻ dài quá thì cắt tiếp theo CÂU (0,50 s), và
    câu đơn lẻ vẫn dài quá thì cắt theo TỪ (0,30 s — ranh giới nhỏ nhất của SDK).

    Bước cắt theo từ trông thừa nhưng không phải: văn bản dán từ PDF hoặc phụ đề
    có thể dài hàng trăm nghìn ký tự mà không có lấy một dấu chấm, và nếu để
    nguyên thì cả khối đó rơi vào MỘT lần infer() — đúng cái vấn đề bộ nhớ mà
    việc cắt đoạn sinh ra để tránh.
    """
    paras = [p.strip() for p in _PARA_SPLIT_RE.split(text or "") if p.strip()]
    if not paras:
        return []

    # (text, gap_sau) — gap của phần tử cuối được sửa thành "" ở cuối hàm.
    segments: List[List[str]] = []

    def _flush(buf: List[str], gap: str) -> None:
        if buf:
            segments.append(["\n\n".join(buf), gap])

    buf: List[str] = []
    buf_len = 0
    for para in paras:
        if len(para) > max_chars:
            # Đoạn này một mình đã quá dài → xả buffer rồi cắt nó theo câu.
            _flush(buf, "para")
            buf, buf_len = [], 0
            para_start = len(segments)   # để chốt gap "para" sau khi xong cả đoạn
            sent_buf: List[str] = []
            sent_len = 0
            for sent in _SENT_SPLIT_RE.split(para):
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) > max_chars:
                    # Câu này một mình đã quá dài → xả buffer, cắt theo từ.
                    if sent_buf:
                        segments.append([" ".join(sent_buf), "sentence"])
                        sent_buf, sent_len = [], 0
                    pieces = _split_on_words(sent, max_chars)
                    if pieces:
                        for piece in pieces:
                            segments.append([piece, "minor"])
                        segments[-1][1] = "sentence"   # hết câu (có thể bị nâng
                        #                                lên "para" ở cuối đoạn)
                    continue
                if sent_buf and sent_len + len(sent) + 1 > max_chars:
                    segments.append([" ".join(sent_buf), "sentence"])
                    sent_buf, sent_len = [], 0
                sent_buf.append(sent)
                sent_len += len(sent) + 1
            if sent_buf:
                segments.append([" ".join(sent_buf), "sentence"])
            # Ranh giới sau mảnh CUỐI của đoạn này là ranh giới ĐOẠN — chốt ở đây,
            # sau khi đã xử lý xong cả đoạn. Gán sớm hơn (ngay trong nhánh cắt câu
            # hoặc cắt từ) thì mảnh cuối giữ gap "sentence" và ranh giới đoạn chỉ
            # được nghỉ 0,50 s thay vì 0,70 s.
            if len(segments) > para_start:
                segments[-1][1] = "para"
            continue

        if buf and buf_len + len(para) + 2 > max_chars:
            _flush(buf, "para")
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += len(para) + 2
    _flush(buf, "para")

    if segments:
        segments[-1][1] = ""
    return [(t, g) for t, g in segments]


# ── Sinh audio ────────────────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    """Giây -> chuỗi kiểu '1 giờ 12 phút' / '3 phút 5 giây' cho người đọc."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h} giờ {m} phút"
    if m:
        return f"{m} phút {sec} giây"
    return f"{sec} giây"


def _new_output_path() -> Path:
    """Đường dẫn WAV mới, không đụng file cũ.

    Hậu tố ngẫu nhiên: mốc thời gian tới giây thôi thì hai lần tạo sát nhau
    (hoặc hai tab cùng bấm) sẽ ra trùng tên và file trước bị ghi đè — trong khi
    app hứa với người dùng là mọi kết quả đều được lưu lại.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"vieneu_{stamp}_{uuid.uuid4().hex[:8]}.wav"


# Ước lượng thô để báo trước cho người dùng, KHÔNG dùng vào tính toán gì khác.
# ~15 ký tự tiếng Việt cho mỗi giây audio, và RTF ~0,5 trên CPU phổ thông (xem
# mục Benchmarks trong README). Sai số lớn là chấp nhận được — mục đích chỉ là
# phân biệt "vài giây" với "hơn một tiếng".
CHARS_PER_AUDIO_SECOND = 15.0
CPU_RTF_ESTIMATE = 0.5


def describe_workload(text: str) -> str:
    """Dòng gợi ý dưới ô text: bao nhiêu đoạn, dự kiến bao lâu."""
    text = (text or "").strip()
    if not text:
        return ""
    segments = split_into_segments(text)
    est_audio = len(text) / CHARS_PER_AUDIO_SECOND
    est_work = est_audio * CPU_RTF_ESTIMATE
    n_chars = f"{len(text):,}".replace(",", ".")   # chỉ dấu phân cách hàng nghìn
    line = (
        f"{n_chars} ký tự · {len(segments)} đoạn · "
        f"ước tính ~{_fmt_duration(est_audio)} audio, xử lý ~{_fmt_duration(est_work)}"
    )
    if est_work >= 600:
        return f"⏳ {line} — khá lâu, bạn có thể bấm **Dừng** giữa chừng."
    return line


def synthesize(ref_audio: Optional[str], text: str, denoise: bool, progress=None):
    """Clone giọng từ ``ref_audio`` rồi đọc ``text``.

    Không giới hạn độ dài: văn bản dài được cắt đoạn, sinh lần lượt và GHI DẦN
    xuống file WAV, nên RAM chỉ giữ một đoạn tại một thời điểm.

    Là GENERATOR, không phải hàm thường: mỗi đoạn xong thì yield một lần. Nhờ đó
    nút Dừng huỷ được thật (xem ghi chú ở chỗ yield) và trạng thái cập nhật dần.
    Yield ``(None, trạng_thái)`` trong lúc chạy, ``(đường_dẫn_wav, tổng_kết)`` khi xong.
    """
    import gradio as gr
    import numpy as np
    import soundfile as sf
    from vieneu_utils.core_utils import V3_GAP_SILENCE, pause_pad_samples

    if progress is None:
        progress = gr.Progress()

    if not ref_audio:
        raise gr.Error("Hãy tải lên hoặc ghi âm một clip giọng mẫu 3–8 giây trước.")
    text = (text or "").strip()
    if not text:
        raise gr.Error("Hãy nhập đoạn text cần đọc.")

    # Cố nạp lại thay vì nhớ lỗi cũ: thất bại lúc khởi động
    # thường chỉ là rớt mạng khi tải model, và người dùng bấm lại sau khi có
    # mạng thì phải chạy được, không bắt họ khởi động lại app.
    progress(0.0, desc="Chuẩn bị model…")
    try:
        tts = get_tts()
    except Exception as exc:  # noqa: BLE001 — đổi thành thông báo đọc được trong UI
        raise gr.Error(
            f"Không nạp được model: {exc}\n"
            "Kiểm tra kết nối Internet (lần chạy đầu cần tải model) rồi bấm lại."
        ) from exc

    segments = split_into_segments(text)
    if not segments:
        raise gr.Error("Không tìm thấy nội dung đọc được trong text.")

    sr = tts.sample_rate
    out_path = _new_output_path()
    started = time.time()
    total = len(segments)
    n_samples = 0

    # Ghi dần: giữ lại đoạn TRƯỚC để tính khoảng nghỉ với đoạn kế (pause_pad_samples
    # cần cả hai để đo im lặng sẵn có ở đuôi/đầu), rồi mới ghi nó ra. Nhờ vậy RAM
    # chỉ giữ hai đoạn, không phải cả bài.
    prev_wav = None
    prev_gap = ""
    try:
        with sf.SoundFile(str(out_path), "w", samplerate=sr, channels=1,
                          subtype="PCM_16") as fh:
            for i, (seg_text, gap) in enumerate(segments):
                progress(
                    i / total,
                    desc=f"Đang sinh đoạn {i + 1}/{total}"
                         + (f" · đã có {_fmt_duration(n_samples / sr)} audio" if n_samples else ""),
                )
                wav = tts.infer(seg_text, ref_audio=ref_audio, denoise=bool(denoise))
                if prev_wav is not None:
                    pad = pause_pad_samples(
                        prev_wav, wav, sr, V3_GAP_SILENCE.get(prev_gap, 0.5)
                    )
                    fh.write(prev_wav)
                    n_samples += len(prev_wav)
                    if pad > 0:
                        fh.write(np.zeros(pad, dtype=np.float32))
                        n_samples += pad
                prev_wav, prev_gap = wav, gap

                # Điểm nhả duy nhất cho nút Dừng. Gradio KHÔNG cắt ngang được một
                # hàm thường đang chạy (Python không kill thread giữa chừng) — huỷ
                # một hàm thường chỉ bỏ kết quả, CPU vẫn chạy nốt hàng giờ. Với
                # generator thì Gradio đóng generator, GeneratorExit bật lên đúng
                # chỗ yield này, và ta dừng thật ở ranh giới đoạn.
                if i + 1 < total:
                    yield None, (
                        f"⏳ Đang sinh đoạn {i + 1}/{total} · "
                        f"đã có {_fmt_duration(n_samples / sr)} audio"
                    )
            if prev_wav is not None:
                fh.write(prev_wav)
                n_samples += len(prev_wav)
    except BaseException:
        # GeneratorExit (bấm Dừng) là BaseException, không phải Exception — phải
        # bắt cả hai, nếu không bản dở dang sẽ nằm lại trong thư mục output và
        # trông y như một kết quả hợp lệ.
        out_path.unlink(missing_ok=True)
        raise

    progress(1.0, desc="Hoàn tất")
    elapsed = time.time() - started
    duration = n_samples / sr if sr else 0.0
    rtf = elapsed / duration if duration else 0.0

    parts = [
        f"✅ Xong sau {_fmt_duration(elapsed)} · audio {_fmt_duration(duration)}",
    ]
    if total > 1:
        parts.append(f"🧩 {total} đoạn, ghép tự động")
    if rtf:
        parts.append(f"📊 RTF {rtf:.2f} ({1 / rtf:.1f}× real-time)")
    parts.append(f"📁 Đã lưu: {out_path}")
    yield str(out_path), "\n".join(parts)


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
                    label="2️⃣ Text cần đọc (dài bao nhiêu cũng được)",
                    value=SAMPLE_TEXT,
                    lines=7,
                    max_lines=20,
                    placeholder="Dán cả chương sách cũng được — app tự cắt đoạn và ghép lại.",
                )
                text_info = gr.Markdown("")
                with gr.Row():
                    generate_btn = gr.Button("🎙️ Tạo giọng nói", variant="primary", size="lg", scale=3)
                    stop_btn = gr.Button("⏹ Dừng", variant="stop", size="lg", scale=1)

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
                    "- File WAV được lưu tự động vào thư mục dữ liệu của app.\n"
                    "- Text dài được cắt ở ranh giới đoạn/câu rồi ghép lại, nghỉ đúng nhịp; "
                    "để trống một dòng giữa các đoạn sẽ cho chỗ ngắt tự nhiên nhất."
                )

        # Ước lượng ngay khi gõ/dán: người dán cả chương sách cần biết trước là
        # việc này mất hàng giờ, chứ không phải bấm rồi ngồi đoán.
        text.change(fn=describe_workload, inputs=text, outputs=text_info, show_progress="hidden")

        run_event = generate_btn.click(
            fn=synthesize,
            inputs=[ref_audio, text, denoise],
            outputs=[output_audio, status],
            api_name="synthesize",
        )
        # Bài dài chạy hàng giờ — phải có đường thoát, không thì người dùng kẹt
        # với việc tắt cả app.
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

        # Người dùng chỉ được quyền dùng giọng mình có quyền sử dụng — nói rõ trong UI.
        # KHÔNG hứa watermark ở đây: bản đóng gói loại torch, mà watermark
        # (resemble-perth) lại cần torch, nên SDK để watermarker = None và audio
        # đi ra KHÔNG có dấu chìm. Hứa sai một thuộc tính an toàn còn tệ hơn là
        # không có nó, vì người dùng sẽ tin vào thứ không tồn tại.
        gr.Markdown(
            "---\n"
            "⚠️ **Chỉ nhân bản giọng của chính bạn, hoặc giọng bạn đã được chủ giọng "
            "đồng ý rõ ràng.**\n\n"
            "ℹ️ Bản app này **không** đóng dấu chìm (watermark) vào audio. Nếu cần "
            "watermark, hãy chạy từ source và cài thêm `pip install vieneu[watermark]` "
            "(kéo theo torch, nên không gộp vào bản đóng gói CPU gọn nhẹ này)."
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
    # queue() phải bật thì gr.Progress và `cancels` mới hoạt động. Gradio 4+ tạo
    # sẵn hàng đợi, nhưng gọi thẳng ở đây để hành vi không phụ thuộc mặc định của
    # từng phiên bản — app này chạy trên cả gradio 5 lẫn 6.
    demo.queue()
    preload_in_background()

    # inbrowser: xem should_open_browser() — mặc định bật ở bản đóng gói, tắt được
    # bằng VIENEU_OPEN_BROWSER=0.
    launch_kwargs: dict = {
        "server_name": os.environ.get("VIENEU_HOST", "127.0.0.1"),
        "server_port": int(os.environ.get("VIENEU_PORT", "7861")),
        "inbrowser": should_open_browser(),
        "quiet": False,
    }
    if _gradio_major() >= 6:
        launch_kwargs.update(_style_kwargs())
    else:
        launch_kwargs["show_api"] = False

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
