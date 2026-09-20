# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec cho VieNeu Voice Clone (Windows, CPU/ONNX).

Build:
    cd packaging
    pyinstaller vieneu_clone.spec --noconfirm

Kết quả: packaging/dist/VieNeuVoiceClone/VieNeuVoiceClone.exe (onedir).

Ghi chú về onedir vs onefile: giữ onedir là cố ý. Bundle này ~1 GB vì
onnxruntime + gradio; bản onefile sẽ phải giải nén toàn bộ ra thư mục tạm ở
MỖI lần mở app, làm thời gian khởi động tăng vọt. Muốn người dùng chỉ thấy
một file duy nhất thì bọc thư mục onedir bằng installer (xem build_windows.ps1).
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()
PROJECT_ROOT = SPEC_DIR.parent

datas = []
binaries = []
hiddenimports = []

# ── Gradio ────────────────────────────────────────────────────────────────────
# Gradio phục vụ frontend từ các file tĩnh nằm trong package, và runtime của nó
# gọi inspect.getsource() lên chính code Python của mình → phải gom cả data lẫn
# source (xem MODULE_COLLECTION_MODE ngay bên dưới).
for pkg in ("gradio", "gradio_client", "safehttpx", "groovy"):
    datas += collect_data_files(pkg)
    hiddenimports += collect_submodules(pkg)

# Gradio đọc source của chính nó lúc chạy (inspect.getsource) → phải giữ dạng .py
# trên đĩa thay vì nén vào PYZ. Biến này BẮT BUỘC truyền vào Analysis(); đặt nó
# làm biến toàn cục trong file spec không có tác dụng gì, vì PyInstaller exec
# spec rồi không đọc lại tên đó.
MODULE_COLLECTION_MODE = {
    "gradio": "py",
    "gradio_client": "py",
}

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
# Các .dll của provider không được PyInstaller phát hiện qua import tĩnh.
binaries += collect_dynamic_libs("onnxruntime")
datas += collect_data_files("onnxruntime")
hiddenimports += ["onnxruntime", "onnxruntime.capi", "onnxruntime.capi._pybind_state"]

# ── Phonemizer / codec / audio I/O ────────────────────────────────────────────
for pkg in ("sea_g2p", "kaldi_native_fbank", "tokenizers", "soxr", "soundfile"):
    try:
        datas += collect_data_files(pkg)
        binaries += collect_dynamic_libs(pkg)
        hiddenimports += collect_submodules(pkg)
    except Exception:
        # Package thuần Python không có data/lib — bỏ qua, import tĩnh là đủ.
        hiddenimports.append(pkg)

# ── Assets của chính VieNeu (voices JSON, samples) ────────────────────────────
datas += collect_data_files("vieneu")
hiddenimports += collect_submodules("vieneu")

# librosa/numba nạp submodule động; huggingface_hub cần cho lần tải model đầu.
hiddenimports += collect_submodules("huggingface_hub")
hiddenimports += ["librosa", "soundfile", "numpy", "yaml"]

# ── Loại bỏ những thứ app CPU không dùng ──────────────────────────────────────
# App ép backend="onnx", nên toàn bộ nhánh torch là trọng lượng chết (~2 GB).
excludes = [
    "torch",
    "torchaudio",
    "transformers",
    "neucodec",
    "peft",
    "accelerate",
    "datasets",
    "matplotlib",
    "tkinter",
    "pytest",
    "IPython",
    "notebook",
]


a = Analysis(
    [str(PROJECT_ROOT / "apps" / "voice_clone_app.py")],
    pathex=[str(PROJECT_ROOT), str(PROJECT_ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
    module_collection_mode=MODULE_COLLECTION_MODE,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VieNeuVoiceClone",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # console=True: lần chạy đầu tải vài trăm MB model — người dùng cần thấy
    # tiến trình, và log lỗi cũng hiện ra thay vì app im lặng tắt.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VieNeuVoiceClone",
)
