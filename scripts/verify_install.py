from pathlib import Path
import importlib.metadata as metadata
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "VoxCPM2"
FFMPEG = ROOT / "runtime" / "ffmpeg" / "bin" / "ffmpeg.exe"

print("=" * 70)
print("VoxCPM-2 Pytorch Türkçe Versiyon - UI | Kurulum Doğrulama")
print("=" * 70)

def version(name):
    try:
        return metadata.version(name)
    except Exception:
        return "bulunamadı"

print("Python         :", sys.version.split()[0])
print("VoxCPM         :", version("voxcpm"))
print("TorchCodec     :", version("torchcodec"))
print("Gradio         :", version("gradio"))
print("NumPy          :", version("numpy"))
print("imageio-ffmpeg :", version("imageio-ffmpeg"))

import torch

print("PyTorch        :", torch.__version__)
print("Torch CUDA     :", torch.version.cuda)
print("CUDA mevcut    :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA kullanılamıyor. Bu UI NVIDIA CUDA GPU gerektirir. "
        "NVIDIA sürücünüzü güncelleyip bilgisayarı yeniden başlatın."
    )

print("GPU            :", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print("GPU VRAM       :", f"{props.total_memory / 1024**3:.2f} GB")

if not FFMPEG.exists():
    raise RuntimeError(f"Bundled FFmpeg bulunamadı: {FFMPEG}")

proc = subprocess.run(
    [str(FFMPEG), "-version"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)
if proc.returncode != 0:
    raise RuntimeError("FFmpeg çalıştırılamadı.")

lines = (proc.stdout or "").splitlines()
print("FFmpeg         :", lines[0] if lines else "çalışıyor")

try:
    import torchcodec  # noqa: F401
    print("TorchCodec DLL :", "OK")
except Exception as exc:
    raise RuntimeError(
        "TorchCodec yüklenemedi. FFmpeg Shared/PATH kontrolü başarısız.\\n"
        + str(exc)
    )

required_model_files = {
    "model.safetensors": 4_000_000_000,
    "audiovae.pth": 300_000_000,
    "config.json": 100,
    "tokenizer.json": 100_000,
}

for filename, min_size in required_model_files.items():
    path = MODEL_DIR / filename
    if not path.exists():
        raise RuntimeError(f"Model dosyası eksik: {path}")
    if path.stat().st_size < min_size:
        raise RuntimeError(
            f"Model dosyası beklenenden küçük/eksik olabilir: {path}"
        )

from voxcpm import VoxCPM  # noqa: F401
import gradio  # noqa: F401
import imageio_ffmpeg  # noqa: F401
import soundfile  # noqa: F401
import numpy  # noqa: F401

print("Model klasörü  : OK")
print("Python importları: OK")
print("=" * 70)
print("KURULUM BAŞARILI")
print("=" * 70)
