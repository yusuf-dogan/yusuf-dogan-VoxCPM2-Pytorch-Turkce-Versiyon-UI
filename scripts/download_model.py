from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "VoxCPM2"
MODEL_REVISION = "32279effe8c19989596f05d353d1447f51d9e915"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

required = [
    MODEL_DIR / "model.safetensors",
    MODEL_DIR / "audiovae.pth",
    MODEL_DIR / "config.json",
    MODEL_DIR / "tokenizer.json",
]

if all(path.exists() for path in required):
    print("VoxCPM2 model dosyaları zaten mevcut.")
else:
    print()
    print("VoxCPM2 modeli indiriliyor.")
    print("İndirme yaklaşık 5 GB'tır ve internet hızına göre uzun sürebilir.")
    print("İndirme kesilirse install.bat dosyasını yeniden çalıştırabilirsiniz.")
    print()

    snapshot_download(
        repo_id="openbmb/VoxCPM2",
        revision=MODEL_REVISION,
        local_dir=str(MODEL_DIR),
    )

missing = [str(path) for path in required if not path.exists()]
if missing:
    raise RuntimeError(
        "Model indirme tamamlanamadı. Eksik dosyalar:\n- "
        + "\n- ".join(missing)
    )

(ROOT / "models" / ".model_revision").write_text(
    MODEL_REVISION + "\n",
    encoding="utf-8",
)

print("VoxCPM2 modeli hazır:", MODEL_DIR)
