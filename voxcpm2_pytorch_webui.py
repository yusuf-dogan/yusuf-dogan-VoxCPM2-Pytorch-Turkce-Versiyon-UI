import gc
import re
import secrets
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
from voxcpm import VoxCPM


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODEL_DIR = MODELS / "VoxCPM2"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MODEL = None
LOCK = threading.Lock()

# Üretimi güvenli şekilde chunk sınırında durdurmak için.
# CUDA model.generate() çağrısını zorla öldürmek yerine mevcut chunk
# tamamlanınca sonraki chunk'a geçilmez.
CANCEL_EVENT = threading.Event()

# v4'te iyi sonuç veren güvenli VRAM optimizasyonu.
# Attention / RoPE / SDPA koduna dokunulmaz.
KV_MAX_LENGTH = 2048

LOW_VRAM_STATUS = "Henüz uygulanmadı."

# v1/v4 kalite yoluna yakın kal.
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass


def cuda_memory_text():
    if not torch.cuda.is_available():
        return "CUDA yok"

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    try:
        free, total = torch.cuda.mem_get_info()
        return (
            f"Allocated {allocated:.2f} GB | "
            f"Reserved {reserved:.2f} GB | "
            f"Peak {peak:.2f} GB | "
            f"CUDA free {free/1024**3:.2f}/{total/1024**3:.2f} GB"
        )
    except Exception:
        return (
            f"Allocated {allocated:.2f} GB | "
            f"Reserved {reserved:.2f} GB | "
            f"Peak {peak:.2f} GB"
        )


def runtime_dtype(tts):
    value = str(getattr(tts.config, "dtype", "bfloat16")).lower()

    if "bfloat16" in value or "bf16" in value:
        return torch.bfloat16

    if "float16" in value or "fp16" in value:
        return torch.float16

    return torch.float32


def apply_safe_kv_reduction(model):
    """
    Yalnız statik KV cache uzunluğunu 8192 -> 2048 yapar.
    Attention fonksiyonu, RoPE ve model ağırlıkları değişmez.
    """
    global LOW_VRAM_STATUS

    try:
        tts = model.tts_model
        dtype = runtime_dtype(tts)
        device = getattr(tts, "device", "cuda")

        before = torch.cuda.memory_allocated() / 1024**3

        tts.base_lm.kv_cache = None
        tts.residual_lm.kv_cache = None

        gc.collect()
        torch.cuda.empty_cache()

        tts.base_lm.setup_cache(
            1,
            KV_MAX_LENGTH,
            device,
            dtype,
        )

        tts.residual_lm.setup_cache(
            1,
            KV_MAX_LENGTH,
            device,
            dtype,
        )

        if hasattr(tts.config, "max_length"):
            tts.config.max_length = KV_MAX_LENGTH

        gc.collect()
        torch.cuda.empty_cache()

        after = torch.cuda.memory_allocated() / 1024**3
        saved = max(0.0, before - after)

        LOW_VRAM_STATUS = (
            f"SAFE KV reduction AKTİF | 8192 -> {KV_MAX_LENGTH} | "
            f"yaklaşık {saved:.2f} GB PyTorch VRAM geri kazanıldı. "
            "Attention kodu değişmedi."
        )

    except Exception as exc:
        LOW_VRAM_STATUS = (
            "SAFE KV reduction uygulanamadı; varsayılan KV ile devam: "
            f"{exc}"
        )


def get_model():
    global MODEL

    if MODEL is not None:
        return MODEL

    if not torch.cuda.is_available():
        raise gr.Error("CUDA bulunamadı.")

    print("VoxCPM2 yükleniyor...")

    if not MODEL_DIR.exists():
        raise gr.Error(
            "VoxCPM2 model dosyaları bulunamadı. "
            "Lütfen önce install.bat dosyasını çalıştır."
        )

    MODEL = VoxCPM.from_pretrained(
        str(MODEL_DIR),
        load_denoiser=False,
        optimize=False,
        device="cuda",
    )

    print("Model yüklendi:", cuda_memory_text())

    apply_safe_kv_reduction(MODEL)

    print(LOW_VRAM_STATUS)
    print("KV küçültme sonrası:", cuda_memory_text())

    torch.cuda.reset_peak_memory_stats()

    return MODEL



def request_cancel():
    """
    UI'den anında çalışır. Aktif model.generate() güvenlik nedeniyle
    zorla kesilmez; mevcut chunk biter bitmez üretim sonlandırılır.
    """
    CANCEL_EVENT.set()

    return (
        "İptal isteği alındı. Mevcut chunk tamamlanınca "
        "üretim durdurulacak; yeni chunk başlatılmayacak."
    )


def cancel_is_requested():
    return CANCEL_EVENT.is_set()


def cancelled_generation_result(logs, where=""):
    if where:
        logs.append(f"İPTAL EDİLDİ — {where}")
    else:
        logs.append("İPTAL EDİLDİ — kullanıcı isteği.")

    logs += [
        "",
        "Yeni chunk başlatılmadı.",
        "GPU modeli bellekte bırakıldı.",
        "İstersen daha sonra ayrıca 'Modeli GPU'dan Boşalt' kullanabilirsin.",
    ]

    return (
        None,
        None,
        None,
        "\n".join(logs),
    )


def unload_model():
    global MODEL

    MODEL = None
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return "Model GPU belleğinden boşaltıldı.\n" + cuda_memory_text()



def clean_pasted_text_for_tts(text):
    """
    Markdown/rich-text işaretlerini okunabilir düz metne çevirir.
    Bu fonksiyon üretim öncesindeki ikinci güvenlik katmanıdır.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")

    # Markdown görselleri ve linkler -> yalnız görünen metin.
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Başlık, alıntı, liste işaretleri.
    text = re.sub(r'(?m)^\s{0,3}#{1,6}\s*', '', text)
    text = re.sub(r'(?m)^\s*>\s?', '', text)
    text = re.sub(r'(?m)^\s*[-+*]\s+', '', text)
    text = re.sub(r'(?m)^\s*\d+[.)]\s+', '', text)

    # Bold / italic / strike / inline code.
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'___(.+?)___', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    text = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)_([^_\n]+)_(?!_)', r'\1', text)
    text = re.sub(r'`([^`\n]+)`', r'\1', text)

    # Markdown yatay çizgileri.
    text = re.sub(r'(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$', '', text)

    # Basit HTML artıkları.
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.I)
    text = re.sub(
        r'</?(?:p|div|h[1-6]|li|ul|ol|blockquote)[^>]*>',
        '\n',
        text,
        flags=re.I,
    )
    text = re.sub(r'<[^>]+>', '', text)

    # Satır ve paragraf düzenini KORU.
    # Yalnız satır içindeki gereksiz tab/çoklu boşlukları sadeleştir.
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = re.sub(r'[ \t]+', ' ', line).strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Kullanıcının mevcut boş satırlarını koru.
    # Sadece aşırı (4+) boş satırı en fazla 2 boş satıra indir.
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text.strip()


# Bu JS Gradio event sistemini kullanmaz.
# Doğrudan browser native "paste" olayını yakalar.
NATIVE_PASTE_HEAD = r"""
<script>
(function () {
    const TARGET_ID = "voxcpm_clean_paste_text";

    function stripMarkdown(value) {
        let t = value || "";

        t = t.replace(/\r\n?/g, "\n");

        // Images / links: görünen metni tut.
        t = t.replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1");
        t = t.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");

        // Headings / quotes / lists.
        t = t.replace(/^\s{0,3}#{1,6}\s*/gm, "");
        t = t.replace(/^\s*>\s?/gm, "");
        t = t.replace(/^\s*[-+*]\s+/gm, "");
        t = t.replace(/^\s*\d+[.)]\s+/gm, "");

        // Bold / italic / strike / inline-code.
        t = t.replace(/\*\*\*([\s\S]*?)\*\*\*/g, "$1");
        t = t.replace(/___([\s\S]*?)___/g, "$1");
        t = t.replace(/\*\*([\s\S]*?)\*\*/g, "$1");
        t = t.replace(/__([\s\S]*?)__/g, "$1");
        t = t.replace(/~~([\s\S]*?)~~/g, "$1");
        t = t.replace(/(^|[^\*])\*([^*\n]+)\*(?!\*)/g, "$1$2");
        t = t.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1$2");
        t = t.replace(/`([^`\n]+)`/g, "$1");

        // Horizontal rules.
        t = t.replace(/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/gm, "");

        // Satır/paragraf düzenini koru.
        // Her satır içindeki fazla boşluğu sadeleştir; boş satırlara dokunma.
        t = t
            .split("\n")
            .map((line) => line.replace(/[ \t]+/g, " ").trim())
            .join("\n");

        // Yalnız aşırı uzun boşluk bloklarını sınırla.
        t = t.replace(/\n{4,}/g, "\n\n\n");

        return t.trim();
    }

    function htmlToReadableText(html) {
        if (!html) return "";

        const box = document.createElement("div");
        box.innerHTML = html;

        box.querySelectorAll("br").forEach((br) => {
            br.replaceWith(document.createTextNode("\n"));
        });

        box.querySelectorAll(
            "p,div,h1,h2,h3,h4,h5,h6,blockquote"
        ).forEach((el) => {
            el.appendChild(document.createTextNode("\n\n"));
        });

        box.querySelectorAll("li").forEach((el) => {
            el.appendChild(document.createTextNode("\n"));
        });

        return stripMarkdown(
            box.innerText || box.textContent || ""
        );
    }

    function setNativeTextareaValue(textarea, value) {
        const descriptor = Object.getOwnPropertyDescriptor(
            HTMLTextAreaElement.prototype,
            "value"
        );

        if (descriptor && descriptor.set) {
            descriptor.set.call(textarea, value);
        } else {
            textarea.value = value;
        }

        textarea.dispatchEvent(
            new Event("input", {
                bubbles: true,
                composed: true
            })
        );

        textarea.dispatchEvent(
            new Event("change", {
                bubbles: true,
                composed: true
            })
        );
    }

    function bind() {
        const host = document.getElementById(TARGET_ID);
        if (!host) return false;

        const textarea = host.querySelector("textarea");
        if (!textarea) return false;

        if (textarea.dataset.nativeCleanPaste === "1") {
            return true;
        }

        textarea.dataset.nativeCleanPaste = "1";

        // CAPTURE=true: Gradio'nun kendi handler'ından önce yakala.
        textarea.addEventListener(
            "paste",
            function (event) {
                const clipboard = event.clipboardData;
                if (!clipboard) return;

                const html = clipboard.getData("text/html");
                const plain = clipboard.getData("text/plain");

                let cleaned = "";

                // ChatGPT panosunda text/plain genellikle Markdown satır ve
                // paragraf düzenini HTML'den daha doğru korur.
                if (plain) {
                    cleaned = stripMarkdown(plain);
                }

                // Düz metin yoksa HTML'i yedek olarak kullan.
                if (!cleaned && html) {
                    cleaned = htmlToReadableText(html);
                }

                if (!cleaned) return;

                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();

                const start =
                    textarea.selectionStart ?? textarea.value.length;
                const end =
                    textarea.selectionEnd ?? textarea.value.length;

                const before = textarea.value.slice(0, start);
                const after = textarea.value.slice(end);

                const next = before + cleaned + after;

                setNativeTextareaValue(textarea, next);

                const cursor = start + cleaned.length;

                requestAnimationFrame(() => {
                    textarea.focus();
                    textarea.setSelectionRange(cursor, cursor);
                });
            },
            true
        );

        console.log(
            "[VoxCPM-2 Pytorch Türkçe Versiyon] Native Clean Paste aktif."
        );

        return true;
    }

    function start() {
        if (bind()) return;

        const observer = new MutationObserver(function () {
            if (bind()) {
                observer.disconnect();
            }
        });

        observer.observe(
            document.documentElement,
            {
                childList: true,
                subtree: true
            }
        );

        // Gradio sayfası geç render edilirse ek güvenlik.
        let tries = 0;
        const timer = setInterval(function () {
            tries += 1;

            if (bind() || tries >= 60) {
                clearInterval(timer);
            }
        }, 500);
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            start,
            { once: true }
        );
    } else {
        start();
    }
})();
</script>
"""


def split_long_piece(piece, max_chars):
    """Split only when a single sentence exceeds the hard maximum."""
    piece = piece.strip()
    if len(piece) <= max_chars:
        return [piece]

    clauses = re.split(r'(?<=[,;:])\s+', piece)
    parts, cur = [], ""
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        candidate = clause if not cur else cur + " " + clause
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                parts.append(cur)
            cur = clause
    if cur:
        parts.append(cur)

    final = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
            continue
        words, cur = part.split(), ""
        for word in words:
            candidate = word if not cur else cur + " " + word
            if len(candidate) <= max_chars:
                cur = candidate
            else:
                if cur:
                    final.append(cur)
                cur = word
        if cur:
            final.append(cur)
    return final


def split_text(text, target_chars, hard_max_chars=220):
    """Sentence-first chunker for more stable long-form voice."""
    target_chars = max(120, int(target_chars))
    hard_max_chars = max(target_chars, int(hard_max_chars))
    sentences = re.split(r'(?<=[.!?…])\s+|\n+', text.strip())
    units = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= hard_max_chars:
            units.append(sentence)
        else:
            units.extend(split_long_piece(sentence, hard_max_chars))

    chunks, cur = [], ""
    for unit in units:
        if not cur:
            cur = unit
            continue
        candidate = cur + " " + unit
        if len(candidate) <= hard_max_chars:
            cur = candidate
        else:
            chunks.append(cur)
            cur = unit
    if cur:
        chunks.append(cur)

    # Merge very short chunks only when safe.
    balanced = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        if len(current) < 75 and i + 1 < len(chunks):
            candidate = current + " " + chunks[i + 1]
            if len(candidate) <= hard_max_chars:
                balanced.append(candidate)
                i += 2
                continue
        if balanced and len(current) < 75 and len(balanced[-1]) + 1 + len(current) <= hard_max_chars:
            balanced[-1] += " " + current
        else:
            balanced.append(current)
        i += 1
    return balanced


def dynamic_min_len_for_chunk(chunk):
    # Matches the last good profile: ~160 chars -> min_len 20.
    return max(6, min(24, round(len(chunk.strip()) / 8)))


def audio_quality_stats(wav):
    if wav.size == 0:
        return {"peak": 0.0, "rms": 0.0, "clip_ratio": 0.0}
    abs_wav = np.abs(wav)
    peak = float(np.max(abs_wav))
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
    clip_ratio = float(np.mean(abs_wav >= 1.0))
    return {"peak": peak, "rms": rms, "clip_ratio": clip_ratio}


def apply_peak_guard(wav, ceiling=0.98):
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak <= ceiling or peak <= 0.0:
        return wav, False, 1.0
    gain = float(ceiling / peak)
    return (wav * gain).astype(np.float32, copy=False), True, gain



def apply_micro_declick_guard(
    wav,
    residual_threshold=0.45,
    neighbor_threshold=0.08,
    min_abs_sample=0.30,
):
    """
    Çok muhafazakâr tek-sample dijital spike düzeltmesi.

    Sadece:
    - merkez örnek iki komşusunun ortalamasından çok uzaksa,
    - iki komşu birbirine oldukça yakınsa,
    - merkez örnek yeterince yüksek genlikliyse,
    - aday örnek tek başına/izole ise

    müdahale eder.

    Normal konuşma transientlerine, sibilanslara veya genel frekans
    içeriğine filtre uygulanmaz.
    """
    if wav.size < 5:
        return wav, 0

    y = wav.astype(np.float32, copy=True)

    left = y[:-2]
    center = y[1:-1]
    right = y[2:]

    neighbor_mean = (left + right) * 0.5
    residual = np.abs(center - neighbor_mean)
    neighbor_gap = np.abs(left - right)

    candidates = (
        (residual >= float(residual_threshold))
        & (neighbor_gap <= float(neighbor_threshold))
        & (np.abs(center) >= float(min_abs_sample))
    )

    # Yalnız izole tek-sample adayları düzelt.
    isolated = candidates.copy()

    if isolated.size >= 3:
        adjacent = np.zeros_like(isolated, dtype=bool)
        adjacent[1:] |= candidates[:-1]
        adjacent[:-1] |= candidates[1:]
        isolated &= ~adjacent

    indices = np.flatnonzero(isolated) + 1

    if indices.size == 0:
        return wav, 0

    y[indices] = (
        y[indices - 1] + y[indices + 1]
    ) * 0.5

    return y, int(indices.size)


def apply_edge_fade(wav, sample_rate, fade_ms=12):
    if wav.size == 0 or fade_ms <= 0:
        return wav
    n = int(sample_rate * fade_ms / 1000.0)
    if n <= 1 or wav.size < n * 2:
        return wav
    out = wav.copy()
    out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def apply_optional_style(text, style):
    """
    Checkbox yok.
    Alan boşsa hiçbir şey eklenmez.
    Kullanıcı stil yazarsa yalnız o zaman metnin başına eklenir.
    """
    style = (style or "").strip()

    if not style:
        return text

    if not (
        style.startswith("(")
        and style.endswith(")")
    ):
        style = f"({style})"

    return f"{style} {text}"


def to_numpy(wav):
    if isinstance(wav, np.ndarray):
        return wav.astype(
            np.float32,
            copy=False,
        ).reshape(-1)

    if torch.is_tensor(wav):
        return (
            wav.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
            .reshape(-1)
        )

    return np.asarray(
        wav,
        dtype=np.float32,
    ).reshape(-1)


def make_mp3(
    wav_path,
    mp3_path,
    speed,
    bitrate,
):
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(wav_path),
    ]

    # 1.00'da hiçbir atempo DSP'si uygulanmaz.
    if abs(float(speed) - 1.0) > 0.001:
        cmd += [
            "-filter:a",
            f"atempo={float(speed):.4f}",
        ]

    cmd += [
        "-c:a", "libmp3lame",
        "-b:a", str(bitrate),
        str(mp3_path),
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr
            or "MP3 dönüştürme hatası"
        )


def generate(
    text,
    reference_audio,
    style,
    chunk_chars,
    pause_ms,
    cfg,
    timesteps,
    max_len,
    auto_seed,
    seed,
    final_speed,
    bitrate,
    progress=gr.Progress(),
):
    with LOCK:
        # Önceki bir iptal isteğinin sonraki üretimi etkilememesi için.
        CANCEL_EVENT.clear()

        text = clean_pasted_text_for_tts(text)

        if not text:
            raise gr.Error("Metin gir.")

        if not reference_audio:
            raise gr.Error(
                "Referans MP3/WAV seç."
            )

        chunks = split_text(
            text,
            int(chunk_chars),
            hard_max_chars=220,
        )

        # Her üretim için TEK bir run seed seçilir.
        # auto_seed açıksa UI'deki sabit Seed alanı kullanılmaz.
        if bool(auto_seed):
            run_seed = secrets.randbelow(2_147_483_646) + 1
            seed_mode_text = "OTOMATİK — her üretimde yeni"
        else:
            run_seed = int(seed)
            seed_mode_text = "SABİT — Seed alanı kullanılıyor"

        if not chunks:
            raise gr.Error(
                "Chunk oluşturulamadı."
            )

        if cancel_is_requested():
            return (
                None,
                None,
                None,
                "İPTAL EDİLDİ — model yüklenmeden önce kullanıcı isteği.",
            )

        model = get_model()

        if cancel_is_requested():
            return (
                None,
                None,
                None,
                "İPTAL EDİLDİ — model hazırlandıktan sonra kullanıcı isteği.",
            )

        sr = int(
            model.tts_model.sample_rate
        )

        logs = [
            "VoxCPM-2 Pytorch Türkçe Versiyon",
            "",
            LOW_VRAM_STATUS,
            "FAST attention patch: KAPALI",
            "CPU prompt cache: KAPALI",
            "Bad-case retry: KAPALI (sabit)",
            "Stil checkbox: YOK",
            (
                "Stil: BOŞ — kullanılmıyor"
                if not (style or "").strip()
                else f"Stil: {style.strip()}"
            ),
            "Public model.generate(): AKTİF",
            "Denoiser: KAPALI",
            "Normalize: KAPALI",
            "",
            f"Toplam karakter: {len(text)}",
            f"Chunk sayısı: {len(chunks)}",
            f"Chunk hedefi: {int(chunk_chars)}",
            f"CFG: {float(cfg):.2f}",
            f"Timesteps: {int(timesteps)}",
            "Min Len: OTOMATİK (chunk uzunluğuna göre)",
            f"Max Len: {int(max_len)}",
            "Peak Guard: AÇIK (peak > 0.98 ise)",
            "Micro De-click Guard: AÇIK (yalnız izole dijital spike)",
            "Edge Fade: 12 ms",
            "WAV ara/final formatı: PCM_24",
            f"Seed modu: {seed_mode_text}",
            f"Run Seed: {int(run_seed)}",
            (
                f"Sabit Seed alanı: {int(seed)} (otomatik modda kullanılmaz)"
                if bool(auto_seed)
                else f"Sabit Seed alanı: {int(seed)}"
            ),
            f"Son konuşma hızı: {float(final_speed):.2f}",
            f"MP3 bitrate: {bitrate}",
            "",
            "Başlangıç GPU:",
            cuda_memory_text(),
            "",
        ]

        wavs = []
        speed_history = []
        suspect_chunks = []
        total_started = time.perf_counter()

        for i, chunk in enumerate(
            chunks,
            start=1,
        ):
            if cancel_is_requested():
                return cancelled_generation_result(
                    logs,
                    where=f"Chunk {i} başlamadan önce",
                )

            progress(
                (i - 1) / len(chunks),
                desc=f"Chunk {i}/{len(chunks)}",
            )

            final_text = apply_optional_style(
                chunk,
                style,
            )

            # Bütün chunk'larda aynı RUN seed kullanılır.
            # Böylece tek üretim içinde ses karakteri daha tutarlı kalır;
            # bir sonraki üretimde ise auto_seed açıksa tamamen yeni run seed gelir.
            torch.manual_seed(int(run_seed))
            torch.cuda.manual_seed_all(
                int(run_seed)
            )

            torch.cuda.reset_peak_memory_stats()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            dynamic_min_len = dynamic_min_len_for_chunk(chunk)

            started = time.perf_counter()

            try:
                with torch.inference_mode():
                    wav = model.generate(
                        text=final_text,
                        reference_wav_path=reference_audio,
                        cfg_value=float(cfg),
                        inference_timesteps=int(timesteps),
                        min_len=int(dynamic_min_len),
                        max_len=int(max_len),
                        normalize=False,

                        # Kullanıcının en iyi sonucu aldığı davranış:
                        retry_badcase=False,
                    )

            except torch.cuda.OutOfMemoryError as exc:
                gc.collect()
                torch.cuda.empty_cache()

                raise gr.Error(
                    f"Chunk {i} sırasında CUDA VRAM yetmedi.\n\n{exc}"
                )

            except Exception as exc:
                raise gr.Error(
                    f"Chunk {i} üretilemedi:\n\n{exc}"
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - started
            )

            # Kullanıcı üretim sırasında iptal ettiyse mevcut chunk GPU'da
            # güvenli şekilde tamamlandı; fakat sonuç birleştirmeye alınmaz
            # ve yeni chunk başlatılmaz.
            if cancel_is_requested():
                logs += [
                    f"[{i}/{len(chunks)}] mevcut chunk güvenli şekilde tamamlandı.",
                    "İptal isteği nedeniyle chunk çıktısı final sese eklenmedi.",
                ]

                return cancelled_generation_result(
                    logs,
                    where=f"Chunk {i} sonrasında",
                )

            wav = to_numpy(wav)

            wav = np.nan_to_num(
                wav,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if wav.size == 0:
                raise gr.Error(
                    f"Chunk {i} boş ses üretti."
                )

            quality_before = audio_quality_stats(wav)

            wav, declick_repaired = apply_micro_declick_guard(wav)

            wav, peak_guard_applied, peak_gain = apply_peak_guard(
                wav,
                ceiling=0.98,
            )

            wav = apply_edge_fade(
                wav,
                sample_rate=sr,
                fade_ms=12,
            )

            quality_after = audio_quality_stats(wav)

            wavs.append(wav)
            duration = len(wav) / sr
            chars_per_sec = len(chunk) / duration if duration > 0 else 0.0
            vram_peak = torch.cuda.max_memory_allocated() / 1024**3
            rtf = elapsed / duration if duration > 0 else 0.0

            warning = ""
            if speed_history:
                median_speed = float(np.median(speed_history))
                if median_speed > 0:
                    ratio = chars_per_sec / median_speed
                    if ratio > 1.45:
                        warning = "⚠ ŞÜPHELİ: Bu chunk önceki medyandan belirgin hızlı."
                    elif ratio < 0.65:
                        warning = "⚠ ŞÜPHELİ: Bu chunk önceki medyandan belirgin yavaş."
            speed_history.append(chars_per_sec)
            if warning:
                suspect_chunks.append(i)

            logs += [
                f"[{i}/{len(chunks)}] {len(chunk)} karakter | Auto Min Len: {dynamic_min_len}",
                f"→ Ses: {duration:.2f} sn | Yoğunluk: {chars_per_sec:.2f} karakter/sn",
                f"→ Üretim: {elapsed:.2f} sn | RTF: {rtf:.2f}x",
                f"→ Audio Peak: {quality_before['peak']:.4f} | RMS: {quality_before['rms']:.4f}",
                f"→ Clipping oranı: %{quality_before['clip_ratio'] * 100.0:.4f}",
                (
                    f"→ Micro De-click: {declick_repaired} izole spike düzeltildi"
                    if declick_repaired
                    else "→ Micro De-click: müdahale gerekmedi"
                ),
                (
                    f"→ Peak Guard uygulandı | gain={peak_gain:.4f} | son peak={quality_after['peak']:.4f}"
                    if peak_guard_applied
                    else "→ Peak Guard gerekmedi"
                ),
                f"→ Peak PyTorch VRAM: {vram_peak:.2f} GB",
            ]
            if warning:
                logs.append(warning)
            logs += [
                chunk[:150] + ("..." if len(chunk) > 150 else ""),
                "",
            ]

            # empty_cache bilerek yok:
            # allocator sonraki chunk'ta belleği tekrar kullanır.

        if cancel_is_requested():
            return cancelled_generation_result(
                logs,
                where="chunk üretimleri tamamlandıktan sonra",
            )

        pause = np.zeros(
            int(
                sr
                * max(0, int(pause_ms))
                / 1000
            ),
            dtype=np.float32,
        )

        parts = []

        for i, wav in enumerate(wavs):
            parts.append(wav)

            if (
                i < len(wavs) - 1
                and pause.size
            ):
                parts.append(pause)

        merged = np.concatenate(
            parts
        ).astype(
            np.float32,
            copy=False,
        )

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        wav_path = (
            OUT
            / f"voxcpm2_turkce_{stamp}.wav"
        )

        mp3_path = (
            OUT
            / f"voxcpm2_turkce_{stamp}.mp3"
        )

        # Ham WAV hiçbir ek DSP görmez.
        # PCM_24, MP3 hızlandırma/encode öncesinde daha yüksek ara hassasiyet
        # sağlar; dosya boyutu PCM_16'dan biraz daha büyüktür.
        sf.write(
            wav_path,
            merged,
            sr,
            subtype="PCM_24",
        )

        # Son hız yalnız MP3 üzerinde uygulanır.
        make_mp3(
            wav_path,
            mp3_path,
            final_speed,
            bitrate,
        )

        total_elapsed = (
            time.perf_counter()
            - total_started
        )

        raw_duration = (
            len(merged) / sr
        )

        final_duration = (
            raw_duration / float(final_speed)
        )

        logs += [
            "TAMAMLANDI",
            f"Toplam üretim: {total_elapsed/60:.2f} dk",
            f"Ham WAV süresi: {raw_duration:.2f} sn",
            f"MP3 yaklaşık süre: {final_duration:.2f} sn",
            f"WAV: {wav_path}",
            f"MP3: {mp3_path}",
            "",
            "Bitiş GPU:",
            cuda_memory_text(),
            "",
            (
                "Stability Guard: Şüpheli chunk yok."
                if not suspect_chunks
                else "Stability Guard: Şüpheli chunk(lar): " + ", ".join(str(x) for x in suspect_chunks)
            ),
        ]

        progress(
            1.0,
            desc="Tamamlandı",
        )

        # MP3 player'a normal dosya yolu gönderilir.
        return (
            str(mp3_path),
            str(wav_path),
            str(mp3_path),
            "\n".join(logs),
        )



def find_available_port(start=7861, end=7870):
    """7861 doluysa 7862-7870 aralığında ilk boş portu seçer."""
    for port in range(int(start), int(end) + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port

    raise RuntimeError(
        "7861-7870 aralığında boş port bulunamadı. "
        "Bu portları kullanan uygulamalardan birini kapatıp tekrar deneyin."
    )


with gr.Blocks(
    title="VoxCPM-2 Pytorch Türkçe Versiyon - UI",
) as demo:

    gr.Markdown(
        """
# VoxCPM-2 Pytorch Türkçe Versiyon
"""
    )

    with gr.Row():

        with gr.Column(scale=2):

            text = gr.Textbox(
                label="Seslendirilecek metin",
                lines=16,
                placeholder=(
                    "Türkçe metni buraya yapıştır..."
                ),
                elem_id="voxcpm_clean_paste_text",
            )

            ref = gr.Audio(
                label="Referans MP3 / WAV",
                type="filepath",
                sources=["upload"],
            )

            style = gr.Textbox(
                label=(
                    "Stil — boş bırakırsan hiçbir stil komutu kullanılmaz"
                ),
                value="",
                lines=3,
                placeholder=(
                    "İsteğe bağlı. Son iyi sonuç için boş bırak."
                ),
            )

        with gr.Column(scale=1):

            chunk = gr.Slider(
                140,
                220,
                value=200,
                step=10,
                label="Chunk hedefi — karakter",
            )

            pause = gr.Slider(
                0,
                300,
                value=140,
                step=20,
                label="Chunk arası duraklama — ms",
            )

            cfg = gr.Slider(
                1.0,
                2.5,
                value=2.0,
                step=0.1,
                label="CFG",
            )

            steps = gr.Slider(
                6,
                16,
                value=12,
                step=1,
                label="Inference Timesteps",
            )

            gr.Markdown(
                "**Min Len:** Otomatik — her chunk uzunluğuna göre ayarlanır."
            )

            max_len = gr.Slider(
                120,
                240,
                value=180,
                step=10,
                label="Max Len",
            )

            auto_seed = gr.Checkbox(
                value=True,
                label="Her üretimde yeni Seed",
            )

            seed = gr.Number(
                value=42,
                precision=0,
                label="Seed — otomatik Seed kapalıysa kullanılır",
            )

            final_speed = gr.Slider(
                0.85,
                1.10,
                value=1.00,
                step=0.01,
                label=(
                    "Son konuşma hızı — 1.00 normal"
                ),
            )

            bitrate = gr.Dropdown(
                [
                    "128k",
                    "160k",
                    "192k",
                    "256k",
                    "320k",
                ],
                value="192k",
                label="MP3 bitrate",
            )


    with gr.Row():

        btn = gr.Button(
            "Tek Ses Oluştur",
            variant="primary",
        )

        cancel_btn = gr.Button(
            "İşlemi İptal Et",
            variant="stop",
        )

        unload = gr.Button(
            "Modeli GPU'dan Boşalt"
        )

    gr.Markdown(
        "### Oluşturulan MP3"
    )

    mp3_player = gr.Audio(
        label="MP3 oynatıcı",
        type="filepath",
        autoplay=False,
    )

    gr.Markdown(
        "### Dosyalar"
    )

    with gr.Row():

        wav_file = gr.File(
            label="WAV indir"
        )

        mp3_file = gr.File(
            label="MP3 indir"
        )

    log = gr.Textbox(
        label="Üretim günlüğü / hız / VRAM",
        lines=22,
    )

    status = gr.Textbox(
        label="Model durumu",
        value="İlk üretimde model yüklenecek.",
        interactive=False,
    )

    btn.click(
        generate,
        inputs=[
            text,
            ref,
            style,
            chunk,
            pause,
            cfg,
            steps,
            max_len,
            auto_seed,
            seed,
            final_speed,
            bitrate,
        ],
        outputs=[
            mp3_player,
            wav_file,
            mp3_file,
            log,
        ],
        concurrency_limit=1,
    )

    cancel_btn.click(
        request_cancel,
        outputs=status,
        queue=False,
    )

    unload.click(
        unload_model,
        outputs=status,
        concurrency_limit=1,
    )


if __name__ == "__main__":

    server_port = find_available_port(7861, 7870)

    print(
        f"VoxCPM-2 Pytorch Türkçe Versiyon - UI: "
        f"http://127.0.0.1:{server_port}"
    )

    demo.queue(
        default_concurrency_limit=1
    ).launch(
        server_name="127.0.0.1",
        server_port=server_port,
        inbrowser=True,
        share=False,
        head=NATIVE_PASTE_HEAD,
    )
