# VoxCPM-2 Pytorch Türkçe Versiyon - UI

Windows üzerinde **VoxCPM2 ile Türkçe metin seslendirme ve referans sesten ses klonlama** için hazırlanmış, sade ve tek tıkla kurulabilen bir Gradio arayüzüdür.

> **Önemli:** Bu proje OpenBMB'nin resmî VoxCPM repository'si değildir. VoxCPM/VoxCPM2 modeli ve Python kütüphanesi OpenBMB tarafından geliştirilmiştir. Bu repository Windows ve Türkçe kullanım için hazırlanmış bağımsız bir UI ve kurulum katmanıdır.

## Hızlı kurulum

1. GitHub'da **Code → Download ZIP** ile repository'yi indirin.
2. ZIP'i bir klasöre çıkarın.
3. `install.bat` dosyasına çift tıklayın.
4. Kurulumun bitmesini bekleyin.
5. Kurulum başarılı olduğunda UI otomatik açılır.
6. Sonraki kullanımlarda yalnızca `run.bat` dosyasına çift tıklayın.

**Bilgisayarda uygun Python 3.11 zaten varsa installer onu kullanır. Python 3.11 yoksa Python 3.11.9 otomatik olarak kurulur.**

Ayrıntılı sıfırdan kurulum ve kullanım anlatımı:

**[KURULUM_VE_KULLANIM.md](KURULUM_VE_KULLANIM.md)**

## Sistem gereksinimleri

- Windows 10/11 64-bit
- NVIDIA CUDA destekli ekran kartı
- Güncel NVIDIA ekran kartı sürücüsü
- İlk kurulum için internet bağlantısı
- Yaklaşık 20 GB veya daha fazla boş disk alanı tavsiye edilir
- 6 GB VRAM bazı sistemlerde bu UI'deki SAFE KV optimizasyonuyla çalışabilir; VoxCPM2 upstream dokümantasyonu yaklaşık 8 GB VRAM belirtmektedir.

CUDA Toolkit'i ayrıca kurmak normalde gerekmez; installer CUDA 12.8 çalışma zamanı içeren PyTorch wheel'lerini kurar. NVIDIA sürücüsünün güncel olması gerekir.

## `install.bat` neleri yapar?

- Mevcut 64-bit Python 3.11'i bulur; yoksa Python 3.11.9 indirip kurar.
- Repository içinde `.venv` oluşturur.
- FFmpeg 7.1.1 Full Shared indirir ve SHA256 ile doğrular.
- PyTorch 2.8.0 + CUDA 12.8, torchvision 0.23.0 ve torchaudio 2.8.0 kurar.
- TorchCodec 0.7.0 kurar.
- VoxCPM 2.0.3 ve gerekli bağımlılıkları kurar.
- VoxCPM2 modelini Hugging Face'ten `models/VoxCPM2` içine indirir.
- CUDA, GPU, FFmpeg, TorchCodec, VoxCPM ve model dosyalarını kontrol eder.
- UI'yi otomatik başlatır.

Model yaklaşık 5 GB'tır. İndirme yarıda kalırsa `install.bat` yeniden çalıştırılabilir.

## Türkçe UI varsayılanları

| Ayar | Varsayılan |
|---|---:|
| Chunk hedefi | 200 karakter |
| Chunk arası duraklama | 140 ms |
| CFG | 2.0 |
| Inference Timesteps | 12 |
| Min Len | Otomatik |
| Max Len | 180 |
| Her üretimde yeni Seed | Açık |
| Seed | 42 — yalnız otomatik Seed kapalıysa |
| Son konuşma hızı | 1.00 |
| MP3 bitrate | 192k |
| Denoiser | Kapalı |
| Normalize | Kapalı |
| Bad-case retry | Kapalı |
| SAFE KV | 2048 |
| WAV | PCM 24-bit |

## Kısa kullanım

1. Türkçe metni **Seslendirilecek metin** alanına yapıştırın.
2. **Referans MP3 / WAV** alanından kullanma izniniz bulunan referans sesi seçin.
3. Stil kullanmak istemiyorsanız **Stil** alanını boş bırakın.
4. **Tek Ses Oluştur** butonuna basın.
5. Üretim tamamlandığında MP3 player ile WAV/MP3 indirme alanları görünür.
6. Çıktılar ayrıca `outputs/` klasörüne yazılır.

`İşlemi İptal Et` aktif CUDA üretimini zorla sonlandırmaz; mevcut chunk bittikten sonra yeni chunk başlatılmadan güvenli biçimde durur.

## Repository yapısı

```text
VoxCPM2-Pytorch-Turkce-Versiyon-UI/
├─ install.bat
├─ run.bat
├─ voxcpm2_pytorch_webui.py
├─ requirements.txt
├─ requirements-torch-cu128.txt
├─ README.md
├─ KURULUM_VE_KULLANIM.md
├─ LICENSE
├─ NOTICE.md
├─ CHANGELOG.md
├─ CONTRIBUTING.md
├─ VERSION
├─ .gitignore
├─ .gitattributes
├─ scripts/
│  ├─ install.ps1
│  ├─ download_model.py
│  └─ verify_install.py
├─ models/
│  └─ .gitkeep
├─ outputs/
│  └─ .gitkeep
└─ logs/
   └─ .gitkeep
```

`.venv`, `runtime`, model ağırlıkları, loglar ve üretilen sesler GitHub'a commit edilmez.

## Upstream / kaynaklar

- OpenBMB VoxCPM: https://github.com/OpenBMB/VoxCPM
- VoxCPM2 model: https://huggingface.co/openbmb/VoxCPM2
- PyTorch: https://pytorch.org/
- TorchCodec: https://github.com/meta-pytorch/torchcodec
- FFmpeg: https://ffmpeg.org/
- Python: https://www.python.org/

## Lisans

Bu repository'nin UI/kurulum katmanı `LICENSE` dosyasındaki MIT lisansı ile sunulur.

VoxCPM/VoxCPM2, PyTorch, TorchCodec, FFmpeg, Python ve diğer üçüncü taraf bileşenler kendi lisanslarına tabidir. Ayrıntı için [NOTICE.md](NOTICE.md) dosyasına bakın.

Ses klonlama özelliklerini yalnızca kullanma izniniz bulunan seslerle kullanın.

## Windows terminal kodlaması

Kurulum scriptleri Türkçe karakterler için UTF-8 uyumlu hazırlanmıştır. PowerShell scripti UTF-8 BOM ile kaydedilir ve konsol giriş/çıkış kodlaması açıkça UTF-8'e ayarlanır.
