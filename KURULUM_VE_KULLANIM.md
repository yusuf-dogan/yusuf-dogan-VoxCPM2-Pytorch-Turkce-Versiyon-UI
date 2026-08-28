# VoxCPM-2 Pytorch Türkçe Versiyon - UI
# Sıfırdan Kurulum ve Kullanım Kılavuzu

Bu belge, daha önce Python, `.venv`, PyTorch veya yapay zekâ modeli kurmamış bir Windows kullanıcısının sistemi baştan sona çalıştırabilmesi için hazırlanmıştır.

## 1. Gereksinimler

Bu sürüm:

- Windows 10/11 64-bit
- NVIDIA CUDA ekran kartı
- Güncel NVIDIA sürücüsü

için hazırlanmıştır.

İlk kurulumda internet bağlantısı gerekir. Model yaklaşık 5 GB'tır; PyTorch ve diğer bağımlılıklar da birkaç GB kullanır. En az yaklaşık 20 GB boş alan tavsiye edilir.

## 2. GitHub'dan indirme

1. Repository sayfasında **Code** butonuna basın.
2. **Download ZIP** seçeneğini seçin.
3. İnen ZIP dosyasına sağ tıklayıp **Tümünü ayıkla / Extract All** seçeneğini kullanın.
4. Mümkünse repository'yi kısa bir yola çıkarın. Örnek:

```text
C:\AI\VoxCPM2-Pytorch-Turkce-Versiyon-UI
```

## 3. Tek tık kurulum

Klasör içindeki:

```text
install.bat
```

dosyasına çift tıklayın.

Pencereyi kurulum tamamlanmadan kapatmayın.

### Python aşaması

Installer önce bilgisayarda 64-bit Python 3.11 olup olmadığını kontrol eder.

Varsa onu yalnız `.venv` oluşturmak için kullanır.

Yoksa resmî python.org adresinden Python 3.11.9 indirir ve repository içindeki `runtime/python311` klasörüne sessiz biçimde kurar.

### FFmpeg aşaması

FFmpeg 7.1.1 Full Shared indirilir.

ZIP SHA256 değeri doğrulanır. Doğrulama başarısızsa kurulum durur.

FFmpeg repository içinde:

```text
runtime\ffmpeg
```

altında tutulur. Sisteminizde ayrıca FFmpeg kurulu olması gerekmez.

### `.venv` aşaması

Repository içinde:

```text
.venv
```

oluşturulur.

Bütün Python paketleri bu sanal ortama kurulur.

### PyTorch aşaması

Aşağıdaki sürümler CUDA 12.8 PyTorch index'inden kurulur:

```text
torch 2.8.0
torchvision 0.23.0
torchaudio 2.8.0
```

### VoxCPM aşaması

Temel sabitlenmiş paketler:

```text
voxcpm 2.0.3
torchcodec 0.7.0
numpy 1.26.4
gradio >=6,<7
```

### Model aşaması

VoxCPM2 model dosyaları:

```text
models\VoxCPM2
```

klasörüne indirilir.

Model ağırlıkları GitHub repository'sine commit edilmez.

İndirme yarıda kalırsa `install.bat` yeniden çalıştırılabilir.

## 4. Kurulum sonu kontrolü

Installer otomatik olarak şunları kontrol eder:

- Python sürümü
- PyTorch sürümü
- CUDA kullanılabilirliği
- NVIDIA GPU adı ve VRAM
- FFmpeg
- TorchCodec DLL yüklemesi
- VoxCPM importu
- Gradio ve ses bağımlılıkları
- VoxCPM2 model dosyaları

Başarılı olduğunda:

```text
KURULUM BAŞARILI
```

yazısı görülür ve UI yeni pencerede açılır.

Kurulum günlüğü:

```text
logs\
```

klasöründe tutulur.

## 5. Daha sonraki açılışlar

Kurulum yalnız ilk sefer gereklidir.

Sonraki kullanımlarda:

```text
run.bat
```

dosyasına çift tıklayın.

Tarayıcı otomatik açılır.

Normalde:

```text
http://127.0.0.1:7861
```

kullanılır.

7861 doluysa 7862, 7863 ... 7870 aralığında ilk boş port otomatik seçilir.

UI açıkken terminal penceresini kapatmayın.

## 6. Seslendirme

### Seslendirilecek metin

Türkçe metni buraya yazın veya yapıştırın.

ChatGPT gibi yerlerden kopyalanan Markdown metnindeki `**kalın**`, başlık ve benzeri işaretler Clean Paste sistemiyle sadeleştirilir. Satır ve paragraf düzeni korunmaya çalışılır.

### Referans MP3 / WAV

Klonlanacak ses için referans kayıt yükleyin.

Temiz, anlaşılır ve arka plan gürültüsü az bir ses genellikle daha iyi sonuç verir.

Yalnızca kullanma izniniz bulunan sesleri kullanın.

### Stil

İsteğe bağlıdır.

Boş bırakıldığında hiçbir stil komutu modele eklenmez.

## 7. Varsayılan üretim ayarları

### Chunk hedefi

```text
200 karakter
```

Uzun metin sentence-first mantığıyla parçalanır.

### Chunk arası duraklama

```text
140 ms
```

### CFG

```text
2.0
```

### Inference Timesteps

```text
12
```

### Min Len

Otomatiktir.

Chunk karakter sayısına göre ayrı hesaplanır.

### Max Len

```text
180
```

### Her üretimde yeni Seed

Varsayılan olarak açıktır.

Her yeni üretimde yeni bir Run Seed oluşturulur. Böylece aynı metni yeniden ürettiğinizde önceki sesin saniyesi saniyesine kopyası gelmez.

Tek üretim içindeki bütün chunk'lar aynı Run Seed'i kullanır.

### Seed

```text
42
```

değeri ekranda kalır fakat **Her üretimde yeni Seed** açıkken kullanılmaz.

Otomatik Seed kapatıldığında bu alan devreye girer.

### Son konuşma hızı

```text
1.00
```

1.00 değerinde FFmpeg `atempo` filtresi uygulanmaz.

### MP3 bitrate

```text
192k
```

## 8. Çıktı dosyaları

Üretim sonunda:

- MP3 player
- WAV indirme
- MP3 indirme
- üretim günlüğü
- VRAM bilgileri

görünür.

Dosyalar ayrıca:

```text
outputs\
```

klasörüne yazılır.

Ham/final WAV PCM 24-bit yazılır.

## 9. Türkçe telaffuz düzeltmesi

Model tarafında şu dönüşüm aktiftir:

```text
acve -> ac ve
```

Textbox içinde `acve` yazmaya devam edebilirsiniz. Yalnız modele gönderilen metin değiştirilir ve tam kelime eşleşmesi kullanılır.

## 10. Kararlılık özellikleri

UI'de aşağıdaki yapı korunmuştur:

- SAFE KV cache: 2048
- TF32 kapalı
- `optimize=False`
- Denoiser kapalı
- Normalize kapalı
- Bad-case retry kapalı
- Dynamic Min Len
- Sentence-first chunking
- 12 ms Edge Fade
- Peak Guard
- Micro De-click Guard
- Stability Guard
- PCM 24-bit WAV
- Güvenli chunk-sınırı iptali
- Otomatik yeni Run Seed

`İşlemi İptal Et`, aktif CUDA `model.generate()` çağrısını zorla öldürmez. Mevcut chunk tamamlandıktan sonra yeni chunk başlatılmaz.

## 11. Sorun giderme

### CUDA bulunamadı

NVIDIA sürücüsünü güncelleyin ve bilgisayarı yeniden başlatın.

Installer CUDA 12.8 runtime içeren PyTorch wheel'ini kurar. Normal kullanım için ayrıca CUDA Toolkit kurmanız çoğunlukla gerekmez.

### NVIDIA ekran kartım yok

Bu UI mevcut hâliyle `device="cuda"` kullandığından NVIDIA CUDA GPU gerektirir.

### Model indirme yarıda kaldı

`install.bat` dosyasını yeniden çalıştırın.

### TorchCodec / libtorchcodec hatası

`install.bat` dosyasını yeniden çalıştırın. UI, repository ile birlikte indirilen FFmpeg 7.1.1 Full Shared klasörünü kullanır.

### 7861 portu dolu

Bir şey yapmanız gerekmez; UI 7861-7870 arasında otomatik port seçer.

### 7861-7870 tamamen dolu

Bu portları kullanan uygulamalardan bazılarını kapatın.

### Windows SmartScreen / Defender uyarısı

BAT ve PowerShell scriptleri repository içinde düz metindir ve GitHub üzerinden incelenebilir. Windows internetten indirilen scriptlerde güvenlik uyarısı gösterebilir.

### Kurulum hatasını nasıl paylaşırım?

`logs` klasöründeki en yeni:

```text
install_YYYYAAGG_SSDDSS.log
```

dosyasını hata bildiriminize ekleyin.

## 12. Temiz kurulum

Önemli `outputs` dosyalarınızı yedekleyin.

Ardından şu klasörleri silip `install.bat` çalıştırabilirsiniz:

```text
.venv
runtime
models\VoxCPM2
```

## 13. Upstream projeler

- https://github.com/OpenBMB/VoxCPM
- https://huggingface.co/openbmb/VoxCPM2
- https://pytorch.org/
- https://github.com/meta-pytorch/torchcodec
- https://ffmpeg.org/
- https://www.python.org/

Bu repository bağımsız bir Türkçe UI/kurulum katmanıdır; OpenBMB'nin resmî repository'si değildir.

## Terminalde Türkçe karakterler bozuk görünürse

Bu repository'de Windows terminal kodlaması için ek önlem alınmıştır:

- `install.bat` ve `run.bat` ASCII karakterlerle tutulur ve `chcp 65001` kullanır.
- `scripts/install.ps1` dosyası **UTF-8 BOM** ile kaydedilir.
- PowerShell içinde `[Console]::InputEncoding`, `[Console]::OutputEncoding` ve `$OutputEncoding` açıkça UTF-8 olarak ayarlanır.
- Python çalıştırılırken `PYTHONUTF8=1` ve `PYTHONIOENCODING=utf-8` kullanılır.

Buna rağmen eski `cmd.exe` penceresinde karakterler hatalı görünüyorsa Windows Terminal kullanmanız tavsiye edilir. Terminal yazı tipi olarak **Cascadia Mono** veya **Consolas** seçebilirsiniz.
