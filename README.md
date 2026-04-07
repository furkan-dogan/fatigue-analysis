# Kickboks Yorgunluk Analizi

Kickboks / taekwondo antrenmanlarında **antrenman öncesi ve sonrası video** karşılaştırarak sporcuda oluşan yorgunluğu açısal ve biyomekanik metriklerle ölçen analiz sistemi.

## Özellikler

### Pose & Hareket Analizi
- MediaPipe ile 33 nokta pose tespiti
- 10 eklem açısı hesabı (omuz, dirsek, kalça, diz, ayak bileği — sağ/sol)
- Normalize tekme yüksekliği (torso uzunluğuna göre)

### Hız & İvme
- Savitzky-Golay filtresi ile açısal hız (°/s) ve ivme (°/s²) — faz kayması yok
- Normalize ayak ucu hızı (torso/s)

### Tekme Tespiti
- Otomatik kick event detection (peak tabanlı)
- Doğrulama filtreleri: min diz ROM, min peak yükseklik
- Sahte tespitleri eler (weight-shift, duruş değişikliği)

### Faz Segmentasyonu
Her tekme 3 faza ayrılır:
- **Yüklenme** (start → chamber): diz en fazla büküldüğü ana kadar
- **Uzatma** (chamber → extension): patlayıcı uzatma, impact noktası
- **Geri Çekim** (extension → end): yorgunlukta en belirgin yavaşlama

### Yorgunluk Analizi
- Composite Yorgunluk İndeksi (0–100)
- Bilateral Asimetri İndeksi (Sağ-Sol farkı %)
- Cohen's d etki büyüklüğü + %95 güven aralığı
- Pose güven skoru — düşük kaliteli tekmeleri otomatik işaretler

### EMG Senkronizasyonu (altyapı hazır)
- Delsys / Noraxon / generic EMG CSV desteği
- Video frame zamanlarına interpolasyon
- Kick başına pencere RMS hesabı

---

## Kurulum

### Gereksinimler
- Python 3.12
- macOS / Linux (Windows test edilmedi)

```bash
git clone https://github.com/furkan-dogan/fatigue-analysis.git
cd fatigue-analysis

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Kullanım

### Streamlit Dashboard (önerilen)

```bash
.venv/bin/streamlit run app.py
```

Tarayıcıda `http://localhost:8501` açılır. Üç sayfa:

| Sayfa | Ne yapar |
|-------|----------|
| **Tek Video Analizi** | Tekli video analizi, eklem açıları, hız grafikleri, kick inceleme |
| **Çift Video Analizi** | Pre + Post video yükle, yorgunluk indeksi, faz analizi, istatistik |
| **EMG Sync** | EMG CSV ile video verilerini hizala |

### Komut Satırı

```bash
# Tek video analizi
python main.py \
  --input videos/pre_video.mp4 \
  --output output/pre_annotated.mp4 \
  --frame-csv output/pre_frame_metrics.csv \
  --events-csv output/pre_kick_events.csv

# Pre/Post karşılaştırma
python compare_pre_post.py \
  --pre-events output/pre_kick_events.csv \
  --post-events output/after_kick_events.csv \
  --output output/pre_post_comparison.csv
```

---

## VPS Kurulumu (uzaktan erişim)

```bash
# Ubuntu 22.04
apt update && apt install python3.12 python3.12-venv python3.12-dev \
    git libgl1 libglib2.0-0 ffmpeg -y

git clone https://github.com/furkan-dogan/fatigue-analysis.git
cd fatigue-analysis
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Arka planda çalıştır
nohup .venv/bin/streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.maxUploadSize 200 > streamlit.log 2>&1 &

ufw allow 8501 && ufw enable
```

Erişim: `http://VPS_IP:8501`

---

## Çıktılar

| Dosya | İçerik |
|-------|--------|
| `*_annotated.mp4` | İskelet + eklem açıları overlay'li video |
| `*_frame_metrics.csv` | Frame başına açı, hız, ivme, güven skoru |
| `*_kick_events.csv` | Kick başına tüm metrikler (faz süreleri, ASI, güven) |
| `yorgunluk_raporu.csv` | Cohen's d, %95 CI, yorgunluk indeksi |

---

## Metrikler

### Frame Düzeyi
- `R/L_KNEE/HIP/ANKLE/SHOULDER/ELBOW` — eklem açıları (°)
- `R/L_KNEE/HIP/ANKLE_vel_deg_s` — açısal hız (°/s)
- `R/L_KNEE/HIP/ANKLE_acc_deg_s2` — açısal ivme (°/s²)
- `R/L_FOOT_speed_norm` — normalize ayak hızı (torso/s)
- `pose_confidence` — landmark güven skoru (0–1)

### Kick Düzeyi
- `active_knee_rom_deg` — aktif diz hareket açısı
- `active_peak_knee_vel_deg_s` — peak açısal hız
- `time_to_peak_knee_vel_sec` — patlayıcı güç göstergesi
- `loading/extension/retraction_dur_sec` — faz süreleri
- `extension/retraction_peak_vel_deg_s` — faz hızları
- `knee_asi / hip_asi` — bilateral asimetri indeksi (%)
- `pose_confidence / confidence_flag` — güven skoru

### Yorgunluk İndeksi Ağırlıkları
| Metrik | Ağırlık |
|--------|---------|
| Diz ROM azalması | 0.25 |
| Peak hız azalması | 0.25 |
| Peak hıza ulaşma süresi artışı | 0.15 |
| Tekme yüksekliği azalması | 0.15 |
| Ayak hızı azalması | 0.10 |
| Tekme süresi artışı | 0.05 |
| Ort. hız azalması | 0.05 |

---

## Proje Yapısı

```
fatigue-analysis/
├── app.py                  # Streamlit dashboard
├── main.py                 # CLI — tek video analizi
├── compare_pre_post.py     # CLI — pre/post CSV karşılaştırma
├── requirements.txt
└── src/
    ├── pipeline.py         # Ana analiz pipeline (importable)
    ├── metrics.py          # Açı, hız, ASI hesapları
    ├── events.py           # Kick tespiti + faz segmentasyonu
    ├── pose_runner.py      # MediaPipe backend
    ├── draw.py             # Video annotasyon
    ├── exporter.py         # CSV export
    ├── stats.py            # Cohen's d, CI, etki büyüklüğü
    └── emg_sync.py         # EMG senkronizasyon
```

---

## Gelecek Çalışmalar
- [ ] YOLOv8-pose backend (hızlı tekmelerde daha doğru tracking)
- [ ] Trunk normalizasyonu (gövde eğimi kompanzasyonu)
- [ ] Kamera kalibrasyonu (piksel → gerçek birim)
- [ ] 3D pose lifting (MotionBERT)
- [ ] EMG donanım entegrasyonu (Delsys Trigno)
