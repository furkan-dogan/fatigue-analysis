# Mimari ve dosya taşıma haritası

## Katmanlar

| Katman | Sorumluluk | Bağımlılık |
|---|---|---|
| `src/core` | Veri modeli, geometri, sinyal, istatistik | Python, NumPy/SciPy; UI/model backend yok |
| `src/adapters` | MediaPipe, OpenCV, CSV | Çekirdek ve ilgili dış kütüphane |
| `src/sports/<sport>` | Branş olayları, metrikleri, pipeline | Çekirdek ve adaptörler; diğer branş yok |
| `ui/components` | Ortak gösterim | Streamlit/Plotly; branş import yok |
| `ui/sports/<sport>` | Branş ekranı, rapor sunumu | Ortak bileşenler ve kendi branşı |
| `ui/app.py`, `cli` | Giriş/orkestrasyon | İlgili branş ve UI |

Şimdilik yalnızca taekwondo çalışır. Voleybol/basketbol klasörleri boş algoritma vaadi yerine
gelecek implementasyonun sorumluluğunu belgeler. Uygulama kabuğu hâlen taekwondo'yu açar;
branş navigasyonu ayrı adımdır.

## Eski → yeni

| Eski | Yeni |
|---|---|
| `src/pose_runner.py` | `src/adapters/mediapipe_pose.py`; veri tipi `src/core/types.py` |
| `src/metrics.py` | Geometri `src/core/geometry.py`, türev `src/core/signals.py`, tekme metrikleri `src/sports/taekwondo/metrics.py` |
| `src/utils.py`, `src/stats.py` | `src/core/numeric.py`, `src/core/statistics.py` |
| `src/events.py`, `src/fatigue.py`, `src/pipeline.py` | `src/sports/taekwondo/` altında aynı adlar |
| `src/sensors.py`, `src/emg_sync.py` | `src/sports/taekwondo/simulation.py`, `sensor_sync.py` |
| `src/draw.py`, `src/exporter.py` | `src/adapters/drawing.py`, `csv_export.py` |
| `ui/video_analysis.py` | `ui/sports/taekwondo/page.py` |
| `ui/report.py`, `ui/analysis_helpers.py` | `ui/sports/taekwondo/report.py`, `presentation.py` |
| `ui/components.py` | Ortak oynatıcı `ui/components/video_player.py`, tekme panelleri `ui/sports/taekwondo/components.py` |
| UI içindeki klip/sensör hesabı | `src/adapters/video_clips.py`, `src/sports/taekwondo/sensor_summary.py` |
| `ui/charts.py` | Genel grafikler `ui/components/charts.py`, tekme grafikleri `ui/sports/taekwondo/charts.py` |
| `main.py` içeriği | `cli/analyze.py`; kökte uyumlu giriş dosyası |
| `app.py` içindeki CSS | `ui/theme.py` |
| Eski kök belgeler | `docs/archive/` |
| `generate_report_doc.py` | `tools/legacy/generate_report_doc.py` |
| Kök YOLO ağırlıkları | `models/` |

Eski iç Python import yolları korunmuyor; repo içindeki çağrılar güncellendi.
Kullanıcı komutları `streamlit run app.py`, `python main.py`, `python cli/compare.py` korunuyor.
Harici scriptler eski `src.pipeline` importunu kullanıyorsa yeni açık yola güncellenmeli.

## Doğrulama

- Mimari testler katman bağımlılıklarını tarar.
- Regresyon kaydı taşıma öncesi `439b3e9` commitinde iki sentetik tekmeden oluşturuldu.
- Bu kayıt bilinen eski fatigue/simülasyon davranışını da içerir; bilimsel doğruluk referansı değildir.
- Pipeline testi model çalıştırmadan geçici videoyu açar ve CSV/anotasyon çıktısını okur.
- AppTest boş yükleme ekranını açar; CLI testleri giriş yollarını kontrol eder.
- Gerçek video, tüm rapor sekmeleri ve uzak sunucu davranışı bu aşamada doğrulanmış sayılmaz.

## Sonraki sınırlar

`src/movements/` sıçrama/iniş/sprint gerçekten uygulanırken açılacak. Voleybolun protokol,
kalibrasyon ve ölçüm kuralları burada ortak yetenekleri birleştirecek.
Kalıcı analiz modelleri/kayıt, ortak metrik/kalite kartları ve branş navigasyonu sıradaki aşamalardır.
İş durumunun tek kaynağı `upgrade.md` dosyasıdır.
