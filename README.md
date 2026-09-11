# Spor Hareket Analizi

Video üzerinden sporcu hareketlerini inceleyen Python/Streamlit uygulaması.
Voleybol öncelikli dönüşüm sürüyor; mevcut çalışan analiz taekwondo pre/post tekme analizidir.
Voleybol ve basketbol için modül sınırları açıldı, algoritmaları henüz uygulanmadı.

## Başlangıç

Python 3.12 kullanın. Komutları proje kökünden çalıştırın.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Mevcut ekran iki video yükleyerek taekwondo karşılaştırması yapar.
Tek video yüklemeli voleybol ekranı, [upgrade.md](upgrade.md) planının sonraki aşamasıdır.

```bash
# Taekwondo: tek video, anotasyon ve CSV çıktıları
.venv/bin/python main.py --input videos/pre_video.mp4

# Aynı analiz komutunun modül girişi
.venv/bin/python -m cli.analyze --input videos/pre_video.mp4

# İki olay CSV'sini karşılaştır
.venv/bin/python -m cli.compare \
  --pre-events output/pre_kick_events.csv \
  --post-events output/post_kick_events.csv

# Kontroller
.venv/bin/python -m unittest discover -v
```

## Dosya yapısı

```text
app.py                         Streamlit giriş noktası
main.py                        Geriye uyumlu analiz CLI girişi
cli/                           Analiz ve CSV karşılaştırma komutları
src/
  core/                        Veri tipleri, geometri, sinyal ve istatistik
  adapters/                    MediaPipe, video çizimi/kırpma, CSV export
  sports/
    taekwondo/                 Mevcut pipeline, tekme metrikleri ve sensör araçları
    volleyball/                Voleybol analiz modülü için ayrılan alan
    basketball/                Basketbol analiz modülü için ayrılan alan
ui/
  app.py                       Uygulama kabuğu
  theme.py                     Sayfa ayarları ve ortak stil
  paths.py                     Ortak kaynak yolları
  components/                  Branştan bağımsız video ve grafik bileşenleri
  sports/                      Her branşın ekran ve sunum kodu
tests/                         Regresyon, mimari sınır, giriş ve pipeline kontrolleri
sample_data/                   Örnek sensör CSV dosyaları
models/                        Önceden eklenen, aktif akışta kullanılmayan YOLO ağırlıkları
docs/                          Mimari ve tarihsel belgeler
tools/legacy/                  Eski bağımsız rapor oluşturucu
videos/                        Yerel giriş videoları; Git dışında
output/                        Yerel üretilen çıktılar; Git dışında
upgrade.md                     Sıralı plan, durum, açık sorunlar ve devir notları
```

## Mimari

`UI / CLI → branş pipeline → ortak çekirdek ve adaptörler`

Çekirdek Streamlit, MediaPipe veya branş modüllerini import etmez.
Ortak UI bileşenleri branş ekranlarını import etmez. Branşlar birbirini import etmez.
Ayrıntılı sorumluluklar ve eski/yeni dosya haritası: [docs/architecture.md](docs/architecture.md).

## Mevcut ölçüm sınırları

Yapısal taşıma eski taekwondo hesaplarının doğrulandığı anlamına gelmez.
Yorgunluk skoru, hız birimi, sensör kaynağı ve yerel video sunumu hakkında bilinen sorunlar
[upgrade.md](upgrade.md) içinde açıkça izleniyor. Mevcut EMG/NIRS simülasyonu gerçek ölçüm değildir.
Voleybol için bu hesaplar doğrudan kullanılmayacak.

Testler kod taşınmasını ve temel çalışmayı kontrol eder; gerçek sporcu videosunda doğruluk ölçmez.
Video tabanlı yükseklik/hız doğrulaması ayrıca yapılacak.

## Belgeler ve yardımcı araçlar

- [upgrade.md](upgrade.md): önce bunu okuyun; tamamlanan ve sıradaki işler burada.
- [AGENTS.md](AGENTS.md): geliştirici/AI çalışma kuralları.
- [docs/archive/](docs/archive/README.md): eski proje belgeleri, güncel ürün iddiası değildir.
- [tools/legacy/](tools/legacy/README.md): isteğe bağlı Word rapor aracı.
