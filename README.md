# Spor video analizi

Python, Streamlit ve MediaPipe ile yerel video analizi. Açılış branşı voleybol; üç branş ayrı girişlerden seçilir. Taekwondo analizleri çalışır. Voleybolda tek video yükleme, kare inceleme, manuel tekrar/kalibrasyon işaretleme ve revizyon kaydı vardır; otomatik voleybol ölçümleri henüz yoktur. Basketbol geliştirme durumundadır. Gerçek video ölçüm doğrulaması henüz yapılmadı.

## Çalıştırma

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

CLI: `.venv/bin/python main.py --help`. Varsayılan giriş `data/videos/sample.mp4`, çıktılar `data/output/`. Arayüz analizleri `data/analyses.sqlite3` ve `data/analyses/` altında kalıcıdır. Taekwondo ekranındaki **Kayıtlı analizler → Kaydı aç** ile geri yüklenir; yeniden analiz yeni revizyon oluşturur. CLI dosya dışa aktarma akışını korur.

Kaynak kare zamanlarını saklamak için sistemde `ffprobe` bulunmalı (FFmpeg paketinin parçası). Yoksa kaynak zamanları eksik işaretlenir; nominal FPS zamanı gerçek fiziksel zaman diye sunulmaz. Orijinal videolar, model noktaları ve görünürlükler saklanır. Kayıtları taşırken SQLite dosyasıyla birlikte `data/analyses/` klasörünü de taşıyın.

Voleybolda **Videoyu kaydet ve incele** ile başlayın. Kare numarasıyla kalkış/inişi kontrol edin; test türü, sporcu, inceleme aralığı ve isteğe bağlı referansları girip **İncelemeyi yeni revizyon olarak kaydet** seçeneğini kullanın. Kayıtlı incelemelerden önceki revizyonlar tekrar açılabilir. Kalibrasyon koordinatları kaynak görüntünün sol üstünden piksel cinsindedir; kaydedilmiş referanslar ilgili karede gösterilir.

## Yapı

- `src/core/`: bağımsız sayısal araçlar ve veri tipleri.
- `src/adapters/`: poz kestirimi, çizim ve dosya erişimi.
- `src/sports/`: branşa özel analiz ve rapor hazırlığı.
- `ui/components/`: ortak yükleyici, video, zaman çizelgesi, metrik, kalite ve karşılaştırma bileşenleri.
- `ui/sports/`: branş ekranları ve oturum akışları.
- `cli/`: komut satırı girişleri.
- `tests/`: mimari, regresyon ve arayüz kontrolleri.
- `data/`: Git dışında tutulan yerel videolar ve çıktılar.

Çalışma planı ve devam notları: [upgrade.md](upgrade.md). Mimari sınırlar: [docs/architecture.md](docs/architecture.md).

## Kontrol

```sh
.venv/bin/python -m unittest discover -v
```

Kod testleri ölçüm doğruluğunu kanıtlamaz. Görsel asimetri kuvvet farkı değildir; açı hızları sprint hızı değildir.
