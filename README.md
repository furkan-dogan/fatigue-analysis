# Spor video analizi

Python, Streamlit ve MediaPipe ile yerel video analizi. Açılış branşı voleybol; üç branş ayrı girişlerden seçilir. Çalışan analiz taekwondo; voleybol ve basketbol ekranları geliştirme durumunu gösterir. Gerçek video ölçüm doğrulaması henüz yapılmadı.

## Çalıştırma

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

CLI: `.venv/bin/python main.py --help`. Varsayılan giriş `data/videos/sample.mp4`, çıktılar `data/output/`. Arayüz analizleri `data/analyses.sqlite3` ve `data/analyses/` altında kalıcıdır. Taekwondo ekranındaki **Kayıtlı analizler → Kaydı aç** ile geri yüklenir; yeniden analiz yeni revizyon oluşturur. CLI dosya dışa aktarma akışını korur.

Kaynak kare zamanlarını saklamak için sistemde `ffprobe` bulunmalı (FFmpeg paketinin parçası). Yoksa kaynak zamanları eksik işaretlenir; nominal FPS zamanı gerçek fiziksel zaman diye sunulmaz. Orijinal videolar, model noktaları ve görünürlükler saklanır. Kayıtları taşırken SQLite dosyasıyla birlikte `data/analyses/` klasörünü de taşıyın.

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
