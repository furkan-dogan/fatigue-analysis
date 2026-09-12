# Spor video analizi

Python ve Streamlit ile yerel video analizi. Taekwondo MediaPipe, voleybol RTMPose-L WholeBody kullanır. Açılış branşı voleybol; üç branş ayrı girişlerden seçilir. Taekwondo analizleri çalışır. Voleybolda tek video yükleme, kare inceleme, manuel tekrar/kalibrasyon işaretleme ve revizyon kaydı vardır; protokol kontrollerine bağlı deneysel CMJ, 2D asimetri ve kalibre sprint hesapları vardır. Basketbol geliştirme durumundadır. Gerçek video ölçüm doğrulaması henüz yapılmadı.

## Çalıştırma

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

CLI: `.venv/bin/python main.py --help`. Varsayılan giriş `data/videos/sample.mp4`, çıktılar `data/output/`. Arayüz analizleri `data/analyses.sqlite3` ve `data/analyses/` altında kalıcıdır. Taekwondo ekranındaki **Kayıtlı analizler → Kaydı aç** ile geri yüklenir; yeniden analiz yeni revizyon oluşturur. CLI dosya dışa aktarma akışını korur.

Kaynak kare zamanlarını saklamak için sistemde `ffprobe` bulunmalı (FFmpeg paketinin parçası). Yoksa kaynak zamanları eksik işaretlenir; nominal FPS zamanı gerçek fiziksel zaman diye sunulmaz. Orijinal videolar, model noktaları ve görünürlükler saklanır. Kayıtları taşırken SQLite dosyasıyla birlikte `data/analyses/` klasörünü de taşıyın.

Voleybolda tek video seçip **Videoyu analiz et** düğmesine basın. Kaynak kaydedilir ve tüm videoda sporcu/hareket taraması başlar. Toplu yükleme ve kuyruk ertelendi. Otomatik sonuçlar deneysel hareket adaylarıdır; fiziksel cm/hız hesapları henüz bu akışa bağlı değildir.

Ana ekran hareket listesini ve tek videoyu gösterir. Hareket seçince ilgili ana gider. Sabit zamanlı kaynaklarda iskelet/faz işaretli sessiz önizleme oluşturulur. Belirsiz çoklu sporcu varsa görsel seçim istenir. Eski kayıtlarda **Hareketleri otomatik bul** düğmesi aynı taramayı başlatır. Mevcut manuel akış için **Teknik incelemeyi aç** seçeneğini kullanın: test/çekim/temas ayarlarını inceleyip yeni revizyon olarak kaydedin, ardından **Kaydedilmiş ayarlarla analiz et** düğmesine basın. Teknik düzeltmeler ve önceki kayıtlar korunur.

İlk voleybol analizinde model dosyaları internetten indirilir ve SHA256 ile doğrulanır; sonraki çalıştırmalar yerel önbelleği kullanır. CPU analizi zaman alabilir. Uygun çekim/zaman/kalibrasyon yoksa sayı yerine gerekçe gösterilir. Yöntemler, çekim koşulları ve sınırlar: [docs/measurement-methods.md](docs/measurement-methods.md).

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
