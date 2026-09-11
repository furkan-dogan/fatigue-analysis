# Voleybol öncelikli proje dönüşümü

## Devralan geliştirici için başlangıç

Önce bu dosyayı, `AGENTS.md`, `git status --short` ve son commitleri oku. Kullanıcı yerel düzenleme ve Türkçe küçük commitler istedi; push istenmedi. Python/Streamlit ve tek kullanıcı korunacak. Canlı analiz, çok kullanıcı ve yeni frontend kapsam dışında. Şimdi video isteme: kullanıcı model seçimi için 6. adımda örnek video verecek; bağımsız doğrulama 8. adımda.

## Güncel durum — 2026-09-11

**Adım 1–6 tamamlandı. Adım 7 son kontrollerde.** Ortak bileşenler ve altı bölümlü taekwondo ekranı çalışıyor. Üç branşın ekran girişi var; voleybol varsayılan. Voleybolda manuel inceleme hazır; otomatik ölçümler ve basketbol analizi henüz hazır değil.

Kullanıcının yeni kararı: yalnızca AI görüntü/video işleme. IMU, EMG, NIRS ve tüm cihaz/simülasyon kapsamı kaldırıldı. Eski sensör örnekleri, raporları, kullanılmayan YOLO ağırlıkları ve tarihsel Word/rapor araçları silindi; geçmişleri Git'te bulunur. Eski gerçek çıktı dosyaları silinmeden `data/output/` altına taşındı.

### 1. Proje yapısı — tamamlandı

Çekirdek, adaptör, branş, ortak UI ve CLI ayrıldı. Voleybol ve basketbol için alan açıldı. Detaylı harita `README.md` ve `docs/architecture.md` içinde.

### 2. Component-first arayüz — tamamlandı

- Ortak yükleyici, video, olay zaman çizelgesi, metrik, kalite ve karşılaştırma bileşenleri hazır.
- Sayfa yalnızca oturum ve görünümleri birleştirir; rapor hesapları backend'de.
- Altı bölüm: özet, açı/hız, tekmeler, faz/asimetri, istatistik, rapor/CSV.
- Cihaz alanları pipeline, UI, rapor ve örneklerden kaldırıldı.
- Doğrulanmamış yorgunluk skoru, sağlık/beslenme önerileri kaldırıldı. Açı/faz/olay hesapları korundu.
- Özel localhost video sunucusu yerine Streamlit medya sunumu kullanılıyor.
- Tek kullanımlık çok sayıda rapor/sekme dosyası yerine kısa sorumluluk modülleri kullanıldı.

### 3. Branş navigasyonu — tamamlandı

- Voleybol, taekwondo, basketbol için ayrı ekran girişleri.
- Voleybol varsayılan öncelik; basketbol geliştirme durumunu açık gösterir.
- Branş değişimi eski sonuçları başka branş altında göstermez. Tamamlanmış çift analiz `taekwondo_analysis` altında tutulur; widget anahtarları branşa özeldir. Kalıcı kayıt kimlikleri SQLite kataloğunda tutulur.
- Branş değiştirip dönünce tamamlanmış sonuçlar korunur. Yükleyici widgetları Streamlit yaşam döngüsü gereği yeniden dosya seçimi isteyebilir; mevcut sonuçlar etkilenmez.
- Taekwondo'nun mevcut ekranı erişilebilir kalır.

Kabul: Üç branş seçilebilir; hazır olmayan analizler çalıştırılamaz ve sahte sonuç gösterilmez.

### 4. Analiz sözleşmeleri ve kalıcı kayıt — tamamlandı

- VideoAsset, Session, AnalysisRun, MovementEvent, MetricResult ve Comparison modelleri.
- Sonuç: değer, birim, yöntem, kaynak video/zaman aralığı, kalite, protokol/model/algoritma sürümü.
- Eksik metrik sıfır değildir. Bir alan farklı birimler taşıyamaz.
- SQLite: oturum/analiz/sonuç kayıtları. Yerel analiz kimliği klasörleri: video ve büyük zaman serileri.
- Orijinal video korunur; düzeltmeler ayrı revizyon olarak kaydedilir.
- Pose landmarkları, görünürlük ve gerçek zaman damgaları saklanmalı; yalnızca açı CSV'si yeterli değil.
- Hatalı/yarım analiz, tekrar çalıştırma ve uygulama yeniden açılınca kayıtları bulma senaryoları.

Kabul: Bir analiz kaydedilip yeniden açılır; kaynağı, birimi ve sürümü izlenebilir.

### 5. Voleybol ekranı — tamamlandı

- Tek video yükleme yeterlidir; ikinci video yalnızca karşılaştırmada kullanılır.
- Test seçimi: CMJ/dikey sıçrama, iniş-asimetri, sprint.
- Video metadata ve çekim uygunluğu kontrolü; sporcu/analiz aralığı seçimi.
- Gerekirse kalibrasyon referans noktaları ve mesafeleri girilir.
- Sonuç alanı: video, olay zaman çizelgesi, metrikler, tekrar listesi ve geçmiş.
- Kalkış/iniş kareleri kullanıcı tarafından incelenebilir ve düzeltilebilir olmalı.

Kabul: Yükleme ve inceleme akışı tamamdır; mevcut olmayan hesaplar açıkça belirtilir.

### 6. Örnek video üzerinden model seçimi — ilk değerlendirme tamamlandı

- Kullanıcı örnek videoyu bu aşamada paylaşacak. Önce dosya/kayıt/voleybol inceleme akışı tamamlanır.
- YOLO26x-Pose yalnızca araştırma adayı; mevcut MediaPipe ve ayak noktaları içeren uygun bir pose modeli aynı karelerde karşılaştırılır.
- Video zamanlaması, çekim açısı, ayakların görünürlüğü ve sporcu hareketi incelenir. Model dosyası/sürümü, çalışma ayarları ve donanım kaydedilir.
- Elle işaretlenmiş ortak noktalar, ayak teması çevresindeki kareler, takip kopmaları, örtüşme, zamansal titreme, çalışma süresi ve bellek karşılaştırılır.
- Algoritmalar henüz hazır değilken sıçrama cm veya sprint hız doğruluğu iddiası üretilmez. İlk seçim noktaların/temas görünümünün uygunluğuna göre yapılır; ölçüm sonucu kararı 8. adımda doğrulanır.
- Gerekli noktaları vermeyen modelde ayak bileği ayak ucu yerine konmaz. Hiçbir aday yeterli değilse sonuç açıkça yetersiz kabul edilir.
- Tek örnek videodan tüm sporcular için en iyi model sonucu çıkarılmaz. Bu video geliştirme verisidir; bağımsız doğrulama videolarından ayrı tutulur.
- Çıktı: aday tablosu, görsel karşılaştırma ve gerekçeli ilk model/adaptör kararı. Canlı analiz modeli ayrıca daha sonra değerlendirilebilir.

Kabul: Örnek video üzerinde kanıtlı ilk seçim yapılır; seçim ölçüm doğruluğu onayı gibi sunulmaz.

### 7. Voleybol algoritmaları — son kontrollerde

#### 7A. Dikey sıçrama

- İlk protokol kontrollü, yerinde çift ayak CMJ. Blok ve yaklaşmalı smaç ayrı protokoller olarak sonra.
- Tekrar tespiti, son ayak yerden ayrılması ve ilk ayak yere teması üzerinden uçuş süresi.
- `h = g * t² / 8`: kalkıştan tepeye tahmini yükselme. Ayakta duruştan yükselme ve el erişim yüksekliği farklıdır.
- Kalkış/iniş kütle merkezi yüksekliği benzerliği varsayımı, bacak çekme ve görünmeyen temas durumları kontrol edilmeli.
- Kayıt FPS'i ile ağır çekim oynatma hızı ayrılmalı; gerçek fiziksel zaman doğrulanmadan yükseklik verilmez.
- Önerilen çekim: sabit kamera, iyi ışık, ayaklar net, orijinal 120/240 FPS. Yüksek FPS doğruluk garantisi değildir.
- Otomatik temas aralıkları ve manuel düzeltme birlikte desteklenmeli.

#### 7B. Yana sapma ve iniş asimetrisi

- Önden/arkadan uygun görünümde pelvis orta noktası sapması, gövde/pelvis eğimi, sağ-sol temas zaman farkı.
- Kamera eğimi, vücut yönelimi, örtüşme ve landmark kalitesi kontrol edilmeli.
- Santimetre için hareket düzlemine uygun kalibrasyon; yoksa açıkça normalize ölçüm.
- Görsel asimetri bacak kuvvet farkını kanıtlamaz; güçlü/zayıf bacak teşhisi veya yaralanma riski üretme.
- Sadece gözlenen 2D düzlem hakkında sonuç ver; gerçek 3D ölçüm iddiası yok.

#### 7C. Sprint

- Ölçülmüş düz parkur, sabit yandan kamera, bilinen referanslar, kadrajda tek sporcu.
- Kamera pan/zoom ve kesintiler ilk protokol dışında.
- Ayak ucu hızı sporcu ilerleme hızı değildir. Pelvis takibi kütle merkezi ölçümü diye adlandırılmaz.
- Kalibrasyon takip noktasının hareket düzlemiyle eşleşmeli; zemin homografisi havadaki kalçaya doğrudan uygulanmaz.
- Önce konum/mesafe/geçiş süresi doğrulanır; sonra filtrelenmiş konumdan hız türetilir.
- Anlık hızın zaman penceresi belirtilir; tek gürültülü kare tepe hız sayılmaz.
- Kalibrasyon/zamanlama yetersizse m/s veya km/saat verilmez.

Kabul: Algoritmalar kontrollü/sentetik girdilerle sınanır; gerçek video doğrulaması olmadan doğruluk iddiası yok.

### 8. Bağımsız videolar ve ölçüm doğruluğu — kullanıcı video verecek

- Videoyu kullanıcı bu aşamada sohbete veya uygulamaya verecek; dosya düzenlemek için video bekleme.
- İlk video: metadata, gerçek süre, görünürlük, kamera ve kalibrasyon uygunluğu incelenir.
- Elle işaretlenmiş kalkış/inişle zaman hatası; aynı tanımlı referans yükseklikle cm hatası.
- 2D açıların elle işaretlenmiş aynı düzlemdeki referansıyla derece hatası.
- Anlık hız için radar/lazer gibi senkron hız referansı; fotosel yalnızca aralık süresi/ortalama hızı doğrular.
- Olay tespitinde yanlış pozitifler ve kaçırılan tekrarlar raporlanır.
- MAE, sistematik sapma, büyük hatalar, ölçüm verilmeyen kayıt oranı ve tekrarlanabilirlik değerlendirilir.
- Geliştirme/ayar videoları ve doğrulama videoları sporcu/oturum bazında ayrılır.
- Sayısal kabul eşikleri referans ekipman ve test protokolü belirlendikten sonra yazılır; keyfi yüzde verilmez.

Kabul: Her metrik kendi biriminde, bağımsız videolarla ve açık sınırlarla değerlendirilir.

### 9. Karşılaştırma ve rapor — bekliyor

- Aynı sporcu/test/yöntem, uyumlu çekim ve kalibrasyon koşulları eşleşir.
- CMJ, blok ve yaklaşmalı smaç tek metrikmiş gibi karıştırılmaz.
- Tekrar ve oturum özetleri, mutlak/yüzde değişim ve kalite bilgisi gösterilir.
- Rapor gözlem ve ölçüm yöntemini açıklar; doğrulanmamış yorgunluk/sağlık çıkarımı yapmaz.
- Dışa aktarma ve önceki kaydı açma uçtan uca kontrol edilir.

## Bilinen sınırlar

- Normalize ayak hızında gövde referansı yoksa artık değer üretilmez (algoritma v2). Eski CSV dosyaları geriye dönük değiştirilmedi; geçmiş ham verilerde birim sorunu olabilir.
- Genel landmark görünürlüğü ölçüm doğruluğu değildir; metrik bazında kalite kontrolü eksik.
- Sabit FPS ve eksik nokta doldurma zaman/hız doğruluğunu etkileyebilir.
- Taekwondo iki video, voleybol incelemesi tek video kullanır. Voleybol kaydı manuel işaretleme içerir; otomatik ölçüm içermez.
- Yeni arayüz analizleri kalıcıdır; eski geçici oturumlar ve eski `data/output/` CSV dosyaları otomatik içe aktarılmadı. CLI bağımsız dosya dışa aktarma akışını korur.
- Kalıcı kayıtlar yereldir; `data/` klasörü birlikte taşınmalıdır. Veritabanı tek başına video/pose dosyalarının yerine geçmez.
- Otomatik video/model doğruluğu, sıçrama ve sprint ölçümleri henüz doğrulanmadı.

## Kontroller ve commit günlüğü

- Adım 1: `611bd59`, `439b3e9`, `ecfc058`, `4478f93`, `748305d`.
- `2dea859`: ortak bileşenler ve ilk sekme ayrımı; 17 test geçti.
- `de326c2`: cihaz akışları kaldırıldı; video raporu ve ekranlar sadeleşti.
- Son temizlik commitinde gereksiz model/arşiv/araç klasörleri kaldırıldı, CLI varsayılanları `data/` altına alındı; `git diff --check` temiz. Commitler yereldir; push yapılmadı.
- Yeni video-only düzen: 21 test geçti. Mimari, CLI, boş/dolu altı sekme, bileşen durumları, eksik/0/NaN rapor değerleri, hareket regresyonu, modelsiz video→CSV/anotasyon kontrol edildi.
- Regresyondan yalnızca kaldırılan cihaz/sentetik çıktılar ve eski yorgunluk skoru çıkarıldı; mevcut açı/hız/olay referansı yeniden hesaplanmadı.
- Eski sensör/sağlık raporu metin snapshot'ı kaldırıldı; yeni ölçüm karşılaştırması ve eksik veri testleriyle değiştirildi.
- Çift analiz sırası/ilerleme, hata durumunda yarım dosya temizliği, gerçek olay seçimi ve yönetilen video başlangıç zamanı da test edildi. AST/syntax kontrolü başarılı. Kod testleri bilimsel doğruluk testi değildir.

## Sıradaki somut iş

**Adım 7: seçilen RTMPose-L WholeBody adayını adaptör olarak ekle ve kontrollü CMJ algoritmasına başla.** Önce `docs/model-selection.md` oku. Mevcut motor MediaPipe; henüz değiştirilmedi. Bu örnek yaklaşmalı sıçramadır, CMJ veya fiziksel ölçüm doğrulama videosu değildir. İlk model kararı koşulludur; bağımsız doğrulama 8. adımda. Takip/kalite, ayak noktaları ve sahne/aralık sınırlarını doğrulamadan yükseklik veya hız üretme.

### Adım 3 doğrulaması

23 unittest testi geçti; AST/syntax ve `git diff --check` temiz. Varsayılan voleybol, hazır olmayan branşlarda yükleme/analiz bulunmaması, taekwondo → voleybol → basketbol → taekwondo geçişlerinde sonuçların korunması ve branşlar arasında sızmaması kontrol edildi. Mevcut altı taekwondo sekmesi testleri de geçti.

## Model araştırması — 2026-09-11

Mevcut çalışan motor `src/adapters/mediapipe_pose.py` içindeki MediaPipePoseRunner'dır (`model_complexity=1` varsayılanı). Eski `yolo11n-pose.pt` / `yolo11s-pose.pt` ağırlıkları çalışan pipeline'a bağlı değildi; kullanılmadıkları için kaldırılmıştı. Bu aşamada model indirilmedi, paket eklenmedi veya analiz motoru değiştirilmedi.

**YOLO ailesinde doğruluk öncelikli ilk değerlendirme adayı: `yolo26x-pose.pt`.** Bu bir üretim doğruluğu onayı değildir. Ultralytics güncel model olarak YOLO26'yı gösteriyor; Pose sürümü RLE ile nokta konumlandırmasını geliştiriyor. Kaynak: [YOLO26 resmi dokümanı](https://docs.ultralytics.com/models/yolo26/).

640 piksel COCO Keypoints val2017 için yayınlanan pose mAP50–95 skorları:

| Model | Pose mAP50–95 |
| --- | ---: |
| YOLO11n-Pose | 50.0 |
| YOLO11s-Pose | 58.9 |
| YOLO11x-Pose | 69.5 |
| YOLO26l-Pose | 70.4 |
| YOLO26x-Pose | 71.6 |

Kaynaklar: [YOLO11 resmi tablo kaynağı](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/models/yolo11.md), [YOLO26 Pose tablosu](https://docs.ultralytics.com/tasks/pose/). Bunlar yayıncı benchmarkları; kendi donanımımızda ölçülmedi. COCO skoru santimetre/hız doğruluğu veya başarılı analiz yüzdesi değildir. Sürümler arasında yayınlanan çıkarım ayarları da birebir aynı kabul edilmemeli.

Öneri gerekçesi: canlı çalışma zorunluluğu yok, doğruluk öncelikli; bu nedenle YOLO26 ailesinde yayınlanan en yüksek pose skoruna sahip x ile başla. l sürümü hız/bellek karşılaştırma adayıdır. x için yayınlanan CPU ONNX süresi yaklaşık 565 ms/kare; yerel Mac süresi değildir. Gerçek donanımda işleme süresi ve bellek ayrıca ölçülecek.

**Sıçrama için kritik sınır:** standart YOLO Pose 17 nokta verir; ayak bileği var, topuk/ayak ucu yok. Nokta listesi: [resmi Pose dokümanı](https://docs.ultralytics.com/tasks/pose/). Bizim mevcut Keypoints2D ve tekme hesabımız ayak ucu da kullanıyor; YOLO dosyasını değiştirip devam etmek doğru olmaz. Eksik noktalar açıkça eksik kalmalı; ayak bileği ayak ucu yerine konmamalı.

Bu nedenle ölçüm aşamasında MediaPipe referansı, YOLO26x-Pose ve ayak noktaları içeren RTMPose WholeBody gibi bir aday aynı işaretli videolarda karşılaştırılmalı. MMPose WholeBody ayak değerlendirmesi ve model bilgileri: [resmi model kataloğu](https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose_coco-wholebody.yml). Nokta sayısının fazla olması tek başına daha doğru ölçüm kanıtı değildir.

Seçim deneyi: aynı sporcu/çekim kesitlerinde kalkış–iniş zaman hatası, sıçrama cm hatası, görüntü düzlemi açı hatası, kalibre parkur hız hatası, takip kopması, ölçüm verilemeyen tekrar oranı ve işleme süresi değerlendirilecek. Ayar ve doğrulama videoları ayrılacak. Tek model tüm metriklerde en iyi çıkmak zorunda değil. Model/adaptör arayüzü 4. adımın kayıt sözleşmesiyle kaynak, nokta şeması ve sürüm bilgisini taşımalı; uygulama bir ağırlık adına kilitlenmemeli.

Navigasyon commit: `43b42a2`. Model araştırması önceki aşamada yapıldı; kullanıcı kararıyla örnek video üzerinden değerlendirme artık ayrı 6. adımdır. Bu aşamada model değiştirilmedi.


### Adım 4 çalışma günlüğü

- `src/core/records.py`: bağımsız Session, VideoAsset, AnalysisRun, MovementEvent, MetricResult, Comparison sözleşmeleri; birim/yöntem, eksik değer ve olay zamanı kontrolleri.
- `src/adapters/analysis_store.py`: SQLite şema v1, ilişkili kayıtlar, tek işlemde olay/metrik/tamamlanma kaydı; orijinal video ve çıktı bütünlük hash'leri, değiştirilemeyen tamamlanmış kayıtlar.
- Veriler `data/analyses.sqlite3` ve `data/analyses/<oturum>/sources/`, `runs/<analiz>/` altında. JSON/CSV/video/pose serileri yerel dosyalarda; SQLite sonuç ve kaynak ilişkilerini tutar.
- Mevcut taekwondo çift analiz akışı kalıcı kayda bağlandı. Ortak geçmiş bileşeninden kayıt açma ve kaynakları koruyarak yeni revizyon oluşturma çalışıyor. Uygulama kapansa da kayıt katalogdan seçilebilir.
- Yarım/başarısız analizler başarı sayılmaz. Kaynaklar ve önceki tamamlanmış sonuç korunur. Kesilen analizlerin durumu geçmiş ekranından açıkça işaretlenir; sırf yeni bağlantı açıldı diye çalışan iş kesilmiş sayılmaz.
- Pose JSONL: her çözülen kare için filtrelenmemiş model noktaları ve görünürlük; nokta şeması/koordinat türü metadata'da. Modelin göreli z çıktısı metre veya gerçek 3D diye sunulmaz.
- ffprobe varsa kaynak best-effort PTS saklanır; yoksa zaman damgası eksik olarak kaydedilir. Kaynak PTS ile mevcut nominal FPS hesap zamanı ayrı alanlardır. Ağır çekimin gerçek fiziksel zamanı hâlâ doğrulanmış değildir; algoritma bu aşamada PTS'ye geçirilmedi.
- Model ve dedektör SHA256, paket sürümleri, algoritma kaynak hash'leri, protokol sürümü ve tüm analiz parametreleri kaydedilir.
- 31 test geçti: yeni uygulama oturumunda geçmişten açma, branş izolasyonu, başarısız ikinci analiz, revizyon, kaynak/çıktı bozulması, eksik değer/0, PTS ve ham landmark kontrolleri dahil. AST/syntax başarılı.
- `de8dee0`: kalıcı kayıt ve geçmiş; `1584767`: normalize hız birim düzeltmesi.

- Ayrı davranış düzeltmesi: gövde uzunluğu eksik/geçersiz olduğunda normalize ayak hızı None olur; piksel/s fallback kaldırıldı. Algoritma sürümü taekwondo-2. Geçerli ölçekli eski hareket regresyonu korunur.
- Birim düzeltmesi sonrası 33 test geçti; AST ve diff kontrolleri temiz.

### Adım 4 kapanış kontrolü

**36 test geçti.** Gerçek kodlu video decode→pipeline→SQLite/dosyalar→yeni bağlantıdan geri yükleme, yeni Streamlit oturumunda kayıt açma, atomik işlem geri alma (birim uyuşmazlığı), ileri şema sürümünü değiştirmeden reddetme de kontrol edildi. Model çalıştırması testlerde taklit edildi; gerçek sporcu/model doğruluğu test edilmedi. AST/syntax ve `git diff --check` temiz. Commitler yerel; push yapılmadı.

Sıralama: **5 ekran → 6 örnek videoda model seçimi → 7 algoritmalar → 8 bağımsız ölçüm doğrulaması → 9 karşılaştırma**. Adım 6 geliştirme videosu, Adım 8 bağımsız doğrulama videosu kullanır.


### Adım 5 çalışma günlüğü

- Tek video yükleme, CMJ/asimetri/sprint seçimi, sporcu adı/kodu ve kare aralığı eklendi. Kaynak video modelsiz okunur; çözünürlük, nominal FPS, okunan kare sayısı ve PTS bilgisi tutulur.
- Ortak `frame_inspector` kaynak kareyi sıralı decode ile gösterir; referans karesindeki sporcu kutusu ve mesafe çizgisi kaydedildikten sonra üstüne çizilir. Koordinatlar kaynak görüntünün sol üstüne göre piksel cinsindedir. Kutuyla seçim otomatik takip değildir.
- Elle tekrar listesi, ilk tamamen havada kare / ilk temas karesi ve tekrar başlangıç/bitişi düzenlenebilir. Sprintte yalnızca başlangıç/bitiş kullanılır. Aralık sınırları, sıralama, çakışma, pozitif mesafe ve görüntü içi referanslar backend'de denetlenir.
- Kare PTS bilgisi uygun olduğunda manuel tekrar zaman çizelgesi video oynatma zamanını gösterir. PTS yoksa zaman uydurulmaz; kare listesi ve kare incelemesi çalışır.
- Kayıt türü `manual_video_review`; `model=None`, `metrics=[]`. SQLite run tamamlanması yalnızca inceleme belgesinin kaydedildiğini belirtir; UI bunu performans analizi diye göstermez. Düzeltmeler yeni oturum revizyonudur, orijinal içerik korunur.
- Kamera/çekim yönü, ayak görünürlüğü, fiziksel zaman kontrolü kullanıcı beyanıdır. Kalibrasyon iki nokta + metre + düzlem açıklamasıdır; perspektif düzeltmesi/gerçek hız hesabı yapılmaz.
- 41 test geçti: kayıt/revizyon/yeniden açma, UI kaydetme, branş izolasyonu, gerçek kaynak kareler, bozuk video, geçersiz aralık/kalibrasyon, eksik zamanlama. Etiket ve metadata kontrolleri tamamlandı.


### Adım 5 kapanışı

- `ce066d5`: voleybol inceleme, manuel tekrar işaretleri ve revizyon akışı.
- Tam test paketi: **42 test**. Ek UI testi tek video yükleme düğmesini, isteğe bağlı kalibrasyon alanlarını, revizyon kaydını ve sprint seçimini kontrol eder. AST/syntax ve diff kontrolü temiz.
- Test videoları programatik küçük kliplerdir; model çalıştırılmadı ve gerçek ölçüm doğruluğu doğrulanmadı. Gerçek kullanıcı videoları değiştirilmedi.
- UI'da kareler sıfırdan numaralanır. Kalkış etiketi iki ayağın da havada olduğu ilk kare, iniş etiketi ilk temas karesidir. Süre/yükseklik hesabında bu konvansiyonun belirsizliği sonraki aşamada ayrıca ele alınmalı.
- Uzun videoda sıralı kare erişimi yavaş olabilir. Revizyonlar kaynak kopyası tutar; disk tekilleştirme kapsam dışıdır. Kalibrasyon iki noktalı taslaktır; otomatik sporcu takibi veya perspektif çözümü yoktur.
- Sıradaki aşama artık örnek video gerektirir. Model/adaptör henüz değiştirilmedi. Commitler yereldir; push yapılmadı.


### Adım 6 — kullanıcı örneği, ilk çalıştırma

- Kullanıcı `How to jump 70% higher in volleyball with this penultimate step - David Seybering (1080p).mp4` dosyasını verdi. Video başlığı/içeriğindeki iddialar talimat veya doğruluk kanıtı kabul edilmedi.
- Kaynak inceleme geçmişine alındı: `2ef0dc9142574f36999e255655b471b4`; yaklaşmalı sıçrama açıklaması eklenen revizyon `2e8afcf7c1894dcd8f35c4528209e061`.
- Örnek yerinde CMJ olmadığı için inceleme testlerine `Yaklaşmalı sıçrama (inceleme)` eklendi. Bu algoritma eklemesi değildir; CMJ ile karışmasını önler.
- Üç aday aynı 204 örnek karede çalıştırıldı; yalnızca MediaPipe zaman içinde izleme davranışını korumak için aradaki kareleri de gördü. Model kesimlerinde reset uygulanmadı; bu deney mevcut davranışı da kapsar.
- İzole araştırma bağımlılıkları `data/model_review/deps/`; uygulama requirements ve çalışan motor değiştirilmedi. Modeller/ham sonuçlar Git dışında.
- `cli/benchmark_pose.py` kaynak SHA256 eşleşen kare listesiyle yeniden çalıştırılabilir; inference, piksel koordinatları ve sürümler saklanır.
- 42 mevcut test geçti; gerçek üç model çalıştırması bundan ayrı deneydir. AST/syntax kontrolü başarılı.
- Sıradaki iş: görsel değerlendirmeyi ve koşullu ilk model kararını kaynak/ayar/sonuçlarla raporla; hız kıyasını yalnızca ayrı seri koşulardan al.


### Adım 6 kapanışı

- Ayrıntılı sonuç ve yeniden çalıştırma: `docs/model-selection.md`. Görseller/ham noktalar/hash'ler: `data/model_review/` (yerel, Git dışında).
- Gerçek üç model 204 ortak örnek karede çalıştırıldı. Poz yok: MediaPipe 4, YOLO26x 1, RTMPose-L 0. Bu sayılar doğruluk yüzdesi değildir.
- İlk aday RTMPose-L WholeBody seçildi; ayak noktaları ve bu örnekte takip devamlılığı gerekçesiyle. Üretim adaptörü henüz eklenmedi; uygulama requirements değişmedi.
- Ayrı sıralı 15 karelik hız koşuları yapıldı; sonuçlar raporda. İlk paralel koşu süreleri kıyaslanmadı. Model ağırlık hash'leri `summary.json` içinde.
- Temas çevresinde görsel belirsizlik aralıkları kaydedildi. Uzman landmark ground truth'u olmadığı için piksel/temas MAE ve fiziksel ölçüm hatası hesaplanmadı; bu doğrulama aşaması tamamlanmış değildir.
- Kod testleri: 42 test geçti. Gerçek model çalıştırmaları ayrı deneydir. `e7986f6` deney komutu ve yaklaşmalı inceleme etiketini ekler. AST/diff temiz; commitler yereldir.


### Adım 7 — motor ve hesaplar

- `src/core/pose.py`: modelden bağımsız piksel koordinatları + skor + kaynak zamanları. Eksik noktalar doldurulmaz.
- `src/adapters/rtmpose_pose.py`: seçilen YOLOX-m / RTMPose-L WholeBody ağırlıkları SHA256 ile doğrulanır. 133 noktalı şemadan gereken 14 omuz/kalça/diz/ayak noktası açık adlarla saklanır. Taekwondo MediaPipe akışı korunur.
- Voleybol: `signals.py` izleme/aday olay yardımcıları, `measurements.py` deneysel hesaplar, `pipeline.py` frame orkestrasyonu. Algoritma sürümü `volleyball-1`.
- CMJ: durağan başlangıç ayak referansına göre otomatik aday; manuel temas ve protokol onayı olmadan yükseklik yok. Uçuş süresi sınır aralığı ve g·t²/8 tahmini; diz/ayak bileği duruşu ve pelvis/ayak ilişkisi kontrolleri. Alt/üst değerler yalnızca kare belirsizliği, istatistiksel güven aralığı değil.
- Asimetri: görüntü düzleminde gövdeye normalize pelvis sapması, gövde/pelvis eğimi; ayrı manuel sağ/sol temas varsa zaman farkı. Kuvvet/sağlık çıkarımı yapılmaz.
- Sprint: dik kamera + aynı pelvis hareket düzleminde iki noktalı mesafe referansı; konumdan 0,20 s merkezli yerel doğrusal uyum ile hız. Kenarlarda eksik değer, boşluk doldurma yok; yüksek uyum artığı olan pencere reddedilir.
- Backend saf matematik/kalite testleri: 9 test geçti. Geçerli sıçrama süresi, ağır çekim çarpanı, VFR doğrusal hız, asimetri, yanlış düzlem, eksik nokta, farklı iniş duruşu, belirsiz takip kontrol edildi.
- Gerçek model entegrasyonu: kullanıcı videosu 55–85 karelerinde 31 kare işlendi; kayıt `d6f3f72d6cc4457f9cc57a529da952fb`, rapor `data/model_review/step7_integration.json`. Yaklaşmalı protokol reddedildi; CMJ metriği üretilmedi. Bu gerçek ölçüm doğrulaması değildir.
- UI hesap düğmesi, aday aktarımı ve kalıcı sonuç sunumu son kontrollerde. Çalışma ağacında toplam 54 test geçti; kapanışta tekrar güncellenecek.
