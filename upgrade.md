# Voleybol öncelikli proje dönüşümü

## Devralan geliştirici için başlangıç

Önce bu dosyayı, `AGENTS.md`, `git status --short` ve son commitleri oku. Kullanıcı yerel düzenleme ve Türkçe küçük commitler istedi; push istenmedi. Python/Streamlit ve tek kullanıcı korunacak. Canlı analiz, çok kullanıcı ve yeni frontend kapsam dışında. Şimdi video isteme: kullanıcı model seçimi için 6. adımda örnek video verecek; bağımsız doğrulama 8. adımda.

## Güncel durum — 2026-09-11

**Adım 1, 2 ve 3 tamamlandı. Adım 4 son kontrollerde.** Ortak bileşenler ve altı bölümlü taekwondo ekranı çalışıyor. Üç branşın ekran girişi var; voleybol varsayılan. Voleybol/basketbol analizleri henüz hazır değil.

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
- Branş değişimi eski sonuçları başka branş altında göstermez. Tamamlanmış çift analiz `taekwondo_analysis` altında tutulur; widget anahtarları branşa özeldir. Kalıcı oturum kimlikleri 4. adımda.
- Branş değiştirip dönünce tamamlanmış sonuçlar korunur. Yükleyici widgetları Streamlit yaşam döngüsü gereği yeniden dosya seçimi isteyebilir; mevcut sonuçlar etkilenmez.
- Taekwondo'nun mevcut ekranı erişilebilir kalır.

Kabul: Üç branş seçilebilir; hazır olmayan analizler çalıştırılamaz ve sahte sonuç gösterilmez.

### 4. Analiz sözleşmeleri ve kalıcı kayıt — son kontrollerde

- VideoAsset, Session, AnalysisRun, MovementEvent, MetricResult ve Comparison modelleri.
- Sonuç: değer, birim, yöntem, kaynak video/zaman aralığı, kalite, protokol/model/algoritma sürümü.
- Eksik metrik sıfır değildir. Bir alan farklı birimler taşıyamaz.
- SQLite: oturum/analiz/sonuç kayıtları. Yerel analiz kimliği klasörleri: video ve büyük zaman serileri.
- Orijinal video korunur; düzeltmeler ayrı revizyon olarak kaydedilir.
- Pose landmarkları, görünürlük ve gerçek zaman damgaları saklanmalı; yalnızca açı CSV'si yeterli değil.
- Hatalı/yarım analiz, tekrar çalıştırma ve uygulama yeniden açılınca kayıtları bulma senaryoları.

Kabul: Bir analiz kaydedilip yeniden açılır; kaynağı, birimi ve sürümü izlenebilir.

### 5. Voleybol ekranı — bekliyor

- Tek video yükleme yeterlidir; ikinci video yalnızca karşılaştırmada kullanılır.
- Test seçimi: CMJ/dikey sıçrama, iniş-asimetri, sprint.
- Video metadata ve çekim uygunluğu kontrolü; sporcu/analiz aralığı seçimi.
- Gerekirse kalibrasyon referans noktaları ve mesafeleri girilir.
- Sonuç alanı: video, olay zaman çizelgesi, metrikler, tekrar listesi ve geçmiş.
- Kalkış/iniş kareleri kullanıcı tarafından incelenebilir ve düzeltilebilir olmalı.

Kabul: Yükleme ve inceleme akışı tamamdır; mevcut olmayan hesaplar açıkça belirtilir.

### 6. Örnek video üzerinden model seçimi — bekliyor, ayrı karar aşaması

- Kullanıcı örnek videoyu bu aşamada paylaşacak. Önce dosya/kayıt/voleybol inceleme akışı tamamlanır.
- YOLO26x-Pose yalnızca araştırma adayı; mevcut MediaPipe ve ayak noktaları içeren uygun bir pose modeli aynı karelerde karşılaştırılır.
- Video zamanlaması, çekim açısı, ayakların görünürlüğü ve sporcu hareketi incelenir. Model dosyası/sürümü, çalışma ayarları ve donanım kaydedilir.
- Elle işaretlenmiş ortak noktalar, ayak teması çevresindeki kareler, takip kopmaları, örtüşme, zamansal titreme, çalışma süresi ve bellek karşılaştırılır.
- Algoritmalar henüz hazır değilken sıçrama cm veya sprint hız doğruluğu iddiası üretilmez. İlk seçim noktaların/temas görünümünün uygunluğuna göre yapılır; ölçüm sonucu kararı 8. adımda doğrulanır.
- Gerekli noktaları vermeyen modelde ayak bileği ayak ucu yerine konmaz. Hiçbir aday yeterli değilse sonuç açıkça yetersiz kabul edilir.
- Tek örnek videodan tüm sporcular için en iyi model sonucu çıkarılmaz. Bu video geliştirme verisidir; bağımsız doğrulama videolarından ayrı tutulur.
- Çıktı: aday tablosu, görsel karşılaştırma ve gerekçeli ilk model/adaptör kararı. Canlı analiz modeli ayrıca daha sonra değerlendirilebilir.

Kabul: Örnek video üzerinde kanıtlı ilk seçim yapılır; seçim ölçüm doğruluğu onayı gibi sunulmaz.

### 7. Voleybol algoritmaları — bekliyor, kendi içinde sırayla

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
- Taekwondo yükleme akışı halen iki video ister. Voleybol tek video akışı 5. adımda.
- Arayüz kayıtları geçici; uygulama yeniden açılınca geçmiş kayıt garantisi yok (4. adım).
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

**Adım 4: analiz sözleşmeleri ve yerel kalıcı kayıt.** Önce sonuç/video/oturum modellerini ve SQLite deposunu kur; sonra mevcut taekwondo akışını bağla. Yeniden açma, eksik metrik, yarım/hatalı analiz ve sürüm bilgisi testlerini ekle. Voleybol algoritmalarına geçme; video isteme.

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

Navigasyon commit: `43b42a2`. Bu araştırma belge güncellemesidir; 4. adım başlatılmadı. Sıradaki iş halen kalıcı kayıt ve analiz sözleşmeleridir.


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
- Sıradaki son kontrol: normalize ayak hızındaki eski birim karışmasını ayrı davranış düzeltmesinde gider; kayıt belgelerini güncelle ve aşamayı kapat.

- Ayrı davranış düzeltmesi: gövde uzunluğu eksik/geçersiz olduğunda normalize ayak hızı None olur; piksel/s fallback kaldırıldı. Algoritma sürümü taekwondo-2. Geçerli ölçekli eski hareket regresyonu korunur.
- Birim düzeltmesi sonrası 33 test geçti; AST ve diff kontrolleri temiz.
