# Voleybol öncelikli proje dönüşümü

## Devralan geliştirici için başlangıç

Önce bu dosyayı, `AGENTS.md`, `git status --short` ve son commitleri oku. Kullanıcı yerel düzenleme ve Türkçe küçük commitler istedi; push istenmedi. Python/Streamlit ve tek kullanıcı korunacak. Canlı analiz, çok kullanıcı ve yeni frontend kapsam dışında. Video isteme: kullanıcı 7. adımda verecek.

## Güncel durum — 2026-09-11

**Adım 1, 2 ve 3 tamamlandı. Sırada Adım 4 var.** Ortak bileşenler ve altı bölümlü taekwondo ekranı çalışıyor. Üç branşın ekran girişi var; voleybol varsayılan. Voleybol/basketbol analizleri henüz hazır değil.

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

### 4. Analiz sözleşmeleri ve kalıcı kayıt — bekliyor

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

### 6. Voleybol algoritmaları — bekliyor, kendi içinde sırayla

#### 6A. Dikey sıçrama

- İlk protokol kontrollü, yerinde çift ayak CMJ. Blok ve yaklaşmalı smaç ayrı protokoller olarak sonra.
- Tekrar tespiti, son ayak yerden ayrılması ve ilk ayak yere teması üzerinden uçuş süresi.
- `h = g * t² / 8`: kalkıştan tepeye tahmini yükselme. Ayakta duruştan yükselme ve el erişim yüksekliği farklıdır.
- Kalkış/iniş kütle merkezi yüksekliği benzerliği varsayımı, bacak çekme ve görünmeyen temas durumları kontrol edilmeli.
- Kayıt FPS'i ile ağır çekim oynatma hızı ayrılmalı; gerçek fiziksel zaman doğrulanmadan yükseklik verilmez.
- Önerilen çekim: sabit kamera, iyi ışık, ayaklar net, orijinal 120/240 FPS. Yüksek FPS doğruluk garantisi değildir.
- Otomatik temas aralıkları ve manuel düzeltme birlikte desteklenmeli.

#### 6B. Yana sapma ve iniş asimetrisi

- Önden/arkadan uygun görünümde pelvis orta noktası sapması, gövde/pelvis eğimi, sağ-sol temas zaman farkı.
- Kamera eğimi, vücut yönelimi, örtüşme ve landmark kalitesi kontrol edilmeli.
- Santimetre için hareket düzlemine uygun kalibrasyon; yoksa açıkça normalize ölçüm.
- Görsel asimetri bacak kuvvet farkını kanıtlamaz; güçlü/zayıf bacak teşhisi veya yaralanma riski üretme.
- Sadece gözlenen 2D düzlem hakkında sonuç ver; gerçek 3D ölçüm iddiası yok.

#### 6C. Sprint

- Ölçülmüş düz parkur, sabit yandan kamera, bilinen referanslar, kadrajda tek sporcu.
- Kamera pan/zoom ve kesintiler ilk protokol dışında.
- Ayak ucu hızı sporcu ilerleme hızı değildir. Pelvis takibi kütle merkezi ölçümü diye adlandırılmaz.
- Kalibrasyon takip noktasının hareket düzlemiyle eşleşmeli; zemin homografisi havadaki kalçaya doğrudan uygulanmaz.
- Önce konum/mesafe/geçiş süresi doğrulanır; sonra filtrelenmiş konumdan hız türetilir.
- Anlık hızın zaman penceresi belirtilir; tek gürültülü kare tepe hız sayılmaz.
- Kalibrasyon/zamanlama yetersizse m/s veya km/saat verilmez.

Kabul: Algoritmalar kontrollü/sentetik girdilerle sınanır; gerçek video doğrulaması olmadan doğruluk iddiası yok.

### 7. Örnek videolar ve doğruluk — kullanıcı video verecek

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

### 8. Karşılaştırma ve rapor — bekliyor

- Aynı sporcu/test/yöntem, uyumlu çekim ve kalibrasyon koşulları eşleşir.
- CMJ, blok ve yaklaşmalı smaç tek metrikmiş gibi karıştırılmaz.
- Tekrar ve oturum özetleri, mutlak/yüzde değişim ve kalite bilgisi gösterilir.
- Rapor gözlem ve ölçüm yöntemini açıklar; doğrulanmamış yorgunluk/sağlık çıkarımı yapmaz.
- Dışa aktarma ve önceki kaydı açma uçtan uca kontrol edilir.

## Bilinen sınırlar

- Normalize ayak hızı gövde uzunluğu bulunamayınca piksel/s değerine düşebiliyor. Mevcut ham CSV'de bu eski sorun sürüyor; UI raporunda bu metrik gösterilmiyor. Ölçüm aşamasında birim sözleşmesiyle düzelt.
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
