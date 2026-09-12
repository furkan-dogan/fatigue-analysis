# Voleybol öncelikli proje dönüşümü

## Devralan geliştirici için başlangıç

Önce bu dosyayı, `AGENTS.md`, `git status --short` ve son commitleri oku. Kullanıcı yerel düzenleme ve Türkçe küçük commitler istedi; push istenmedi. Python/Streamlit, yerel tek kondisyoner ve yalnızca video/görüntü analizi korunacak. Canlı analiz, çok kullanıcı altyapısı ve yeni frontend kapsam dışında. Aşağıdaki **aktif iş sırası** geçerlidir; eski adım numaraları tarihsel kayıttır.

## Güncel durum — 2026-09-12, son kullanıcı kararı

**Toplu yükleme ve kuyruk ertelendi. Tek video yüklenince uçtan uca otomatik analiz önceliklidir.** Kullanıcı minimum ayar ve doğrudan analiz motoruna odaklanılmasını istedi. Yeni kuyruk, çoklu video veya ortak çekim grubu işi yapma.

Tek video girişi geri getirildi; yüklenen kayıt doğrudan açılır. Sporcu eşleştirme tablosu, toplu kayıt listesi ve ortak çekim formu kaldırıldı. Eski kayıtlar korunur. Mevcut düğme yalnızca videoyu açar; otomatik tam analiz henüz hazır değildir.

### Aktif iş sırası

1. **Tek video ve sade giriş — tamamlandı.**
2. **Otomatik video anlama — sıradaki iş:** videonun tamamında çekim bölümleri, sporcu takibi, hareket/tekrar ve kalkış–iniş adayları. Test türü ve kare aralığı normal kullanıcıdan istenmeyecek. Mevcut örnek yaklaşmalı sıçramadır; CMJ reddi tek başına sonuç değildir.
3. **Metrik bazında analiz:** uygun sıçrama/görsel asimetri/koşu ölçümleri; eksik fiziksel zaman veya kalibrasyon yalnızca ilgili ölçümü bekletir. Kullanıcı onaylarını otomatik true yapma.
4. **Tek sonuç ekranı:** işaretli video, hareket listesi, ölçümler ve yalnızca gerektiğinde kısa düzeltme.
5. **Gerçek örneklerde doğrulama:** tespit ve ölçüm hataları ayrı değerlendirilecek; ardından önce–sonra/rapor.

Mevcut model, kayıt ve deneysel hesap altyapısı kullanılacak. Sırf kapsam değişti diye yeniden mimari kurma veya yeni plan belgeleri üretme. Sonraki geliştirme doğrudan otomatik analiz motorudur. Toplu yükleme sonraya bırakıldı; aşağıdaki sekiz adım ve günlükler tarihsel kayıttır.

Kontrol: 59 kod testi; tek dosyanın açılması, tekrar tıklamada kayıt korunması ve bozuk yeni dosyada önceki kaydın kaybolmaması dahil. AST/diff kontrolü yapıldı. Fiziksel doğruluk doğrulanmadı.

## Ertelenen toplu ürün planı — tarihsel

| Adım | Kapsam | Durum / kabul koşulu |
| --- | --- | --- |
| 1 | Plan ve kullanım akışı | Tamamlandı: aşağıdaki kullanıcı akışı, sınırlar ve kabul senaryoları tanımlandı. |
| 2 | Sade toplu yükleme ekranı | Tamamlandı: çoklu video seçimi, sporcu eşleştirme, isteğe bağlı ortak çekim bilgileri ve tek ana işlem. Teknik inceleme ana ekrandan ayrılır. |
| 3 | Analiz kuyruğu ve kayıt | Bekliyor: işler sırayla çalışır; tek hata diğerlerini durdurmaz; tamamlananlar korunur, kesilen işler açıkça yeniden başlatılabilir. |
| 4 | Otomatik video anlama | Bekliyor: çekim bölümleri, sporcu takibi, hareket/tekrar ve faz adayları çıkarılır; belirsiz sınıf bilinmiyor kalır. |
| 5 | Otomatik ölçüm ve gerekli bilgi | Bekliyor: metrik bazında uygunluk; yalnızca gerekli bilgi istenir; düzeltmeler uygun mevcut pose çıktısını kullanır. |
| 6 | Sonuç merkezi ve hızlı kontrol | Bekliyor: sporcu özeti, hareket klipleri, eksik bilgi ve düzeltme akışı; bütün videoyu izleme zorunluluğu yok. |
| 7 | Gerçek kullanım ve doğruluk doğrulaması | Bekliyor: toplu çalışma, müdahale yükü, tespit ve fiziksel ölçüm hataları bağımsız verilerle değerlendirilir. |
| 8 | Önce–sonra karşılaştırma ve rapor | Bekliyor: uyumlu sporcu/protokol/yöntem kayıtlarında karşılaştırma ve toplu rapor. |

Adım 2 ekranı tek başına otomatik analiz tamamlandı diye sunulmayacak. Ana işlem Adım 3'te gerçek kuyruğa, Adım 4–5'te otomatik yorumlamaya bağlanacak. Her ara sürüm mevcut yeteneğini açık gösterecek; sahte ilerleme ve örnek metrik yok.

### Kullanıcı akışı ve ekran sorumlulukları

1. Kondisyoner bir veya birden çok video seçer. Hedef kabul senaryosu yaklaşık 20 sporcunun videolarıdır; 20 eşzamanlı kullanıcı veya video başına tek sporcu varsayımı değildir.
2. Dosya adından ve mevcut kayıtlardan sporcu eşleştirmesi önerilebilir. Belirsiz öneri kesin kimlik olarak kaydedilmez. Bir sporcunun birden fazla videosu olabilir; kimlik eksikliği teknik taramayı durdurmaz, geçmiş karşılaştırması eşleştirme tamamlanana kadar bekler. Yüz tanıma kapsam dışı.
3. İsteğe bağlı çekim grubu seçilir. Aynı oturum etiketi tek başına aynı kamera/ölçek/zaman koşulu sayılmaz. Ortak bilgiler yalnızca açıkça seçilmiş uyumlu videolara uygulanır; video özelindeki farklılık korunur.
4. Tek **Analizi başlat** işlemi seçili videoları kuyruğa alır. Normal akışta test türü, kare aralığı, koordinat, temas veya model seçimi istenmez.
5. Sistem sıralı olarak kaynağı inceler, bölümleri/sporcuyu/hareketleri çıkarır ve desteklenen metrikleri değerlendirir. Çok sporculu belirsiz bölümde görsel seçim istenir; başka sporcuya sessiz geçilmez.
6. Kondisyoner sonuç listesinden yalnızca kontrol gereken kayıtlara gider. Tek, yüksekliği sınırlı video oynatıcı ve hareket listesi bulunur. Kare inceleme, model ayrıntıları ve manuel düzeltme isteğe bağlı açılır.
7. Düzeltme ve ek bilgi otomatik kaydedilir; sürümler arka planda korunur. Aynı çıkarım koşullarında temas/mesafe düzeltmek bütün videonun pose çıkarımını tekrar gerektirmemeli.

Ana ekran: yükleme, sporcu eşleştirme, tek ana işlem ve geçmiş. İşlem ekranı: dosya bazında durum ve gerçek ilerleme. Sonuç ekranı: sporcu/hareket özeti, uygun metrikler ve gerektiğinde tek somut düzeltme eylemi. Büyük ikinci kare görüntüsü ve teknik onay listesi ana akışta bulunmaz.

### Otomasyon ve ölçüm sözleşmesi

- Hareket tespiti fiziksel ölçümden ayrılacak. Zaman ölçeği bilinmese de hareket bölümleri bulunabilir; mesafe yok diye koşu tespiti iptal edilmez.
- İlk sınıflar yerinde sıçrama, yaklaşmalı sıçrama, koşu ve belirsiz/desteklenmeyen hareket. Yaklaşmalı sıçrama CMJ hesabına zorlanmaz; blok/smaç gibi daha özel sınıflar kanıt ve kapsam olmadan atanmaz.
- Sahne kesimi yeni bölüm oluşturur; tek bozuk bölüm bütün videodaki kullanılabilir hareketleri silmez. Kesintisiz tekrarın içine denk gelen sorun ilgili tekrar/metrikte değerlendirilir.
- Her metrik ayrı hesaplanabilirlik ve gerekçe taşır. Fiziksel zaman veya uygun kalibrasyon yoksa ilgili cm/hız sonucu bekler. Görsel gözlemler görünür kalır.
- Kullanıcı onaylarını otomatik true yapmak otomasyon sayılmaz. Sistem gözlemi, kullanıcı bilgisi ve bilinmeyen durum ayrılır; pose skoru ölçüm doğruluk yüzdesi olarak sunulmaz.
- Gerçek zaman/mesafe görüntüden güvenilir çıkarılamıyorsa yalnızca gerekli soru sorulur. Aynı çekim grubu için uygun cevap tekrar kullanılabilir; farklı videoya körlemesine kopyalanmaz.
- İşin tamamlanması ölçümün doğrulanması değildir. Sonuç kartları yöntem ve doğrulama durumunu korur; görsel asimetri bacak kuvveti veya sağlık teşhisi değildir.
- Kaynak video/model/çıkarım ayarları aynıysa mevcut pose kullanılabilir; model veya ilgili çıkarım ayarı değişmişse yeni çıkarım gerekir. Eski sonuç yeni ayara aitmiş gibi gösterilmez.

### Durumlar ve hata davranışı

İşleme durumu: bekliyor → işleniyor → tamamlandı / başarısız / kesildi. Sonuç inceleme durumu ayrıca tutulur: kontrol gerekmiyor / kontrol gerekli / kullanılabilir sonuç yok. Kullanıcı listesinde anlaşılır karşılıkları **Tamamlandı**, **Kontrol gerekli**, **İşlenemedi** olur; sayısal metriklerin doğrulanmış olduğu ima edilmez.

Bozuk video diğer işleri durdurmaz. Modelin hiç yüklenememesi gibi ortak engellerde kuyruk açık gerekçeyle duraklatılmalı; bütün dosyalar tek tek başarısız diye işaretlenmemeli. Yeniden deneme kaynakları ve tamamlanan sonuçları korur. Uygulama kapanınca kalan işler kaybolmaz; ilk sürümde kesilen dosya baştan işlenebilir, kare düzeyinde devam zorunlu değil. Kalıcı iş kaydı ve kaynak tekrar kullanımının mevcut SQLite şemasıyla uyumu Adım 3'te açık migration kararıyla uygulanır.

### Uçtan uca kabul senaryoları

- Kondisyoner 20 dosyayı topluca seçer; normal akışta her dosyaya test ve temas girmeden tek işlemle kuyruğa alır. Kimlik ve çekim bilgisi yalnızca gerektiğinde istenir.
- Karışık içerikte sıçrama/koşu adayları ayrı gösterilir; desteklenmeyen hareket için uydurma ölçüm çıkmaz.
- Bir bozuk video ve bir belirsiz sporcu kaydı, diğer geçerli dosyaların tamamlanmasını engellemez.
- Kurgu içeren kullanıcı örneğinde kullanılabilir yaklaşmalı sıçrama bölümleri ve temas adayları hedeflenir; CMJ reddi tek başına ürün çıktısı sayılmaz. Bulunacak tekrar sayısı önceden başarı gibi yazılmaz.
- Eksik mesafe yalnızca fiziksel hız hesabını bekletir; eksik zaman yalnızca buna bağlı sonuçları engeller.
- Ortak çekim bilgisi farklı kamera veya ağır çekim koşulundaki dosyaya yanlış uygulanmaz.
- Tekrar tespiti düzeltildikten sonra sonuçlar güncellenir; uygun mevcut pose tekrar kullanılır. Yeni oturumda kayıtlar ve iş durumları açılabilir.
- Otomatik çıkarımlar ile manuel düzeltmelerin kaynağı izlenebilir. Normal ekranda kare/koordinat formu yoktur.

Adım 7 değerlendirmesinde: müdahale gereken video oranı, video başına soru/düzeltme sayısı, kondisyonerin aktif işlem süresi, toplam kuyruk süresi, yanlış/kaçırılan tekrarlar, sporcu takip hataları, metrik bazında ret ve fiziksel hata ölçülecek. Sayısal kabul eşikleri bağımsız deneyden önce belirlenecek; şimdi keyfi doğruluk yüzdesi veya süre garantisi yok. Kullanılabilirlik testi ve fiziksel doğrulama ayrı raporlanacak.

### Adım 1 kapanışı — 2026-09-12

- Kullanıcının onayladığı sekiz adım aktif plan yapıldı; eski tamamlanma ifadeleri tarihsel teknik kapsam olarak ayrıldı.
- Tek kondisyoner/toplu video, minimum ayar, istisna inceleme, kısmi sonuç, kimlik/çekim grubu sınırları ve yeniden işleme sözleşmesi yazıldı.
- Bu commit yalnızca dokümantasyon değişikliğidir; uygulama davranışı değişmedi. Kod testleri yeniden çalıştırılmadı; Markdown içeriği ve diff boşluk kontrolü yapıldı.
- Sonraki somut iş: Adım 2. Mevcut ortak yükleyici/geçmiş/video bileşenlerini incele; voleybol sayfasını toplu yükleme ve sade ekran durumlarına ayır. Gereksiz klasör, servis veya yeni frontend ekleme. Eski inceleme kayıtları erişilebilir kalmalı.

### Adım 2 kapanışı — 2026-09-12

- Ortak yükleyici çoklu dosyayı destekler; taekwondo tekli varsayılanını korur. Branşın `uploads.py` bileşeni toplu seçim, isteğe bağlı sporcu adı/kodu ve ortak çekim grubu/notunu sunar.
- Sporcu adı kullanıcı girdisidir; dosya adından kesin kimlik üretilmez. Boş kimlik kaydı engellemez. Grup/not metadatası hiçbir fiziksel zaman/kamera/kalibrasyon onayını değiştirmez; teknik revizyonda korunur.
- Ana düğme **Videoları kaydet**: kaynaklar gerçek inceleme kayıtlarına dönüştürülür. Sıralı kayıt sırasında dosya bazında ilerleme/hata görünür; bir bozuk video diğerlerini engellemez. Bu otomatik analiz veya kalıcı kuyruk değildir.
- Aynı UI oturumunda aynı içerik/ad/eşleştirme/grup ile tekrar basmak başarılı kayıtları kullanır. Oturumlar arası tekilleştirme ve kesilen toplu işin devamı Adım 3 kapsamındadır. Yükleme seçimi değişince eşleştirme tablosu yeniden kurulur; kaydedilmiş metadata korunur.
- Ana ekran tek dar video önizlemesi ve varsa sonuç gösterir. Kare inceleyici ve teknik formlar **Teknik incelemeyi aç** ile oluşturulur; kapalıyken kare decode edilmez. Mevcut kayıtlar ve deneysel tekli analiz erişilebilir.
- 59 test geçti: 20 geçerli + 1 bozuk dosya, tekrar tıklama, yeni bağlantıdan kayıt açma, kimlik/grup/not korunması, teknik ekran açma ve revizyon, branş geçişi ve eski analiz sonuçları dahil. UI yükleme girdisi testte taklit edildi; videolar gerçek küçük sentetik decode girdileridir. Fiziksel doğruluk veya kondisyoner kullanım deneyi değildir.
- Python AST ve diff boşluk kontrolü yapıldı. Uygulamanın gerçek tarayıcıda görsel/kullanılabilirlik değerlendirmesi henüz yapılmadı.
- **Sonraki somut iş Adım 3:** kalıcı toplu iş/iş öğesi kaydı, sıralı analiz yürütme, ortak model engelinde duraklatma ve kesilen işi yeniden başlatma. Mevcut SQLite sözleşmelerini önce incele; gerekirse açık migration ekle. UI oturumundaki kayıt tekrar kullanımını kalıcı iş kimlikleriyle değiştir. Otomatik hareket tanıma Adım 4'te; kuyruğa bağlanan mevcut motorun manuel sınırlarını gizleme.

## Tarihsel teknik plan — 1–9

Bu bölüm ve devamındaki günlükler önceki uygulamayı açıklar. Buradaki “tamamlandı” etiketleri eski teknik kabul koşullarına aittir; aktif sıra yukarıdaki sekiz adımlı ürün planıdır.

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

### 7. Voleybol algoritmaları — tamamlandı (deneysel)

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
- Taekwondo iki video, voleybol incelemesi tek video kullanır. Voleybol manuel inceleme ve ayrı deneysel analiz revizyonları içerir; koşullar yetersizse gerekçeli eksik sonuç üretir.
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

## Tarihsel plandan geçiş

Eski “iki adım kaldı” sıralaması 2026-09-12 kullanıcı kararıyla değişti. Bağımsız doğrulama güncel Adım 7, karşılaştırma güncel Adım 8 kapsamındadır. Önce toplu yükleme, kuyruk, otomatik video anlama ve kondisyoner sonuç akışı tamamlanacak. Devam için dosyanın başındaki güncel ürün planını kullan.

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
- UI hesap düğmesi, adayları inceleme revizyonuna aktarma, tekrar metrikleri ve hız grafiği tamamlandı. Aday aktarımı temas/duruş onaylarını sıfırlar; ayar revizyonu eski sonuçları taşımaz.
- Ek koruma: durağan pozdan işaretlenmiş sahte uçuş reddedilir; tek konum sıçraması hız tepesine dönüşmez.
- Tam paket **57 test geçti**. Olumlu sentetik hesap → olay/metrik kaydı → yeni uygulama oturumunda sayısal sonuç; protokol reddi → boş sonuç; model hatası → başarısız kayıt yolları dahil. Model test taklitleri fiziksel doğrulama değildir.
- Kullanım ve eşikler: `docs/measurement-methods.md`. Ana bağımlılıklara rtmlib 0.0.16, onnxruntime 1.30.0 ve tqdm 4.70.1 eklendi; gerçek CPU çalıştırması yerel venv'de yapıldı.
- Backend checkpoint: `1abbddd`. Adım 7 kod kapsamı tamamlandı; bağımsız gerçek ölçüm doğrulaması Adım 8'de açık. Sonraki AI yukarıdaki “Sıradaki somut iş” bölümünden devam etmeli.

- Kapanış AST kontrolü: 84 Python dosyası başarılı; git diff --check temiz. Commitler yerel; push yapılmadı.
