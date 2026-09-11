# Voleybol öncelikli proje dönüşümü

## Devralan geliştirici için başlangıç

- Kullanıcının önceliği önce dosya yapısı, sonra voleybol; video en son test aşamasında gelecek.
- Kullanıcı dosya düzenleme ve yerel commit atma yetkisi verdi. Push istenmedi.
- Python/Streamlit korunacak. Şimdilik tek kullanıcı, yüklenen video analizi.
- Canlı analiz, React geçişi, çok kullanıcılı altyapı ve servis kuyruğu bu sürümün kapsamında değil.
- Taekwondo'nun mevcut davranışını yapısal taşıma sırasında koru. Ölçüm düzeltmelerini ayrı commitlerde yap.
- Türkçe arayüz; kısa, mevcut geçmişe uygun `refactor: ...`, `fix: ...`, `feat: ...`, `docs: ...` commitleri.
- Her anlamlı aşamada bu dosyayı güncelle: yapılanlar, doğrulama, açık sorunlar ve sıradaki somut iş.
- Başlamadan `git status --short`, `git log -5 --oneline`, bu dosya ve `CLAUDE.md` okunmalı.
- Tamamlanmayan analizleri hazır gösterme; doğruluk test edilmeden doğruluk yüzdesi verme.

## Güncel durum

2026-09-11: Başlangıç incelemesi tamamlandı. Adım 1 üzerinde çalışılıyor.
Mevcut çalışma ağacında kullanıcıdan kalan UI, grafik, rapor ve sentetik sensör değişiklikleri var.
Bu değişiklikler önce ayrı bir başlangıç commitinde korunacak; sonraki yapısal değişikliklerle karıştırılmayacak.
Kökteki `generate_report_doc.py` futbol odaklı bağımsız bir eski rapor aracıdır; silinmeden arşivlenecek.
Henüz voleybol algoritması veya doğrulanmış ölçümü yok.

## Sıralı uygulama planı

### 1. Proje yapısı ve mevcut dosyalar — devam ediyor

- [x] Mevcut çalışmayı incele, syntax kontrolü yap ve ayrı committe koru.
- [x] Genel sayısal araçları ve pose veri modelini branşlardan ayır.
- [x] MediaPipe, çizim, CSV gibi dış sistem bağlantılarını adaptörlere taşı.
- [x] Tekme olayları, tekme metrikleri, yorgunluk ve mevcut pipeline'ı taekwondo modülüne taşı.
- [ ] UI'da uygulama kabuğu, ortak bileşen ve taekwondo ekranlarını ayır.
- [ ] Voleybol ve basketbol için sorumlulukları açıklanmış modül alanları aç.
- [ ] CLI girişlerini ve tüm importları güncelle; kullanılmayan boş eski sayfa klasörünü kaldır.
- [ ] Eski belgeler ve rapor aracını arşivle; aktif README ve CLAUDE haritasını güncelle.
- [ ] Syntax, import, CLI, arayüz açılışı ve davranış regresyon kontrollerini çalıştır.
- [ ] Taşınan dosyalar için bağımlılık sınırlarını test et ve aşamayı commitle.

Kabul: Taekwondo girişleri açılır, mevcut hesapların davranışı korunur, ortak çekirdek UI/branş import etmez.
Bu aşamada yeni ölçüm algoritması geliştirilmez; bilinen ölçüm sorunları aşağıda izlenir.

### 2. Component-first arayüz — bekliyor

- Ortak video oynatıcı, yükleyici, zaman çizelgesi, metrik kartı, kalite paneli ve karşılaştırma paneli.
- Taekwondo'ya özel tekme tabloları/fazları ortak bileşenlerden ayrılmalı.
- Büyük analiz ve rapor ekranları sorumluluklarına göre bölünmeli.
- UI hesaplama yapmamalı; hesaplanmış sonuçları göstermeli. Rapor kuralları ayrı katmana çıkarılmalı.
- Bileşenlerin boş veri, eksik değer, hata ve yükleniyor durumları olmalı.

Kabul: Genel bileşenler taekwondo import etmez; sayfalar bileşenleri birleştirir.

### 3. Branş navigasyonu — bekliyor

- Voleybol, taekwondo, basketbol için ayrı ekran girişleri.
- Voleybol varsayılan öncelik; basketbol geliştirme durumunu açık gösterir.
- Branş değişimi eski sonuçları başka branş altında göstermez; state anahtarları branş/oturumla ayrılır.
- Taekwondo'nun mevcut ekranı erişilebilir kalır.

Kabul: Üç branş seçilebilir; hazır olmayan analizler çalıştırılamaz ve sahte sonuç gösterilmez.

### 4. Analiz sözleşmeleri ve kalıcı kayıt — bekliyor

- VideoAsset, Session, AnalysisRun, MovementEvent, MetricResult ve Comparison modelleri.
- Sonuç: değer, birim, yöntem, kaynak video/zaman aralığı, kalite, protokol/model/algoritma sürümü.
- Eksik metrik sıfır değildir. Bir alan farklı birimler taşıyamaz.
- SQLite: oturum/analiz/sonuç kayıtları. Yerel analiz kimliği klasörleri: video ve büyük zaman serileri.
- Orijinal video korunur; düzeltmeler ayrı revizyon olarak kaydedilir.
- Pose landmarkları, görünürlük ve gerçek zaman damgaları saklanmalı; yalnızca açı CSV'si yeterli değil.
- Sensör kaynağı sonuçla taşınır; sentetik veri gerçek veriyle karışmaz. Voleybol video akışında sensör simülasyonu yok.
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

## Mimari kurallar

- Ortak çekirdek: geometri, sinyal, sayısal yardımcılar, veri tipleri.
- Adaptörler: MediaPipe, OpenCV çizimi, CSV/sensör dosyaları gibi dış bağımlılıklar.
- Branş: test protokolü, olay tespiti, branşa özel metrik/yorum ve orkestrasyon.
- UI: uygulama kabuğu, ortak bileşenler ve branş ekranları. Analiz motoru Streamlit bilmez.
- Branşlar birbirini import etmez; ortak yetenekler daha sonra `src/movements/` altında paylaşılır.
- Erken framework/servis artışı yok. Canlı video için gelecekte zaman damgalı kare kaynağı; çevrimdışı filtrelerin canlı gecikmesi ayrıca doğrulanır.

## Bilinen mevcut sorunlar — yapısal taşımadan ayrı düzeltilecek

1. Taekwondo fatigue skoru değişim yokken veya veri yokken 50 veriyor. Skor anlamı ve eksik veri davranışı yeniden tasarlanmalı.
2. Ayak hızı gövde uzunluğu yokken piksel/s'ye düşüyor, aynı alanda birimler karışıyor.
3. Pipeline sentetik EMG/NIRS üretiyor; rapor gerçek sensör bayrağı ile sentetik özetleri birlikte kullanabiliyor.
4. Video oynatıcı localhost HTTP adresine bağlı; uzak sunucuda uygun değil. Dosya yolu sınırlandırması ve Range doğrulaması da gerekli.
5. Genel visibility skoru ölçüm doğruluğu değildir; metrik bazında gerekli landmark kontrolü eksik.
6. Sabit FPS üzerinden zamanlama ve kayıp noktaların doldurulması hız/olay doğruluğunu etkileyebilir.
7. Mevcut raporda doğrulanmamış sağlık, takviye ve risk yorumları bulunuyor; voleybola taşınmayacak.

## Doğrulama ve çalışma günlüğü

- Başlangıç: 22 Python dosyası AST/syntax kontrolünden geçti; mevcut venv'de Streamlit, MediaPipe, OpenCV importları başarılı.
- `611bd59`: Kullanıcıdan kalan çalışma ve bu plan korundu.
- Taşıma öncesi Streamlit AppTest: boş yükleme ekranı hatasız açıldı.
- Taşıma öncesi iki sentetik tekme içeren sabit regresyon kaydı alındı; olay/faz metrikleri, hız/ivme, yorgunluk ve simülasyon çıktıları karşılaştırılıyor. Bu test bilimsel geçerlilik iddiası değildir; bilinen eski davranışı da korur.
- `.venv/bin/python -m unittest discover -v`: 1 regresyon testi geçti.
- `439b3e9`: Taşıma öncesi regresyon referansı kaydedildi.
- Backend: `src/core/`, `src/adapters/`, `src/sports/{taekwondo,volleyball,basketball}/` ayrıldı. Genel açı ve sinyal hesapları MediaPipe import etmeden kullanılabiliyor.
- Tekme odaklı sensör pencereleme/simülasyon taekwondo altında tutuldu; ortak sensör adaptörüymüş gibi sunulmadı.
- Backend taşıması sonrası sabit regresyon testi ve `main.py --help` geçti. Mimari sınır testleri eklendi.

## Sıradaki somut iş

Adım 1: UI dosyalarını ortak bileşenler ve taekwondo ekranları olarak ayır; örnek CSV yollarını yeni konuma göre düzelt; belge/araç arşivini düzenle.

## Yöntem kaynakları

- Uçuş süresi varsayımları: https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2023.1112739/full
- Video kare hızı ve sıçrama hatası: https://pmc.ncbi.nlm.nih.gov/articles/PMC10108745/
- Düzlem dönüşümü: https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html
