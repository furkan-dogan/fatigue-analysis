# İlk model seçimi — Seybering örneği

**Karar: voleybol geliştirmesinde ilk bütün-vücut pose adayı RTMPose-L WholeBody.** Bu karar tek örnek üzerindeki takip/ayak noktası uygunluğuna dayanır; ölçüm doğruluğu onayı değildir. Taekwondo'nun çalışan MediaPipe motoru değiştirilmedi. YOLO26x-Pose karşılaştırma adayı olarak tutulacak.

## Video ve kapsam

Kullanıcı kaynağı: `How to jump 70% higher in volleyball with this penultimate step - David Seybering (1080p).mp4`.

- 1080 × 1920, H.264, 893 kare; video akışı 14.898 s, ses dahil konteyner 14.954 s.
- Nominal FPS 60000/1001 ≈ 59.94. 893 PTS okundu; ardışık oynatma zaman farkları yaklaşık 0.016683–0.016684 s.
- SHA256: `ddbd5f1c76a4f5b88c2d7413d18ae7ebf26eb6a291d47d6d7370b7f769a571b5`.
- Görsel olarak beş seviyenin birleştirildiği yaklaşmalı sıçrama gösterimi. Kamera kadrajı/sahne değişimleri var; örnek yerinde CMJ değildir. Otomatik piksel değişim kontrolü 295 ve 711 karelerini işaretledi; bu tüm kurgu kesimlerinin tespit edildiği anlamına gelmez.
- Fiziksel çekim hızı, ağır çekim/hızlandırma geçmişi ve mesafe kalibrasyonu bilinmiyor. Videonun başlığındaki yüzde iddiası doğrulanmadı; cm, km/saat veya performans artışı hesaplanmadı.

## Deney

204 ortak kare: `range(0,893,6)` ve ilk sıçrama çevresinde `range(55,121)` birleşimi. MediaPipe mevcut takip davranışını korumak için aradaki kareleri de işledi; diğer iki yöntem örnek kareleri bağımsız işledi. MediaPipe kurgu geçişlerinde sıfırlanmadı. Bu nedenle deney tamamen aynı zamansal bağlama sahip algoritmaların kontrollü akademik karşılaştırması değildir.

| Aday / ayar | Poz üretilemeyen örnek kareler | Ayak noktaları |
| --- | --- | --- |
| MediaPipe 0.10.21, Full, complexity=1, video modu | 300, 504, 510, 792 — 4/204 | Ayak bileği, topuk, ayak ucu |
| Ultralytics 8.4.48 / Torch 2.11.0, YOLO26x-Pose, 640, conf=0.25 | 558 — 1/204 | Standart 17 nokta; topuk/ayak ucu yok |
| RTMLib 0.0.16 / ONNX Runtime 1.30.0, YOLOX-m + RTMPose-L/DWPose WholeBody 384×288 | Yok — 0/204 | 133 nokta; topuk ve ayak parmak referansları |

**Poz üretmek doğru nokta üretmek değildir.** Model confidence/visibility değerleri birbirine eşdeğer değildir; bunlardan doğruluk yüzdesi hesaplanmadı. Ortak eşikli confidence sıralaması yapılmadı.

RTMPose checkpoint: `rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.onnx`; bu RTMW değildir. Güncel Wholebody varsayılanına güvenmek yerine pose ağırlığı açıkça seçildi. Detector: `yolox_m_8xb8-300e_humanart-c2c7a14a.onnx`. Model arayüzü: [RTMLib resmi README](https://github.com/Tau-J/rtmlib/blob/main/README.md). Standart YOLO noktaları: [Ultralytics Pose](https://docs.ultralytics.com/tasks/pose/).

## Görsel kontrol ve kararın sınırı

[Genel görünüm](../data/model_review/overview.jpg), [ortak karelerde üç model](../data/model_review/comparison.jpg), [ilk kalkış/iniş yakın planı](../data/model_review/contacts.jpg).

60, 75, 90 ve 110. karelerde alt gövde/ayak bindirmeleri incelendi. Üçü de vücut yapısını genel olarak yakalıyor. RTMPose'un ayak referansları ayakkabı bölgeleri üzerinde kullanılabilir adaylar veriyor. YOLO ayak bileğinde bitiyor; bu çıktıyla ayak ucu/zemin temasını doğrudan izleyemeyiz. MediaPipe bazı kurgu/örtüşme anlarında poz kaybediyor; sahne sıfırlamasıyla ayrıca denenebilir.

İlk kalkış için 73–76 ve iniş için 113–116 kareleri **belirsizlik aralığı** olarak görsel incelemeyle not edildi. Bunlar AI tarafından yapılan ön incelemedir; uzman veya bağımsız ground truth değildir. Bulanıklık ve iki ayağın örtüşmesi nedeniyle tek kesin temas karesi atamadım. Anatomik nokta ground truth'u bulunmadığından piksel MAE, açı hatası veya temas tespit doğruluğu raporlanmadı.

RTMPose ilk aday seçildi çünkü örnekte kesintisiz poz üretti ve hedeflenen temas incelemesi için gereken ayak noktalarını sağlıyor. Bu, tüm sporcular veya tüm testlerde en iyi olduğu anlamına gelmez. Bağımsız, elle işaretlenmiş verilerde karar değişebilir.

## Yerel işlem maliyeti

İlk model koşuları kısmen eşzamanlıydı; onların süreleri karşılaştırmada kullanılmadı. Ardından her model ayrı süreçte, sıralı olarak, aynı 15 örnek karede tekrar ölçüldü. Apple M1 Pro / macOS arm64, CPU çalıştırma; warm-up, dosya decode ve model yükleme inference süresinin dışında.

| Aday | Medyan inference / örnek kare | Sürecin tepe RSS'i |
| --- | ---: | ---: |
| MediaPipe | 16.7 ms | 360 MiB |
| YOLO26x-Pose | 197.9 ms | 836 MiB |
| YOLOX-m + RTMPose-L | 372.2 ms | 631 MiB |

Bunlar küçük örnekli yerel ölçümlerdir. Tepe RSS tüm süreci içerir; yalnızca model ağırlığı belleği değildir. Motor/iş parçacığı ayarları aynı değildir (YOLO Torch 4 thread; diğerleri kendi varsayılanları). Gerçek zaman hızı veya nihai uygulama uçtan uca süresi olarak sunulmaz. Çevrimdışı öncelik nedeniyle RTMPose'un daha uzun süresi ilk değerlendirmede kabul edilebilir; toplu kullanım öncesi ayrıca profillenecek.

## Kayıtlar ve yeniden çalıştırma

Voleybol geçmişindeki inceleme revizyonu: `2e8afcf7c1894dcd8f35c4528209e061`. `Yaklaşmalı sıçrama (inceleme)` olarak işaretlendi; metrik listesi boş.

Yerel `data/model_review/`: kaynak kayıt bağlantısı, `samples.json`, `manual_observations.json`, her modelin nokta/süre JSON'u, `summary.json` ağırlık hash'leri, görseller ve `speed/` seri koşuları. Büyük dosyalar Git dışında tutulur; başka makineye devam için bu klasör de taşınmalıdır.

Araştırma bağımlılıkları uygulama requirements dosyasına eklenmedi. Bu çalıştırmada mevcut venv'deki Ultralytics/Torch kullanıldı; RTMLib, ONNX Runtime ve tqdm `data/model_review/deps/` altına kuruldu. RTMLib ağırlıkları kullanıcı önbelleğinde, YOLO ağırlığı deney klasöründe.

```sh
PYTHONPATH=data/model_review/deps .venv/bin/python -m cli.benchmark_pose \
  --source '/tam/yol/video.mp4' \
  --samples data/model_review/samples.json \
  --output data/model_review \
  --model rtmpose
```

Diğer adaylar: `--model mediapipe` ve `--model yolo26x`. Kaynağın SHA256'sı örnekleme listesiyle eşleşmek zorunda. Süre karşılaştırmaları modeller arka arkaya çalıştırılarak yapılmalı. Yeni kaynak için yeni örnekleme listesi oluşturulmalı.

## Sonraki uygulama

1. RTMPose için ortak veri sözleşmesine uyan bir adaptör ekle; ayak noktalarını açık isimlerle eşleştir, olmayan noktaları uydurma. Mevcut MediaPipe karşılaştırma seçeneği olarak kalsın.
2. Sahne kesimlerini analiz aralığı dışında bırak; kullanıcı seçtiği sporcu ve tek kesintisiz tekrar üzerinde çalış.
3. Önce kontrollü CMJ algoritmasını geliştir. Bu yaklaşmalı örneği CMJ doğrulama kaydı gibi kullanma; yaklaşmalı sıçrama ayrı protokoldür.
4. Bağımsız videolarda uzman işaretleri ve uygun fiziksel referanslarla piksel/zaman/cm/hız hatasını ölç. O aşamaya kadar model ve metrikler doğrulanmamış durumdadır.


### Adım 7 uygulama durumu

RTMPose-L WholeBody adaptörü ve deneysel hesaplar uygulamaya eklendi; MediaPipe taekwondoda kaldı. Kullanıcı videosunun 55–85 aralığında 31 kareyle gerçek adaptör/kayıt çalıştırması yapıldı. Yaklaşmalı protokol için CMJ sonucu üretilmedi. Bu entegrasyon kontrolü fiziksel doğruluk kanıtı değildir; koşullu model kararı değişmedi. Sonraki iş bağımsız doğrulama (Adım 8).
