# Deneysel voleybol ölçümleri — volleyball-1

Adım 7 yazılım uygulamasıdır. Fiziksel ölçüm doğruluğu henüz bağımsız videolarda doğrulanmadı; sayısal sonuçların kalite alanı `unvalidated` olur. Başarılı analiz kaydı, doğru ölçüm sertifikası değildir.

## Kayıt ve zaman

Tek sporcu, sabit kamera ve kesintisiz çekim seçin. İnceleme aralığını ve gerekiyorsa ilk karede sporcu kutusunu kaydedin. Kaynak kare PTS'leri eksiksiz ve artan olmalı. Fiziksel zaman çarpanı, oynatma saniyesini gerçek saniyeye çevirir: dört kat yavaşlatılmış video için 0,25; gerçek zaman için 1. Kaydın nasıl üretildiği bilinmiyorsa onay vermeyin. Nominal FPS fiziksel zamanın yerine kullanılmaz.

Kaydedilmiş ayarlarla analiz çalıştırılır. Otomatik adaylar yalnızca başlangıç önerisidir: yeni inceleme revizyonuna aktarın, kaynak karelerden temasları kontrol edin, protokol onaylarını kaydedip yeniden analiz edin. Kare numaraları sıfırdan başlar. Model skorları ölçüm güven yüzdesi değildir.

## CMJ

Yerinde çift ayak sıçrama, yandan çekim ve net ayak görüntüsü gerekir. Yaklaşmalı smaç ve blok ayrı protokollerdir; bu sürüm bunlardan CMJ yüksekliği üretmez. Kalkış işareti iki ayağın da havada olduğu ilk kare, iniş ilk temas karesidir. Tekrar başlangıcı kalkıştan önceki kareyi içermeli.

Kalkış a, iniş b ise uçuş süresinin örnekleme sınırları t[b−1]−t[a] ve t[b]−t[a−1] olur. İki sürenin ortalamasından h = 9,80665 × t² / 8 hesaplanır ve santimetreye çevrilir. Bu kalkıştan tepeye yükselme tahminidir; el erişimi veya ayakta duruştan yükselme değildir. Alt/üst sonuçlar yalnızca kare örnekleme aralığıdır; model, işaretleme, duruş ve zaman ölçeği hatalarını kapsayan güven aralığı değildir.

Kalkış/inişte benzer duruş kullanıcı tarafından doğrulanmalı; ayrıca görünür diz/ayak bileği açıları ve pelvis–ayak ilişkisi denetlenir. Bu kontroller gerçek 3D kütle merkezi yüksekliği eşitliğini kanıtlamaz. Uçuş süresi yönteminin duruşa duyarlılığı için [hakemli yöntem çalışması](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368081/).

## Görüntü düzleminde asimetri

Düz tutulmuş ön/arka kamerayla pelvis orta noktasının başlangıca göre en büyük yatay sapması başlangıç gövde uzunluğunun yüzdesi olarak verilir. Gövde eğimi omuz–pelvis ekseninin düşeye açısı, pelvis eğimi iki kalça ekseninin yataya açısıdır; sonuçlar en büyük mutlak açıdır.

Sağ/sol ilk temas kareleri ayrı elle işaretlenip onaylanırsa sağ eksi sol temas zamanı milisaniye olarak gösterilir. Pozitif değer sağ ayağın daha geç temas ettiğini belirtir. Bu görsel ölçümler kuvvet farkı, zayıf bacak veya yaralanma riski teşhisi değildir; gerçek 3D yönelim hesaplanmaz.

## Sprint

Sabit yandan, hareket düzlemine dik kamera gerekir. Bilinen mesafenin iki görüntü noktası pelvisin hareket düzlemine uygun olmalı; zemin referansını havadaki kalçaya doğrudan uygulamayın. Perspektif/homografi çözümü yoktur. Pelvis yolu referansın uçları arasında kalmalı.

Pelvis orta noktası referans eksenine izdüşürülerek metreye çevrilir. Her zaman noktasında ±0,10 saniyelik yerel doğrusal uyumun eğimi hızdır. Grafik 0,20 saniye pencereli hız ve bunun tepe değerini gösterir; tek kare anlık hızı veya gerçek kütle merkezi hızı değildir. Kenarlarda ve geçersiz pencerelerde boş değer bırakılır.

## Deneysel ret eşikleri

Bu eşikler mühendislik başlangıç ayarlarıdır; bilimsel olarak doğrulanmış kabul sınırları değildir. Adım 8'de hata/ret oranı birlikte ölçülmeli; doğrulama videolarına bakarak eşik ayarlanıp aynı videoda başarı ilan edilmemeli.

| Kontrol | Bu sürüm |
| --- | --- |
| Nokta skoru / gövde ölçeği | En az 0,5 / 10 piksel |
| Zaman örneklemesi | En az 5 kare, fiziksel kare aralığı en çok 0,05 s |
| Takip | Tek uygun sporcu; ardışık pelvis mesafesi önceki gövdenin en çok 0,5 katı |
| Olası kesim | 108×192 küçültülmüş karelerin ortalama mutlak piksel farkı >18 ise aralık reddi |
| Aday başlangıcı | İlk 0,20 s içinde en az 5 örnek; ayak düşey değişimi gövdenin en çok %4'ü |
| Aday uçuş | İki ayağın başlangıca göre yükselmesi gövdenin %2,5'inden fazla; 0,1–1,2 s |
| CMJ görünür hareket | Pelvis ve iki ayak en az 2 ve pencerenin yaklaşık %20'si kadar karede gövdenin %2,5'inden fazla yükselmeli |
| CMJ duruş | Yatay pelvis değişimi ≤%20 gövde; diz/ayak bileği açı farkı ≤15°; pelvis–ayak yükseklik farkı ≤%8 gövde |
| CMJ süre sınırları | 0,1–1,2 s |
| Asimetri görünümü | Kalçalar arası yatay mesafe ≥%10 gövde |
| Sprint referansı | En az 20 piksel; çizgiye dik pelvis uzaklığı ≤referans uzunluğunun %10'u |
| Sprint hız penceresi | En az 5 örnek, iki tarafta tam 0,10 s; en büyük uyum artığı ≤kalibre mesafenin %1'i |

Eksik nokta doldurma veya hız kenarlarında extrapolasyon yoktur. Belirsiz takip/kesim tüm seçili aralığı reddeder. Kamera ve protokol onayları kullanıcı beyanıdır; kesim kontrolü tüm kurgu veya kamera hareketlerini yakalayamaz. Uyum artığı filtresi gerçek yüksek ivmeli hareketi de reddedebilir.

## Doğrulama ve devam

57 kod testi; bilinen sentetik sıçrama, zaman ölçeği, değişken FPS hız, asimetri, ret koşulları, kalıcı kayıt, başarısız koşu ve yeni UI oturumunda açma kontrol edildi. Bunlar gerçek sporcu doğruluğu değildir.

Kullanıcının yaklaşmalı sıçrama videosunda 31 kare gerçek RTMPose çalıştırıldı; kaynak/pose/sonuç kaydı açıldı ve protokol reddi doğrulandı. Santimetre veya hız doğruluğu ölçülmedi.

Adım 8: ayrı CMJ, ön/arka asimetri ve kalibre sprint çekimleri; bağımsız temas işaretleri ve uygun fiziksel referanslar; zaman/cm/açı/hız hataları, takip kopması ve ret oranları. Kabul sınırları kullanım amacıyla önceden belirlenecek. Adım 9: yalnızca uyumlu protokol/model/yöntem kayıtlarında önce–sonra karşılaştırması ve rapor.


## Otomatik aday tespiti (ayrı aşama)

`volleyball-discovery-1` CMJ hesap onaylarından bağımsız tam video taramasıdır. Görünür ayak/pelvis yükselmesini ve bacak hareketiyle birlikte yatay ilerlemeyi adaylaştırır. Eğitilmiş eylem sınıflandırıcısı değildir; yürüyüş/koşu ayrımı ve fiziksel temas doğrulanmadı. Beş karelik medyan yalnızca kesintisiz gözlenen noktalarda kullanılır; eksik kare doldurulmaz. Yaklaşma sınıfı görüntüde yatay yer değiştirmeye bağlıdır, derinliğe doğru yaklaşmayı kaçırabilir.

Başlangıç eşikleri: ayak yükselmesi %12 gövde, tepe %30; pelvis yükselip geri dönmesi %12 gövde. Oynatma zamanında en az 0,08 s / 3 kare, en çok 2,5 s uçuş adayı. Yerel ayak referansı ±1,2 s penceredeki %90 kuantildir. Yaklaşma için önceki 0,65 s içinde yatay pelvis ilerlemesi >%45 gövde; <%15 ise yerinde adayı, arası genel sıçrama adayı. Koşu/yer değiştirme için ±0,2 s pencerede yatay ilerleme >%30 gövde ve ayak bileği yatay ayrımı değişimi >%30 gövde aranır. Bunlar doğrulanmamış geliştirme eşikleridir.

Algoritma bu kullanıcı örneğinde geliştirilip incelendi; bağımsız doğrulama sayılmaz. Güncel gerçek çalıştırma ve devam notları upgrade.md içindedir. Otomatik aday karelerini doğrudan doğrulanmış uçuş süresi hesabına vermeyin.
