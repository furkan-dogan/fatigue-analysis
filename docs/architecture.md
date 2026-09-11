# Mimari sınırlar

`app.py → ui/app.py → ui/sports/<branş>/page.py`

Sayfa ortak bileşenleri birleştirir. `session.py` yükleme, kalıcı geçmiş ve aktif ekran oturumunu yönetir; `views.py` hesaplanmış sonuçları gösterir; `report.py` dışa aktarmayı sunar. Rapor hesapları `src/sports/<branş>/reporting.py` içinde hazırlanır.

`src/sports/<branş>/pipeline.py → src/core + src/adapters`

Çekirdek UI, adaptör veya branş import etmez. Adaptörler branş bilmez. Ortak bileşenler branş bilmez. Branşlar birbirini import etmez. Bu sınırlar `tests/test_architecture.py` ile denetlenir.

Yeni branşta yalnızca gerçekten ihtiyaç olan dosyalar eklenir. Voleybol girişi manuel video inceleme ve kalıcı revizyon akışını sunar; basketbol ortak geliştirme durumu bileşenini gösterir. Otomatik voleybol ölçümleri henüz yok. Uygulama kabuğu yalnızca seçili branşı yükler. Tamamlanmış taekwondo analizi widgetlardan ayrı `taekwondo_analysis` state alanındadır; diğer branşlar bu alanı okumaz. Kullanılmayan model, cihaz veya servis altyapısı eklenmez.

Ortak bileşenler hazır/boş/yükleniyor/hata durumlarını destekler. Eksik metrik sıfıra çevrilmez; kalite paneli kendiliğinden doğruluk veya başarı oranı üretmez. Video Streamlit üzerinden sunulur; ayrı HTTP sunucusu yoktur.


## Kalıcı kayıt sınırı

`src/core/records.py` sözleşmelerini `src/adapters/analysis_store.py` SQLite ve yerel dosyalara kaydeder. Adaptör hiçbir branşı import etmez. Branşın `service.py` modülü pipeline sonucunu sözleşmelere çevirir ve geri yükler; UI SQL veya dosya formatı bilmez.

Şema v1: sessions → videos → runs → events → metrics; comparisons tamamlanmış analizlere bağlanır. Metrik kaynağı event → run → video üzerinden izlenir; yöntem, birim, kalite ve eksiklik gerekçesi ayrıca saklanır. Karşılaştırma uyumu bu aşamada `unverified` durumundadır.

Orijinaller ve her çalışmanın çıktıları UUID klasörlerinde ayrıdır. Tamamlanmış kayıtlar güncellenmez; tekrar çalıştırma `revision_of` ile yeni oturum açar. Çıktı JSON'u atomik dosya değişimiyle yazılır; olay/metrik/tamamlanma durumu tek SQLite işlemindedir. Dosya sistemi ve SQLite ortak bir işlem değildir: çökme halinde sahipsiz/yarım dosya kalabilir, fakat tamamlanmamış analiz başarı diye açılmaz. Kayıt açılırken kaynak ve çıktı hash'leri doğrulanır.

Yalnızca mevcut yerel tek kullanıcı akışı desteklenir. Kesilen işler açık bir kurtarma eylemiyle `interrupted` işaretlenir; otomatik bağlantı açılışı çalışan işi değiştirmez. Eski veriler otomatik içe aktarılmaz. İleri şema sürümü veri değiştirilmeden reddedilir; v1 dışındaki geçişler ileride açık migration gerektirir.

Pose JSONL modelin filtrelenmemiş noktalarını/görünürlüğünü saklar. ffprobe best-effort PTS, nominal FPS hesap zamanından ayrıdır; zaman kaynağı, eksik/kare sayısı uyuşmazlığı metadata'dadır. Mevcut hesapların VFR/ağır çekim doğruluğu bu kayıt değişikliğiyle çözülmüş sayılmaz.


## Voleybol incelemesi

`ui/sports/volleyball/page.py` ortak video/kare/kalite/zaman çizelgesi bileşenlerini ve `editor.py` düzenleyicisini birleştirir. `src/sports/volleyball/review.py` test/aralık/tekrar/kalibrasyon kontrollerini; `service.py` kayıt ve revizyonu yönetir. `src/adapters/video_review.py` model çalıştırmadan kaynak kareleri okur.

Kayıt `manual_video_review` türündedir. Bir inceleme run'ının `completed` olması yalnızca belgenin tamamlandığını gösterir; provenance içinde model `None`, sonuçta metrikler boş listedir. Manuel işaretler ilerideki otomatik MovementEvent sonuçlarıyla karıştırılmaz. PTS uygun değilse zaman çizelgesi yerine kare listesi kullanılır.

Kare inceleme sıralı decode yapar; doğru kare seçimi için keyframe seek varsayımı kullanılmaz. Uzun videoda uzak karelere erişim yavaş olabilir. Sekiz kareyle sınırlı UI önbelleği vardır. Yeni revizyon kaynak videoyu koruyarak ayrı kopya oluşturur; disk tekilleştirme henüz uygulanmaz. Kullanıcı beyanı çekim kontrolleri ve iki noktalı mesafe referansı, doğrulanmış hareket düzlemi kalibrasyonu değildir.
