# Mimari sınırlar

`app.py → ui/app.py → ui/sports/<branş>/page.py`

Sayfa ortak bileşenleri birleştirir. `session.py` yükleme ve geçici oturumu yönetir; `views.py` hesaplanmış sonuçları gösterir; `report.py` dışa aktarmayı sunar. Rapor hesapları `src/sports/<branş>/reporting.py` içinde hazırlanır.

`src/sports/<branş>/pipeline.py → src/core + src/adapters`

Çekirdek UI, adaptör veya branş import etmez. Adaptörler branş bilmez. Ortak bileşenler branş bilmez. Branşlar birbirini import etmez. Bu sınırlar `tests/test_architecture.py` ile denetlenir.

Yeni branşta yalnızca gerçekten ihtiyaç olan dosyalar eklenir. Voleybol ve basketbol alanları henüz yer tutucudur. Kullanılmayan model, cihaz veya servis altyapısı eklenmez.

Ortak bileşenler hazır/boş/yükleniyor/hata durumlarını destekler. Eksik metrik sıfıra çevrilmez; kalite paneli kendiliğinden doğruluk veya başarı oranı üretmez. Video Streamlit üzerinden sunulur; ayrı HTTP sunucusu yoktur.
