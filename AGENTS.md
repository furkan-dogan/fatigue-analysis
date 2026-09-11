# Geliştirici ve AI çalışma notları

- Önce `upgrade.md`, `git status --short` ve son commitleri oku. Sıradaki işi oradaki güncel durumdan al.
- Kullanıcı önceliği: proje yapısı → ortak bileşenler → branş navigasyonu → kayıt → voleybol → video doğrulama → karşılaştırma.
- Her anlamlı committe `upgrade.md` durumunu, yapılan kontrolleri ve sonraki somut işi güncelle.
- Tamamlanmayan adımları tamamlandı işaretleme. Kod testi ile ölçüm doğrulamasını ayır.
- Mevcut kullanıcı değişikliklerini incelemeden silme. Küçük, anlamlı ve Türkçe commit açıklamaları kullan.
- Şimdilik Python/Streamlit ve yerel tek kullanıcı akışını koru. Canlı analiz/çok kullanıcı altyapısı ekleme.
- UI Türkçe. Ortak çekirdek UI/adaptör/branş bilmez; ortak bileşenler branş bilmez; branşlar birbirini import etmez.
- Taşıma ve davranış düzeltmelerini ayrı tut. Yalnızca video/görüntü analizi geliştir; cihaz entegrasyonu veya simülasyonu ekleme.
- Python değişikliklerinde AST/syntax kontrolü; işlevsel taşımalarda `.venv/bin/python -m unittest discover -v` çalıştır.
- Video/model doğruluğu için sahte yüzde verme. Yeterli çekim/kalibrasyon yoksa metrik üretmemek geçerli sonuçtur.
