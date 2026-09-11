# Eski Word rapor aracı

`generate_report_doc.py` önceden hazırlanmış futbol performans raporunu üretir.
Voleybol analiz uygulamasının parçası değildir; metindeki iddialar bu dönüşümde doğrulanmadı.
Kullanıcı çalışması kaybolmasın diye korunmuştur.

İsteğe bağlı bağımlılık ve kullanım:

```bash
.venv/bin/pip install -r tools/legacy/requirements.txt
.venv/bin/python tools/legacy/generate_report_doc.py
```

Çıktı `output/reports/Futbol_Performans_Analiz_Sistemi_Rapor.docx` altına yazılır.
Eski kullanıcıya özel masaüstü yolu kaldırıldı. Araç normal uygulama başlatılırken çalışmaz.
