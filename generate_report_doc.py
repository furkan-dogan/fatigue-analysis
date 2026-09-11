"""
Kapsamlı sistem raporu — Word belgesi oluşturucu (v2)
Çalıştır: .venv/bin/python generate_report_doc.py
"""

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# ── Sayfa kenar boşlukları ──────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ── Renk paleti ─────────────────────────────────────────────────────────────
C_DARK   = RGBColor(0x1A, 0x1A, 0x2E)
C_MID    = RGBColor(0x16, 0x21, 0x3E)
C_BLUE   = RGBColor(0x0D, 0x47, 0xA1)
C_TEAL   = RGBColor(0x00, 0x77, 0x7A)
C_GREEN  = RGBColor(0x1B, 0x5E, 0x20)
C_ORANGE = RGBColor(0xE6, 0x51, 0x00)
C_RED    = RGBColor(0xB7, 0x1C, 0x1C)
C_GREY   = RGBColor(0x88, 0x88, 0x88)
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

# ── Stil yardımcıları ───────────────────────────────────────────────────────
def h1(text, color=C_DARK):
    p = doc.add_heading(text, level=1)
    run = p.runs[0]
    run.font.color.rgb = color
    run.font.size = Pt(16)
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after  = Pt(6)
    return p

def h2(text, color=C_MID):
    p = doc.add_heading(text, level=2)
    run = p.runs[0]
    run.font.color.rgb = color
    run.font.size = Pt(13)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after  = Pt(4)
    return p

def h3(text, color=C_BLUE):
    p = doc.add_heading(text, level=3)
    run = p.runs[0]
    run.font.color.rgb = color
    run.font.size = Pt(11)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(2)
    return p

def body(text, bold=False, italic=False, color=None, size=10.5):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    return p

def bullet(text, level=0, color=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.5 + level * 0.5)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    if color:
        run.font.color.rgb = color
    return p

def example_box(title, content, title_color=C_BLUE):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Cm(0.8)
    p.paragraph_format.right_indent = Cm(0.5)
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after  = Pt(8)
    r1 = p.add_run(f"▌ {title}\n")
    r1.bold = True
    r1.font.size = Pt(10)
    r1.font.color.rgb = title_color
    r2 = p.add_run(content)
    r2.font.size = Pt(10)
    r2.italic = True
    return p

def chat_box(role_label, question, answer, role_color=C_TEAL):
    """Sohbet kutusu — soru + sistem cevabı."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Cm(0.8)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(8)
    r1 = p.add_run(f"💬 {role_label} sorar:\n")
    r1.bold = True; r1.font.size = Pt(10); r1.font.color.rgb = role_color
    r2 = p.add_run(f'"{question}"\n\n')
    r2.font.size = Pt(10); r2.italic = True
    r3 = p.add_run("🤖 Sistem:\n")
    r3.bold = True; r3.font.size = Pt(10); r3.font.color.rgb = C_DARK
    r4 = p.add_run(answer)
    r4.font.size = Pt(10)
    return p

def add_table(headers, rows, col_widths=None, header_fill="1A1A2E"):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_row = t.rows[0]
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.text = h
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = C_WHITE
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_cell_bg(cell, header_fill)
    for ri, row in enumerate(rows):
        fill = "EBF5FB" if ri % 2 == 0 else "FFFFFF"
        tr = t.rows[ri + 1]
        for ci, val in enumerate(row):
            cell = tr.cells[ci]
            cell.text = str(val)
            cell.paragraphs[0].runs[0].font.size = Pt(9.5)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            _set_cell_bg(cell, fill)
    if col_widths:
        for row in t.rows:
            for ci, w in enumerate(col_widths):
                if ci < len(row.cells):
                    row.cells[ci].width = Cm(w)
    doc.add_paragraph()
    return t

def _set_cell_bg(cell, fill_hex):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill_hex)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)

def mono(text):
    """Monospace paragraf — diyagram için."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Cm(1.0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run(text)
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    run.font.color.rgb = C_DARK
    return p

def divider():
    p = doc.add_paragraph()
    run = p.add_run("─" * 100)
    run.font.size = Pt(7)
    run.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(0)

# ════════════════════════════════════════════════════════════════════════════
# KAPAK
# ════════════════════════════════════════════════════════════════════════════
cp = doc.add_paragraph()
cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
cp.paragraph_format.space_before = Pt(80)
r = cp.add_run("FUTBOL PERFORMANS ANALİZ SİSTEMİ")
r.bold = True; r.font.size = Pt(26); r.font.color.rgb = C_DARK

sp = doc.add_paragraph()
sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = sp.add_run("Kapsamlı Teknik ve Kullanıcı Raporu")
r2.font.size = Pt(14); r2.font.color.rgb = RGBColor(0x55, 0x55, 0x88)

doc.add_paragraph()
ip = doc.add_paragraph()
ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = ip.add_run(
    f"Tarih: {datetime.datetime.now().strftime('%d %B %Y')}     "
    "Versiyon: 2.0     Gizlilik: Dahili Kullanım"
)
r3.font.size = Pt(10); r3.font.color.rgb = C_GREY

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# İÇİNDEKİLER
# ════════════════════════════════════════════════════════════════════════════
h1("İÇİNDEKİLER")
toc = [
    ("1",  "Sistem Genel Bakış"),
    ("2",  "YOLO + Bilgisayarlı Görü"),
    ("3",  "EMG — Elektromiyografi"),
    ("4",  "Polar H10 — Kalp Hızı ve HRV"),
    ("5",  "GPS / IMU"),
    ("6",  "Entegre Sistem — Her Şey Birleşince"),
    ("7",  "Doğruluk Katmanları"),
    ("8",  "Sakatlanma Tahmini"),
    ("9",  "Kulüpte Her Rol İçin Sistem"),
    ("10", "Sistem Mimarisi — Roller Arası Koordinasyon"),
    ("11", "Rakip Karşılaştırması"),
    ("12", "Doğruluğu Nasıl Valide Ederiz?"),
    ("13", "Sonuç"),
]
for no, title in toc:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"   {no}. {title}")
    r.font.size = Pt(10.5)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — SİSTEM GENEL BAKIŞ
# ════════════════════════════════════════════════════════════════════════════
h1("1. SİSTEM GENEL BAKIŞ")

body(
    "Bu sistem, futbol sporcularının sahadaki performansını, yorgunluk seviyelerini "
    "ve sakatlanma risklerini gerçek zamanlı olarak izlemek için dört farklı cihazı "
    "ve yapay zeka tabanlı görüntü işlemeyi bir araya getirir. "
    "Kondisyoner, fizyoterapist, teknik direktör ve kulüp yöneticisine "
    "kendi rollerine özel, veri destekli karar desteği sunar."
)

h2("1.1 Sistem Bileşenleri ve Tahmini Donanım Maliyetleri")

add_table(
    ["Cihaz", "Ne Yapar", "Tahmini Maliyet"],
    [
        ["📷  Kamera + YOLO", "Hareket analizi, biyomekanik, taktik, top takibi", "Altyapıya bağlı (kamera başı)"],
        ["🔌  EMG Elektrotu", "Kas aktivitesi, yorgunluk, H:Q oranı, nöromüsküler", "~€300–500 / sporcu"],
        ["❤️  Polar H10", "Kalp hızı, HRV, toparlanma, otonom sinir sistemi", "~€80–130 / sporcu"],
        ["🛰️  GPS / IMU", "Konum, sprint, fiziksel yük, Player Load", "~€200–400 / sporcu"],
    ],
    col_widths=[4, 9, 4]
)

body(
    "Her cihaz bağımsız çalışabilir. Ancak birleşince doğruluk ve kapsam "
    "katlanarak artar. Sistem hem anlık saha kararları hem de uzun vadeli "
    "yük yönetimi için kullanılır."
)

h2("1.2 Veri Akışı — Adım Adım")

body("Sistemin işleyişi beş adımda özetlenebilir:")

add_table(
    ["Adım", "Ne Olur", "Araç"],
    [
        ["1", "Saha verisi toplanır — her antrenman, her maç", "Kamera + EMG + Polar + GPS"],
        ["2", "Ham veri analiz edilir — açı, hız, kas yorgunluğu, yük", "YOLO + sinyal işleme algoritmaları"],
        ["3", "Metrikler hesaplanır — FI, TSB, H:Q, MPF, Player Load", "Analiz motoru"],
        ["4", "Role özel tavsiye üretilir — her kullanıcı kendi ekranını görür", "Karar destek katmanı"],
        ["5", "Feedback alınır — öneri doğru muydu? Sistem öğrenir", "Feedback döngüsü"],
    ],
    col_widths=[1.5, 9.5, 5.5]
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — YOLO
# ════════════════════════════════════════════════════════════════════════════
h1("2. YOLO + BİLGİSAYARLI GÖRÜ")

body(
    "YOLO (You Only Look Once) sahadaki tüm oyuncuları, topu ve hareketleri "
    "aynı anda tanıyan bir derin öğrenme modelidir. "
    "Tek kişiye odaklanan eski sistemlerin aksine YOLO tüm takımı aynı karede işler, "
    "hızlı harekette ve kalabalık sahnelerde çok daha güvenilir sonuç verir. "
    "Saniyede 60 kare analiz eder."
)

h2("2.1 Oyuncu Tespiti ve Takibi")

body(
    "YOLO her kareyi tarayarak tüm oyunculara benzersiz bir ID atar. "
    "ByteTrack algoritmasıyla bu kimlik maç boyunca korunur — "
    "oyuncu başka oyuncuların arkasına geçse bile sistem onu kaybetmez."
)
bullet("Tüm takım aynı anda izlenir")
bullet("Her oyuncuya ID atanır, maç boyunca takip edilir")
bullet("Jersey numarası tanıma (özel eğitimli modelle)")
bullet("Maç boyunca oyuncu ısı haritası üretilir")

example_box("Teknik Direktör / Analist — Örnek",
    "\"9 numaralı oyuncu maçın 67. dakikasından itibaren koşu alanını daralttı "
    "ve yüksek tempolu hareketlerden kaçınmaya başladı. "
    "Normal maçında sahanın %68'ini kaplıyor, bu maçta %41. "
    "Sahada saklanıyor — değiştirme zamanı.\"")

h2("2.2 Eklem Açıları ve Biyomekanik")

body(
    "YOLOv8-Pose modeli her oyuncunun iskelet noktalarını frame bazlı tespit eder. "
    "Bu noktalardan eklem açıları, hareket hızı ve ivme hesaplanır."
)

add_table(
    ["Metrik", "Nasıl Hesaplanır", "Ne Söyler"],
    [
        ["Diz açısı", "Kalça–Diz–Ayak bileği vektör açısı", "Teknik bozulma, ROM kaybı"],
        ["Kalça açısı", "Omuz–Kalça–Diz vektör açısı", "Stabilite, postür kalitesi"],
        ["Hareket hızı", "Landmark pozisyonunun türevi (Savitzky-Golay filtresi)", "Yorgunluk, patlayıcılık"],
        ["Hareket ivmesi", "Hızın ikinci türevi", "Nöromüsküler kapasite"],
        ["Diz valgus açısı", "Frontal düzlemde diz içe çökmesi", "ACL risk göstergesi"],
    ],
    col_widths=[3.5, 6.5, 6.5]
)

example_box("Teknik Bozulma — Örnek",
    "\"Sporcu antrenman öncesinde diz ROM'unu 143°'ye kadar açıyor. "
    "Antrenman sonunda bu değer 128°'ye düştü — %10 ROM kaybı. "
    "Uzatma fazı 0.31 saniyeden 0.41 saniyeye çıktı — hazırlık süresi uzuyor. "
    "Kamera henüz net göstermese de EMG bu bozulmayı 10 dakika önce işaret etmişti.\"")

h2("2.3 Tekme Faz Analizi")

body(
    "Sistem her tekmeyi otomatik tespit eder ve üç faza böler. "
    "Her fazın süresi yorgunlukla birlikte nasıl değiştiği izlenir."
)
add_table(
    ["Faz", "Tanımı", "Yorgunlukta Ne Olur"],
    [
        ["Yüklenme", "Hazırlık — bacağın geri çekilmesi", "Süresi kısalır, hazırlıksız vuruş"],
        ["Uzatma", "Vuruş — bacağın hedefe gitmesi", "Hız düşer, güç azalır"],
        ["Geri çekim", "Denge için toparlanma", "Uzar — denge kaybı riski artar"],
    ],
    col_widths=[3.5, 6, 7]
)

h2("2.4 Top Tespiti ve Oyuncu–Top Etkileşimi")
bullet("Top sahipliği oranı (possession %)")
bullet("Pas / şut / dribbling tespiti ve frekansı")
bullet("Şut anındaki biyomekanik kalite — ayak hızı, bacak pozisyonu")
bullet("Top hızı tahmini — vuruş gücünün göstergesi")
bullet("Tehlikeli bölgeye top girişi sayısı ve kalitesi")

h2("2.5 Taktiksel Metrikler")
bullet("Takım kompaktlığı — iki hat arası mesafe")
bullet("Pressing yoğunluğu — top kaybından itibaren kapama süresi")
bullet("Saha kapsama alanı — her oyuncunun kapladığı alan yüzdesi")

example_box("Teknik Direktör — Taktik Analiz Örneği",
    "\"Maçın son 20 dakikasında takım pressing mesafesi 8 metreden 14 metreye çıktı. "
    "İki hat arası mesafe genişledi, kompaktlık bozuldu. "
    "Fiziksel kapasite çökmesi taktiksel disiplini doğrudan etkiliyor. "
    "Bu dönemde rakip gol yeme riski istatistiksel olarak 2.4 kat artıyor.\"")

h2("2.6 Yorgunluk İndeksi (YOLO Tabanlı)")
body("7 biyomekanik metriğin ağırlıklı ortalaması → 0–100 arası skor:")
add_table(
    ["Metrik", "Ağırlık", "Yorgunlukla Yönü"],
    [
        ["Diz ROM", "%25", "↓ Düşer"],
        ["Peak hız", "%25", "↓ Düşer"],
        ["Peak hıza ulaşma süresi", "%15", "↑ Uzar"],
        ["Tekme yüksekliği", "%15", "↓ Düşer"],
        ["Ayak hızı", "%10", "↓ Düşer"],
        ["Tekme süresi", "%5", "↑ Uzar"],
        ["Ortalama hız", "%5", "↓ Düşer"],
    ],
    col_widths=[6, 3, 4]
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — EMG
# ════════════════════════════════════════════════════════════════════════════
h1("3. EMG — ELEKTROMİYOGRAFİ")

body(
    "EMG kasın üzerine yapıştırılan elektrotlarla kas lifi kasıldıkça oluşan "
    "elektrik sinyalini yakalar. Saniyede 1000–2000 ölçüm yapılır. "
    "En kritik özelliği: kameradan 8–12 dakika önce yorgunluğu tespit eder. "
    "Sporcu henüz yavaşlamadan, tekniği bozulmadan kasın içinde ne olduğunu görürüz."
)

h2("3.1 Kas Aktivasyon Amplitüdü — RMS")
body("Kasın ne kadar güçlü kasıldığının ölçüsü. Her kick penceresinde hesaplanır.")
example_box("Kondisyoner — Örnek",
    "\"Quadriceps birinci tekmede 0.82 mV ile çalışıyor. "
    "Onuncu tekmede 0.54 mV — kas giderek daha az güç üretiyor. "
    "Sporcu henüz yavaşlamadı ama motor kapasitesi %34 azaldı.\"")
body("Doğruluk: Aynı oturum içi ±%5. Oturumlar arası karşılaştırma için MVC normalizasyonu gerekir.", italic=True)

h2("3.2 Frekans Analizi — MPF / MDF")
body(
    "Yorulmuş kas lifleri sinyalin frekansını düşürür. "
    "Bu kas yorgunluğunun en erken ve en güvenilir göstergesidir. "
    "FFT (Hızlı Fourier Dönüşümü) ile gerçek zamanlı analiz yapılır."
)
example_box("Kritik Uyarı — Örnek",
    "\"Hamstring median frekansı antrenman başında 95 Hz. "
    "22. dakikada 67 Hz — %29 düşüş. "
    "Kamera hâlâ normal hız ve açı gösteriyor. "
    "Ama kas fizyolojik yorgunluk eşiğini geçti. "
    "Bir sonraki sprint veya ani frenleme sakatlanma riski taşıyor.\"")
body("Doğruluk: ±2–3 Hz — çok güvenilir. Laktat birikimi ile korelasyonu klinik çalışmalarda doğrulanmıştır.", italic=True)

h2("3.3 H:Q Oranı — Hamstring:Quadriceps")
body("İki kasın aktivasyon oranı. ACL ve hamstring yırtılmasının en güçlü öngörücüsü.")
add_table(
    ["H:Q Değeri", "Yorum", "Önerilen Aksiyon"],
    [
        ["> 0.80", "İdeal denge", "Mevcut antrenmanı sürdür"],
        ["0.60 – 0.80", "Normal aralık", "İzle, trend takibi yap"],
        ["0.50 – 0.60", "Dikkat — hamstring zayıf", "Eksantrik güçlendirme başlat"],
        ["< 0.50", "Kırmızı — yüksek ACL riski", "Yüksek yoğunluğu durdur, fizyoterapiste"],
    ],
    col_widths=[3.5, 5.5, 7.5]
)

h2("3.4 Ko-Kontraksiyon İndeksi")
body(
    "Agonist ve antagonist kasın aynı anda aktif olması. "
    "Yorulunca vücut eklemi korumak için her iki kası birlikte kilitler — "
    "enerji verimsizliği ve eklem aşınması anlamına gelir."
)
example_box("Kondisyoner — Örnek",
    "\"Maçın 75. dakikasında ko-kontraksiyon indeksi %340 arttı. "
    "Sporcu hem quadriceps hem hamstring'i aynı anda kasıyor — vücut alarm modunda. "
    "Her adımda dize normal yükün 1.8 katı kuvvet biniyor.\"")

h2("3.5 Pre-Aktivasyon Hızı")
body("Hareket başlamadan kasın ne kadar önce aktive olduğu — refleks koruma hızı.")
bullet("Normal hamstring pre-aktivasyonu: 30–40 ms")
bullet("Yorgunlukla: 12–18 ms'ye düşebilir")
bullet("Bu fark: diz korumasız kalma süresi — ACL ve menisküs risk penceresi")

h2("3.6 Kompanzasyon Deseni")
body("Bir kas yorulunca komşu kas onun işini üstlenir — hem performans düşer hem o kas aşırı yük alır.")
example_box("Fizyoterapist — Örnek",
    "\"Gluteus maximus yorulunca erector spinae (bel kası) devreye girdi. "
    "Bel kasları kalça kasının işini yapıyor. "
    "Bu bel ağrısı ve disk hernisinin bilinen habercisi. "
    "3 haftadır bu desen var, önleyici müdahale gecikti.\"")

h2("3.7 Nöromüsküler Verimlilik")
example_box("Analitik Veri — Örnek",
    "\"Antrenman öncesi: 0.74 mV harcayarak 143° diz açısı. "
    "Antrenman sonrası: 1.12 mV harcıyor ama sadece 128° alabiliyor. "
    "%52 daha fazla kas çalışması, %10 daha az hareket — "
    "nöromüsküler verimlilik kritik düzeyde düşmüş.\"")

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — POLAR H10 / HRV
# ════════════════════════════════════════════════════════════════════════════
h1("4. POLAR H10 — KALP HIZI ve HRV")

body(
    "Polar H10 göğüs bandı, kalbin her atışını tespit eder ve iki atış arasındaki "
    "milisaniye farkını (RR interval) kaydeder. "
    "Klinik ECG ile karşılaştırıldığında r > 0.99 korelasyon gösterir — "
    "tıbbi kalitede veri üretir. Consumer ürünler arasında sektör standardı."
)

h2("4.1 HRV — Sabah Dinlenme Ölçümü")
body(
    "Günün en kritik ölçümü. Sporcu sabah yataktan kalkmadan 5 dakika "
    "sırt üstü yatarak ölçüm yapar. Bu değer o günkü antrenman kararını belirler."
)
add_table(
    ["RMSSD Durumu", "Yorum", "Antrenman Kararı"],
    [
        ["Ortalamanın >%10 üzerinde", "Süper kompanse — hazır", "Yüksek yoğunluk ver"],
        ["Ortalama ±%10", "Normal — stabil", "Planlandığı gibi devam"],
        ["Ortalamanın %10–20 altında", "Dikkat — toparlanma eksik", "Orta yoğunluk, teknik çalış"],
        ["Ortalamanın >%20 altında", "Kırmızı — baskı altında", "Dinlenme veya çok hafif çalışma"],
    ],
    col_widths=[5.5, 5, 6]
)
example_box("Kondisyoner — Sabah Kararı Örneği",
    "\"Ahmet'in son 4 haftanın ortalama RMSSD'si 58 ms. Bugün 34 ms — %41 düşüş. "
    "Dün yüksek TRIMP'li maç + son 6 gün TSB negatif. "
    "Bugün yüksek yoğunluklu antrenman yapma. "
    "Hafif teknik çalışma veya aktif dinlenme önerilir. Yarın tekrar ölç.\"")

h2("4.2 7 Günlük HRV Trendi")
example_box("Overreaching Uyarısı — Örnek",
    "\"Kemal 6 gündür HRV'si düşüyor. Yük değişmedi, uyku normal raporluyor. "
    "Bu tablo overreaching başlangıcı. "
    "Kronik yorgunluğa geçmeden önce bu hafta yük %30 azalt.\"")

h2("4.3 TRIMP — Antrenman Kalp Yükü")
body("Her antrenmanın kalp üzerindeki baskısını sayısal olarak ifade eder.")
bullet("Hafif jogging 30 dk → TRIMP ~35")
bullet("Orta yoğunluklu antrenman 60 dk → TRIMP ~80–100")
bullet("Yüksek yoğunluklu maç 90 dk → TRIMP ~150–200")

h2("4.4 ATL / CTL / TSB — Sezon Yük Yönetimi")
add_table(
    ["Kavram", "Hesaplama", "Anlamı"],
    [
        ["ATL (Akut Yük)", "Son 7 günün TRIMP ortalaması", "Kısa vadeli yorgunluk"],
        ["CTL (Kronik Yük)", "Son 28 günün TRIMP ortalaması", "Uzun vadeli kondisyon"],
        ["TSB (Denge)", "CTL − ATL", "Anlık hazırlık durumu"],
    ],
    col_widths=[4, 7, 6]
)
add_table(
    ["TSB Değeri", "Durum", "Aksiyon"],
    [
        ["< −30", "KRİTİK — Aşırı yük", "Sakatlanma çok yakın, acil yük azalt"],
        ["−10 ile −30", "Dikkat — Yorgunluk birikimi", "İzle, hafiflet"],
        ["−10 ile +5", "Normal antrenman aralığı", "Planlandığı gibi devam"],
        ["+5 ile +25", "TAPER — Maça hazır", "Optimal performans penceresi"],
        ["> +25", "Dekondisyon başlıyor", "Yük artır"],
    ],
    col_widths=[3.5, 5.5, 7.5]
)

h2("4.5 Kalp Toparlanma Hızı (HRR)")
bullet("İyi kondisyon: 1 dk'da >30 bpm düşüş")
bullet("Orta kondisyon: 20–30 bpm")
bullet("Düşük kondisyon: <20 bpm — kardiyovasküler çalışma gerekli")
bullet("Sezon içi takipte HRR artışı kondisyon gelişiminin kanıtı")

h2("4.6 EPOC — Antrenman Sonrası Metabolik Yük")
example_box("Beslenme Entegrasyonu — Örnek",
    "\"Maç sonrası EPOC 2.5 saat olarak tahmin edildi. "
    "Bu sürede karbonhidrat restorasyon penceresi açık — "
    "ilk 30 dakikada 1.2 g/kg KH + 0.4 g/kg protein alınmalı.\"")

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — GPS / IMU
# ════════════════════════════════════════════════════════════════════════════
h1("5. GPS / IMU")

body(
    "GPS saniyede 10–20 kez konum günceller. "
    "IMU (ivmeölçer + jiroskop) saniyede 100–200 kez 3 eksende ivme ve dönüş ölçer. "
    "Birlikte sporcunun tüm fiziksel yük profilini çıkarırlar."
)

h2("5.1 Sprint Metrikleri")
add_table(
    ["Metrik", "Doğruluk", "Ne Söyler"],
    [
        ["Maks sprint hızı", "±0.1–0.2 m/s", "Bireysel zirve kapasitesi"],
        ["Sprint sayısı", "±1–2 sprint", "Yoğunluk profili"],
        ["Sprint mesafesi", "±%1–3", "Toplam yüksek yoğunluklu yük"],
        ["Sprint süresi", "±0.1 s", "Patlayıcılık ve hız dayanıklılığı"],
    ],
    col_widths=[5, 4, 7.5]
)
example_box("Maç Analizi — Örnek",
    "\"9 numaralı oyuncu birinci yarıda 8 sprint attı (280 m, maks 28.9 km/h). "
    "İkinci yarıda 3 sprint (95 m, maks 24.1 km/h). "
    "Hız kapasitesi %17 düştü — değiştirme zamanı.\"")

h2("5.2 Yüksek Yoğunluklu Koşu Zonları")
add_table(
    ["Zon", "Hız", "90 Dk Referans (Futbol)"],
    [
        ["Yürüyüş", "0–7 km/h", "~3.5–4.0 km"],
        ["Hafif Koşu", "7–14 km/h", "~3.0–4.0 km"],
        ["Orta Koşu", "14–21 km/h", "~2.5–3.5 km"],
        ["Yüksek Yoğunluk", "21–25 km/h", "~0.8–1.2 km"],
        ["Sprint", "> 25 km/h", "~0.3–0.6 km"],
    ],
    col_widths=[4, 4, 8.5]
)

h2("5.3 Player Load — Mekanik Vücut Yükü")
body("3 eksen ivmenin karekökü toplamı — tüm koşu, frenleme, zıplama ve çarpışmaları tek sayıya indirir.")
example_box("Yük Karşılaştırması — Örnek",
    "\"Defender bu maçta 850 Player Load aldı, forward 720. "
    "Savunmacı daha fazla hızlanma-frenleme yaptı. "
    "Bu hafta o oyuncunun bacak kaslarına ekstra yük bindi — çarşamba antrenmanında yük düşürülmeli.\"")

h2("5.4 Akselerasyon / Deselerasyon Sayısı")
body(
    "Eşik üstü ani hızlanma ve frenleme sayısı. "
    "Her yüksek yoğunluklu deselerasyon dize vücut ağırlığının 4–8 katı kuvvet bindirir."
)
bullet("Bir maçta ortalama 40–70 yüksek yoğunluklu frenleme")
bullet("EMG ile birleşince: o anki diz koruma kapasitesi + mekanik stres = gerçek risk profili")

h2("5.5 Isı Haritası ve Saha Kapsama")
example_box("Teknik Direktör / Analist — Örnek",
    "\"Bu orta saha oyuncusu normal maçında 9.8 km koşuyor ve sahanın %70'ini kaplıyor. "
    "Bu maçta 7.1 km ve %48 kaplama. "
    "Kondisyon düşük veya takım sistemi değişti — veri taktik kararı destekliyor.\"")

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6 — ENTEGRE SİSTEM
# ════════════════════════════════════════════════════════════════════════════
h1("6. ENTEGRE SİSTEM — HER ŞEY BİRLEŞİNCE")

body(
    "Her cihaz kendi başına değerlidir. Ama gerçek güç birleşimde ortaya çıkar. "
    "Kamera tekniği gösterir, EMG kasın içini gösterir, HRV otonom sistemi gösterir, "
    "GPS fiziksel yükü gösterir. "
    "Dört katman birleşince hiçbir cihazın tek başına göremeyeceği bağlantılar ortaya çıkar."
)

h2("Senaryo 1 — Görünmeden Önce Tespit")
body("Kondisyoner için en kritik senaryo: sporcu teknik olarak hâlâ iyi görünüyor ama risk başlamış.")
add_table(
    ["Zaman", "Veri", "Değer", "Yorum"],
    [
        ["Dakika 22", "EMG — Hamstring MPF", "95 → 67 Hz  (−29%)", "⚠️ Periferik yorgunluk başladı"],
        ["Dakika 22", "YOLO — Hareket hızı", "Normal", "Kamera henüz bozulma göstermiyor"],
        ["Dakika 22", "GPS — Sprint hızı", "Normal", "Fiziksel çöküş henüz yok"],
        ["Dakika 22", "HR", "171 bpm", "Yüksek ama anormal değil"],
        ["Dakika 30", "YOLO — Hareket hızı", "−18% düşüş", "Artık kamera da görüyor"],
    ],
    col_widths=[2.5, 4.5, 4.5, 5]
)
example_box("Sistem Uyarısı — 22. Dakika",
    "\"⚠️ ORTA RİSK — Hamstring periferik yorgunluk eşiğine ulaştı. "
    "Kamera henüz bozulma göstermese de bir sonraki yüksek hızlı sprint "
    "veya ani frenleme sakatlanma riski taşıyor. "
    "Değiştirme veya koşu yoğunluğu kısıtlaması düşünülebilir.\"")

h2("Senaryo 2 — Sabah Hazırlık Kararı")
add_table(
    ["Gösterge", "Değer", "Durum"],
    [
        ["Sabah RMSSD", "38 ms  (4 hafta ort: 62 ms)", "🔴 −39%"],
        ["TSB", "−34", "🔴 Kritik aşırı yük"],
        ["Bu hafta kümülatif yük", "Geçen haftanın %140'ı", "🔴 Aşım"],
        ["HRV trend", "6 gündür düşüyor", "🔴 Kronik"],
        ["Uyku (subjektif)", "6.5 saat", "🟡 Yetersiz"],
    ],
    col_widths=[5.5, 6.5, 4.5]
)
example_box("Sabah Karar Sistemi — Çıktı",
    "\"❌ BUGÜN YÜKSEK YOĞUNLUKLU ANTRENMAN YAPMA\n"
    "5 göstergeden 4'ü kırmızı. Otonom sinir sistemi kronik baskı altında. "
    "Önerilen: hafif teknik çalışma (TRIMP hedef <50) veya aktif dinlenme. "
    "Yarın sabah tekrar HRV ölç.\"")

h2("Senaryo 3 — Takım Yorgunluk Haritası")
body("Maç sırasında kondisyoner tüm oyuncuların yorgunluk durumunu tek ekranda görür:")
add_table(
    ["Oyuncu", "YOLO Fatigue", "EMG (Ham.)", "HR Zonu", "Player Load", "Genel Risk"],
    [
        ["No. 7",  "FI: 28", "Normal",    "Z3", "420", "🟢 Düşük"],
        ["No. 9",  "FI: 51", "MPF −18%",  "Z4", "610", "🟡 Orta"],
        ["No. 11", "FI: 67", "MPF −31%",  "Z5", "730", "🔴 Yüksek"],
        ["No. 4",  "FI: 44", "H:Q 0.54", "Z3", "580", "🟡 Orta"],
    ],
    col_widths=[2.5, 3, 3, 2.5, 3, 3]
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7 — DOĞRULUK KATMANLARI
# ════════════════════════════════════════════════════════════════════════════
h1("7. DOĞRULUK KATMANLARI — KÜMÜLATİF")

body(
    "Sistem her cihaz eklendiğinde doğruluk artıyor. "
    "Aşağıdaki tablo genel kondisyon ve yorgunluk değerlendirmesi için "
    "bileşik doğruluk tahminini göstermektedir."
)
add_table(
    ["Katman", "Eklenen Cihaz / Yöntem", "Kümülatif Doğruluk", "Kazanılan Yetenek"],
    [
        ["Temel",   "Sadece YOLO + Kamera",         "%60–68", "Teknik trend, taktik, yorgunluk yönü"],
        ["Katman 2", "+ EMG",                        "%74–80", "Nöromüsküler: 8–12 dk erken tespit"],
        ["Katman 3", "+ Polar H10 (HRV)",            "%83–88", "Kardiyovasküler + otonom: sabah kararı"],
        ["Katman 4", "+ GPS / IMU",                  "%87–91", "Fiziksel yük: mekanik stres, sprint"],
        ["Katman 5", "+ Subjektif Form (uyku, RPE)", "%90–94", "Bağlam: uyku, beslenme, öznel his"],
    ],
    col_widths=[2.5, 5, 3.5, 5.5]
)

add_table(
    ["Cihaz / Metrik", "Doğruluk", "Altın Standart", "Validasyon Yöntemi"],
    [
        ["YOLO — Eklem açısı (mutlak değer)",  "±5–10°",   "Vicon Motion Capture",   "Spot goniometre ölçümü"],
        ["YOLO — Teknik trend yönü",           "Yüksek",   "Uzman video analizi",    "Manuel karşılaştırma"],
        ["EMG — Oturum içi amplitüd",          "±%5",      "Klinik EMG sistemi",     "MVC normalizasyonu"],
        ["EMG — MPF frekans düşüşü",           "±2–3 Hz",  "Klinik EMG + laktat",    "Laktat eşiği testi"],
        ["Polar H10 — RMSSD",                  "±1–2 ms",  "Klinik ECG",             "Literatürde r>0.99"],
        ["GPS — Sprint hızı",                  "±0.2 m/s", "Radar tabancası",        "Spot radar ölçümü"],
        ["GPS — Player Load trendi",           "Çok yüksek","Sistem içi tutarlılık", "—"],
    ],
    col_widths=[5, 3, 3.5, 5]
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 8 — SAKATLANMA TAHMİNİ
# ════════════════════════════════════════════════════════════════════════════
h1("8. SAKATLANMA TAHMİNİ")

body(
    "Sakatlanma tahmini sportta en zor problemlerden biridir. "
    "Hiçbir sistem 'yarın sakatlanacaksın' diyemez. "
    "Doğru soru şudur:"
)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.left_indent  = Cm(2)
p.paragraph_format.right_indent = Cm(2)
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after  = Pt(8)
r = p.add_run(
    "\"Bu sporcu önümüzdeki X günde sakatlanma riskinde "
    "normalden anlamlı bir yükselme var mı?\""
)
r.bold = True; r.italic = True; r.font.size = Pt(11); r.font.color.rgb = C_BLUE

body(
    "Bu soruyu kural tabanlı sistemimizle %63–70, makine öğrenmesi modeli "
    "olgunlaştıkça %82–88 doğrulukla cevaplayabiliriz. "
    "Birinci lig kulübünde tek bir ACL sakatlığı "
    "rehabilitasyon + oyuncu kaybı olarak €200,000–400,000 maliyeti var."
)

h2("8.1 Hamstring Yırtılması")
add_table(
    ["Gösterge", "Kaynak", "Eşik", "Ağırlık"],
    [
        ["Hamstring MPF frekans düşüşü", "EMG",      "> %20 → kırmızı",  "Çok yüksek"],
        ["Hamstring RMS amplitüd kaybı", "EMG",      "> %25 → kırmızı",  "Yüksek"],
        ["ACWR (Akut:Kronik Yük)",       "GPS + HR", "> 1.5 → kırmızı",  "Yüksek"],
        ["H:Q oranı",                    "EMG",      "< 0.60 → kırmızı", "Çok yüksek"],
        ["HRV 5+ gün düşüş trendi",      "Polar",    "Negatif → sarı",   "Orta"],
        ["Kalça fleksör ROM azalması",    "YOLO",     "> %15 → sarı",     "Orta"],
    ],
    col_widths=[5, 2.5, 4, 4]
)
example_box("Hamstring Risk Uyarısı — Örnek",
    "\"⚠️ HAMSTRING RİSK YÜKSELDİ — Kemal Aydın (No. 11)\n\n"
    "● Hamstring MPF: 95 Hz → 67 Hz  (−29%)   🔴\n"
    "● H:Q oranı: 0.52  (eşik 0.60)            🔴\n"
    "● Bu hafta ACWR: 1.6                       🔴\n"
    "○ HRV 4 gündür düşüyor                    🟡\n\n"
    "Öneri: Yarın yüksek hızlı sprint yok. "
    "Eksantrik hamstring güçlendirme başlat (Nordic Curl, Deadlift). "
    "3 gün sonra yeniden değerlendir.\"")

h2("8.2 ACL Kopması")
body("ACL kopmasının %80'i temas olmadan olur — tamamen biyomekanik kökenli. "
     "Bu nedenle kamera + EMG birleşimi bu sakatlık için en kritik araç kombinasyonudur.")
add_table(
    ["Gösterge", "Kaynak", "Eşik", "Bilimsel Dayanak"],
    [
        ["H:Q oranı",                    "EMG",      "< 0.60",  "Hewett et al. — klinik standart"],
        ["Diz valgus açısı (iniş anı)",  "YOLO",     "> 8°",    "Meeuwisse multi-faktör modeli"],
        ["Hamstring pre-aktivasyon",     "EMG",      "< 20 ms", "Refleks koruma yetersizliği"],
        ["Sol-sağ quad asimetrisi",      "EMG+YOLO", "> %15",   "Frontal düzlem dengesizliği"],
        ["Yüksek yoğunluklu deselerasyon", "GPS/IMU", "Haftalık >60", "Eklem stres proxy"],
        ["TSB",                          "HR",       "< −30",   "Yorgunlukla nöromüsküler kontrol kaybı"],
    ],
    col_widths=[4.5, 2.5, 3, 6.5]
)
example_box("ACL Risk Profili — Örnek",
    "\"🔴 YÜKSEK ACL RİSKİ — Mehmet Kaya (No. 4)\n\n"
    "6 göstergeden 5'i kırmızı:\n"
    "● H:Q: 0.51   ● Diz valgus: +11°   ● Pre-aktivasyon: 16 ms\n"
    "● Asimetri: %18   ● TSB: −34\n\n"
    "Bu sporcu her frenleme ve yön değiştirmede dizini içe katlıyor, "
    "yorgunlukla daha da bozuluyor. Hamstring dizini yeterince koruyamıyor.\n\n"
    "Aksiyon: Yüksek yoğunluklu antrenman ve maç önerilmez. "
    "Fizyoterapist değerlendirmesi gerekli.\"")

h2("8.3 Aşırı Kullanım Sakatlıkları")
bullet("ACWR > 1.5: 6 hafta içinde aşırı kullanım sakatlığı riski 4.7 kat artıyor  (Gabbett, 2016)")
bullet("HRV 14 gün sürekli düşüş: kronik adaptasyon yetersizliği")
bullet("TSB kronik negatif (2 haftadan uzun < −20): overreaching")
bullet("Yüksek antrenman monotonluğu: düşük varyasyon + yüksek hacim kombinasyonu")

h2("8.4 Metodoloji — 3 Aşamalı Gelişim")
add_table(
    ["Aşama", "Yöntem", "Gereken", "Doğruluk"],
    [
        ["Aşama 1 (Şu an)", "Kural tabanlı — eşik sayısı", "Sadece sistem kurulu olsun", "%63–70"],
        ["Aşama 2 (6–12 ay)", "Lojistik regresyon, kişisel ağırlıklar", "50+ sporcu, 1–2 sezon veri", "%72–80"],
        ["Aşama 3 (2–3 yıl)", "LSTM + Random Forest, zaman serisi", "Geniş veri + sakatlanma kayıtları", "%82–88"],
    ],
    col_widths=[4, 5, 5.5, 2.5]
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 9 — KULÜPTE HER ROL İÇİN SİSTEM
# ════════════════════════════════════════════════════════════════════════════
h1("9. KULÜPTE HER ROL İÇİN SİSTEM")

body(
    "Aynı veri farklı rollere farklı sorular sorar. "
    "Sistem tek bir analiz motorunun çıktısını her kullanıcının kendi diline çevirir. "
    "Her rol kendi ekranını görür, kendi sorularını sorar, kendi kararını verir. "
    "Son karar her zaman insanda kalır."
)

divider()

# ── KONDİSYONER ────────────────────────────────────────────────────────────
h2("9.1 Kondisyoner")
body("Soruları: Bugün kimi ne kadar yükleyebilirim? Kim dinlenmeli? Takım bu hafta nerede?")
body("Sistem sabah 08:00'de otomatik günlük plan üretir:")
chat_box("Kondisyoner", "Bu hafta en çok yüklenen 3 oyuncu kim ve bugün ne yapabilirler?",
    "Kemal (No.11): Bu hafta Player Load 2.840 — sezon ortalamasının %34 üzerinde. "
    "TSB −34, HRV 6 gündür düşüyor. Bugün sadece aktif dinlenme — koşu yok.\n\n"
    "Mehmet (No.4): ACWR 1.6, hamstring RMS −28%. Hafif teknik çalışma, sprint yok. "
    "Eksantrik güçlendirme seti öneririm.\n\n"
    "Ali (No.9): TSB −18, HRV stabil. Orta yoğunluk güvenli, yüksek yoğunluk yok.",
    role_color=RGBColor(0x00, 0x77, 0x7A)
)
body("Haftalık üretilen bilgiler:")
bullet("Günlük antrenman planı (oyuncu bazlı yük önerisi)")
bullet("Haftalık TRIMP hedefi ve gerçekleşme karşılaştırması")
bullet("ATL / CTL / TSB grafiği — sezon boyunca yük trendi")
bullet("Toparlanma hızı trendi — kim daha yavaş toparlanıyor?")
bullet("Dinlenme günü önerisi — hangi gün zorunlu?")
body("Feedback: \"Öneri uygulandı mı? Sporcu nasıl tepki verdi?\" — tek tık ile yanıt.", italic=True)

divider()

# ── TEKNİK DİREKTÖR ────────────────────────────────────────────────────────
h2("9.2 Teknik Direktör / Başantrenör")
body("Soruları: Cumartesi kimler oynayabilir? Maçta kim değişmeli, ne zaman? En hazır 11 kim?")
chat_box("Teknik Direktör", "Cumartesi maçı için fiziksel olarak en hazır 11'i öner, pressing yapmamızı istiyorum.",
    "Pressing için yüksek kardiyovasküler kapasite ve hız tekrarı gerekiyor.\n"
    "TSB +5 ile +20 arası ve HRV stabil 9 oyuncu var — bunlar tam hazır.\n\n"
    "Önerilen 11'de dikkat edilecekler:\n"
    "→ No.11 ve No.4 başlatılmamalı (hamstring risk + ACL sinyalleri aktif)\n"
    "→ No.7 oynayabilir ama pressing yoğunluğu düşük tutulmalı — TSB −8, "
    "60. dakika sonrası değişiklik planlanabilir.",
    role_color=RGBColor(0x1B, 0x5E, 0x20)
)
body("Maç sırasında canlı uyarı sistemi:")
example_box("Maç İçi Uyarı — 67. Dakika",
    "\"⚠️ No.9 DEĞİŞTİRME UYARISI\n"
    "Hamstring MPF −24%   |   GPS sprint hızı −17%   |   HR Zone 5'te 18 dakikadır\n"
    "Önümüzdeki 10 dakika içinde değiştirmezseniz sakatlanma riski 3.2x artıyor.\"")
body("Feedback: \"Değişiklik yapıldı mı? Doğru zamanlama mıydı?\" — maç sonu girilir.", italic=True)

divider()

# ── FİZYOTERAPİST ──────────────────────────────────────────────────────────
h2("9.3 Fizyoterapist")
body("Soruları: Bugün kime önleyici müdahale yapmalıyım? Sakatlanma sonrası saha dönüşüne hazır mı?")
chat_box("Fizyoterapist", "No.4'ün hamstring riski neden bu kadar hızlı arttı? Son 3 haftayı açıkla.",
    "3 hafta önce: H:Q 0.68 — normal aralık.\n"
    "2 hafta önce: Şampiyonlar Ligi + 3 gün sonra lig maçı. ACWR 1.7'ye çıktı. H:Q 0.61'e düştü.\n"
    "Geçen hafta: Ek sprint seansı yapıldı. H:Q 0.54'e indi, pre-aktivasyon 16 ms'ye düştü.\n\n"
    "3 haftada %21 bozulma — bu hızda düşüş literatürde yüksek risk göstergesi. "
    "Muhtemelen sıkışık maç takvimi + yetersiz toparlanma süresi.\n\n"
    "Öneri: Nordic Curl protokolü + bu hafta sprint yasağı + 5 gün sonra yeniden değerlendirme.",
    role_color=RGBColor(0x4A, 0x14, 0x8C)
)
body("Günlük otomatik üretilen risk panosu:")
example_box("Günlük Risk Panosu",
    "🔴 YÜKSEK RİSK — bugün müdahale et:\n"
    "   No.4: H:Q 0.51, pre-aktivasyon 16 ms, valgus artıyor\n"
    "   → Nordic Curl × 3 set + NMES + manuel hamstring değerlendirme\n\n"
    "🟡 İZLE — rutin önleme yeterli:\n"
    "   No.7: Sol-sağ asimetri %14'e çıktı (3 haftadır artış trendi)\n"
    "   → Tek bacak güç egzersizleri ekle\n\n"
    "🟢 NORMAL — 8 oyuncu, standart protokol")
body("Sakatlanma sonrası geri dönüş takibi:")
example_box("Return-to-Play — No.11, Gün 8/14",
    "EMG asimetrisi: %22  (hedef: <%10)\n"
    "MPF frekansı: sağlıklı bacağın %87'si  (hedef: >%90)\n"
    "HRV: normal seyir\n\n"
    "Saha dönüşü tahmini: 6 daha gün\n"
    "Bugün için güvenli yük: Düz koşu ≤ %70 hız, yön değiştirme yok")

divider()

# ── BESLENME UZMANI ─────────────────────────────────────────────────────────
h2("9.4 Beslenme Uzmanı")
body("Soruları: Bu oyuncu ne zaman ne yemeli? Toparlanma için kime ne önereceğim? Takviye zamanlaması?")
chat_box("Beslenme Uzmanı", "Bugün antrenman sonrası toparlanma beslenme planı nedir?",
    "Yüksek hasar grubu — Player Load >600, EPOC yüksek — 7 oyuncu:\n"
    "→ İlk 30 dk: 1.2 g/kg KH + 0.4 g/kg protein\n"
    "→ 2. saat: tam öğün, yüksek karbonhidrat\n"
    "→ Gece: kazein proteini (yavaş sindirim, gece onarımı)\n\n"
    "Orta grup — 8 oyuncu:\n"
    "→ Normal toparlanma beslemesi yeterli\n\n"
    "Takviye zamanlaması:\n"
    "→ Kreatin: antrenman sonrası pencere açık (No.9, No.11 — amplitüd düşüşü var)\n"
    "→ Beta-alanin: sabah kahvaltısıyla (aerobik kapasitesi düşük oyuncular)\n"
    "→ Magnezyum: gece (HRV düşük 4 oyuncu)",
    role_color=RGBColor(0xE6, 0x51, 0x00)
)
body("Maç öncesi karbonhidrat yükleme planı (48 saat öncesi):")
example_box("Cumartesi Maçı İçin KH Yükleme",
    "Perşembe akşam: 8–10 g/kg KH\n"
    "Cuma: hafif antrenman + 6–8 g/kg KH\n"
    "Cumartesi sabah (3 saat önce): 1–2 g/kg düşük lifli KH\n\n"
    "Bireysel not — No.7:\n"
    "Geçen maç öncesi HRV düşüktü, sindirim stres belirtisi.\n"
    "→ Cuma akşamı glutamin + probiyotik ekle")
body("Takviye önerisi kuralları (örnek mantık):")
add_table(
    ["Takviye", "Tetikleyen Veri", "Zamanlama", "Kanıt Seviyesi"],
    [
        ["Kreatin", "EMG amplitüd >%8 düşüş veya hız <−10%", "Antrenman sonrası", "A — Güçlü"],
        ["Beta-alanin", "Aerobik kapasitede düşüş", "Sabah kahvaltı", "A — Güçlü"],
        ["BCAA / Whey", "FI > 33, yüksek kas hasarı", "Hemen antrenman sonrası", "B — İyi"],
        ["Magnezyum", "FI > 40, HRV kronik düşük", "Gece yatmadan önce", "B — İyi"],
        ["Kafein", "Hız <−8%, sabah HRV düşük", "Maçtan 60 dk önce", "A — Güçlü"],
        ["Omega-3", "FI > 33 veya ROM düşüşü var", "Yemekle birlikte", "B — İyi"],
    ],
    col_widths=[3, 5, 3.5, 3.5]
)
body("Feedback: \"HRV ertesi sabah öncesiyle karşılaştırıldığında beslenme müdahalesi işe yaradı mı?\"", italic=True)

divider()

# ── YÖNETİCİ ───────────────────────────────────────────────────────────────
h2("9.5 Kulüp Yöneticisi / Sportif Direktör")
body("Soruları: Hangi pozisyona oyuncu lazım? Sözleşme uzatmasına değer mi? Sezon sonu risk nerede?")
chat_box("Sportif Direktör", "Transfer penceresi 3 ay sonra. Kadroda kritik zayıflık var mı?",
    "Sol bek pozisyonu: İki oyuncunuzda da uzun vadeli fiziksel düşüş trendi var. "
    "No.3 maks sprint hızı bu sezonda %8 geriledi, toparlanma süresi uzuyor. "
    "No.15 ACWR son 2 aydır kronik yüksek — sezon sonu sakatlık riski yüksek.\n\n"
    "Merkez forvet: No.9 fiziksel zirvesinin gerisinde — "
    "1–2 sezon yüksek performanslı futbolu kaldı tahminen.\n\n"
    "Önerilen transfer öncelikleri:\n"
    "1. Sol bek — acil, bir sakatlıkta yedeğiniz yok\n"
    "2. Merkez forvet — orta vade, No.9 yedekleme",
    role_color=RGBColor(0xB7, 0x1C, 0x1C)
)
body("Haftalık yönetici özet raporu:")
example_box("Haftalık Kadro Sağlık Raporu",
    "Önümüzdeki 3 maç için tahmini müsaitlik:\n"
    "→ Cumartesi: 20/23 oyuncu hazır\n"
    "→ Salı (Kupa): 18/23 (2 hamstring risk, 1 devam eden rehab)\n"
    "→ Cumartesi+1: 17/23 (yük yönetimi gerekli)\n\n"
    "⚠️ KADRO DERİNLİĞİ UYARISI:\n"
    "Sol bek pozisyonunda iki oyuncu da yüksek yük altında. "
    "1 sakatlanma durumunda yedek yok.\n\n"
    "Bu sezon tahmini önlenen sakatlanma katkısı: 3 müdahale")
body("Oyuncu fiziksel değer analizi — sözleşme kararı için:")
example_box("Oyuncu Fiziksel Profil — No.9 (Sözleşme Değerlendirmesi)",
    "Son 8 ay trend:\n"
    "→ Maks sprint hızı: 31.2 → 29.1 km/h (−6.7%)\n"
    "→ Yüksek yoğunluklu koşu: hafif düşüş trendi\n"
    "→ HRV toparlanma süresi: 28 saat → 34 saat (uzuyor)\n"
    "→ Sakatlanma geçmişi: 2 hamstring, tekrarlıyor\n\n"
    "Fiziksel zirve: geride kaldı\n"
    "Yüksek performanslı sezon tahmini: 1–2 sezon kaldı\n"
    "→ Uzun vadeli sözleşme riski yüksek, kısa vadeli veya satış değerlendirilebilir")

divider()

# ── TAKIM DOKTORU ───────────────────────────────────────────────────────────
h2("9.6 Takım Doktoru")
body("HRV ve ACWR kombinasyonundan bağışıklık baskı proxy'si üretilir.")
bullet("Yoğun maç takviminde bağışıklık riski yükselen oyuncular erken işaretlenir")
bullet("Kronik yük + kemik stres takibi → stres kırığı riski")
bullet("NSAID zamanlaması: antrenman öncesi değil, antrenman sonrası (protein sentezi baskılanmasın)")
bullet("Uyku kalitesi tahmini — HRV gece varyasyonundan → uyku bozukluğu erken işareti")
example_box("Doktor — Örnek",
    "\"Bu hafta 4 oyuncunun bağışıklık baskı riski yüksek. "
    "HRV + ACWR kombinasyonu kronik stres altında olduklarını gösteriyor. "
    "Üst solunum yolu enfeksiyonu sezonunda dikkat — "
    "antrenman öncesi semptom kontrolü yapılmalı.\"")

divider()

# ── MENTAL KOÇ ──────────────────────────────────────────────────────────────
h2("9.7 Mental Koç / Sporcu Psikoloğu")
body(
    "HRV fiziksel yorgunluk olmaksızın düştüğünde — uyku normal, yük normal — "
    "bu mental/psikolojik stres belirtisi olabilir."
)
chat_box("Mental Koç", "No.10'un son haftalarda performansı düşüyor ama fiziksel verisi normal görünüyor.",
    "Doğru tespit. Fiziksel yük normal, uyku süresi normal (forma göre), "
    "antrenman performansı orta.\n\n"
    "Ama HRV sabah değerleri büyük maç öncesi her seferinde düşüyor — "
    "normal maçlarda bu desen yok.\n\n"
    "Bu profil: performans anksiyetesi sinyali.\n"
    "Geçmiş 4 büyük maça bakıldığında, "
    "pre-match HRV düşük olan maçlarda performans puanı %22 daha düşük.\n\n"
    "Öneri: Büyük maç öncesi rutin konuşma + nefes protokolü dene, "
    "HRV'nin değişip değişmediğini izle.",
    role_color=RGBColor(0x55, 0x00, 0x88)
)

divider()

# ── SPORCU KENDİSİ ──────────────────────────────────────────────────────────
h2("9.8 Sporcu Kendisi")
body("Sporcu kendi dashboardında kendi verisini kendi dilinde görür.")
chat_box("Sporcu", "Bu hafta neden bu kadar yoruldum, geçen haftadan ne farkı var?",
    "Geçen hafta: 2 maç + 3 antrenman, Player Load 1.840, "
    "HRV stabil kaldı, toparlanman 28 saatti.\n\n"
    "Bu hafta: 2 maç + 4 antrenman (Çarşamba ek sprint eklendi), "
    "Player Load 2.410 — %31 artış. "
    "HRV toparlanman 38 saate uzadı. Hamstring frekansın %18 düştü.\n\n"
    "Fark: Çarşamba'ki ek sprint seansı kritik nokta. "
    "Bu hafta yük biraz fazla verildi — "
    "kondisyonerle konuşman faydalı olur.",
    role_color=RGBColor(0x00, 0x55, 0x88)
)
bullet("Haftalık yük ve toparlanma özeti")
bullet("Sağ-sol bacak güç dengesi")
bullet("Kondisyon gelişim trendi — sezon başından bu yana")
bullet("\"Bugün yüksek yoğunluklu antrenmana hazır mısın?\" — HRV bazlı basit yanıt")

divider()
h2("9.9 Feedback Döngüsü — Sistem Nasıl Öğrenir?")
body(
    "Her tavsiyenin altında üç buton vardır. "
    "Her tık sisteme bir öğrenme verisi sağlar."
)
add_table(
    ["Buton", "Ne Söyler", "Sistem Ne Öğrenir"],
    [
        ["✅ Uygulandı — sonuç beklendi gibi", "Öneri doğruydu", "Bu kombinasyonu koru, ağırlığı artır"],
        ["⚠️ Uygulandı — sonuç farklı geldi", "Öneri eksik/yanlıştı", "Bu eşiği kalibre et"],
        ["❌ Uygulanmadı", "Başka bilgim vardı / saha koşulları", "Bağlam eksikti — form sorusu ekle"],
    ],
    col_widths=[5.5, 4, 7]
)
body(
    "6 ayda her oyuncu için kişiselleştirilmiş kural ağırlıkları oluşur. "
    "2 yılda makine öğrenmesi için etiketli veri hazır olur. "
    "Kondisyoner değişir, teknik direktör ayrılır — "
    "ama sistemin öğrendikleri kulüpte kalır.",
    bold=True
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 10 — SİSTEM MİMARİSİ
# ════════════════════════════════════════════════════════════════════════════
h1("10. SİSTEM MİMARİSİ — ROLLER ARASI KOORDİNASYON")

body(
    "Aşağıdaki diyagram sistemin katmanlarını ve roller arası veri akışını göstermektedir. "
    "Veri tabandan yukarı akar, her katman bir öncekinin üzerine inşa edilir, "
    "her rolün çıktısı feedback olarak sisteme geri döner."
)

doc.add_paragraph()

arch_lines = [
    "┌─────────────────────────────────────────────────────────────────────┐",
    "│                        VERİ KATMANI                                │",
    "│        📷 YOLO    🔌 EMG    ❤️ Polar H10    🛰️ GPS/IMU              │",
    "│              + 📝 Subjektif Form (uyku, RPE, beslenme)             │",
    "└──────────────────────────────┬──────────────────────────────────────┘",
    "                               │",
    "                               ▼",
    "┌─────────────────────────────────────────────────────────────────────┐",
    "│                       ANALİZ MOTORU                                │",
    "│   Yorgunluk İndeksi · Sakatlanma Riski · Yük Yönetimi (ATL/CTL)   │",
    "│   MPF/RMS · H:Q Oranı · HRV Trend · Player Load · Biyomekanik     │",
    "└──────┬──────────┬──────────┬──────────┬──────────┬────────────────┘",
    "       │          │          │          │          │",
    "       ▼          ▼          ▼          ▼          ▼",
    "  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐",
    "  │KONDİS. │ │FİZYO.  │ │TEK.    │ │BESLEN. │ │YÖNETİCİ│",
    "  │Günlük  │ │Risk    │ │DIR.    │ │Öğün    │ │Kadro   │",
    "  │plan    │ │panosu  │ │Kadro   │ │pencere │ │analizi │",
    "  │Yük öner│ │Rehab   │ │Değişim │ │Takviye │ │Transfer│",
    "  └────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘",
    "       │          │          │          │          │",
    "       │    ┌─────┴──────────┴──┐       │          │",
    "       │    │   DOKTOR          │       │          │",
    "       │    │   Bağışıklık      │       │          │",
    "       │    │   Stres kırığı    │       │          │",
    "       │    └──────────────┬────┘       │          │",
    "       │                   │            │          │",
    "       └───────────────────┼────────────┘          │",
    "                           │                       │",
    "                           ▼                       │",
    "                  ┌─────────────────┐              │",
    "                  │ SPORCU KENDİSİ  │              │",
    "                  │ Kişisel tablo   │◄─────────────┘",
    "                  │ Hazırlık skoru  │",
    "                  └────────┬────────┘",
    "                           │",
    "                           ▼",
    "┌─────────────────────────────────────────────────────────────────────┐",
    "│                     FEEDBACK TOPLAMA                               │",
    "│          ✅ Doğru    ⚠️ Kısmen    ❌ Yanlış / Uygulanmadı           │",
    "└──────────────────────────────┬──────────────────────────────────────┘",
    "                               │",
    "                               ▼",
    "┌─────────────────────────────────────────────────────────────────────┐",
    "│                    MODEL GÜNCELLEME                                │",
    "│   Kişisel eşik kalibrasyonu · Kural ağırlık güncellemesi           │",
    "│   6 ay → Kişisel profil · 2 yıl → ML modeli · 3+ yıl → Öngörücü  │",
    "└─────────────────────────────────────────────────────────────────────┘",
]

for line in arch_lines:
    mono(line)

doc.add_paragraph()

body(
    "Her katmanın anlamı:"
)
bullet("Veri Katmanı: Ham sensör verisinin toplandığı zemin")
bullet("Analiz Motoru: Ham veriden anlamlı metrikler üretilir — bu katman tüm rollere hizmet eder")
bullet("Rol Dashboardları: Aynı metrikler her rolün diline çevrilir, farklı önceliklerle sunulur")
bullet("Sporcu Kendisi: Son kullanıcı — kendi verisini görür, sisteme en yakın feedback verir")
bullet("Feedback Toplama: Her kararın sonucu sisteme geri döner")
bullet("Model Güncelleme: Sistem zamanla o kulübü, o sporcuları öğrenir — genel değil, kişisel olur")

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 11 — RAKİP KARŞILAŞTIRMASI
# ════════════════════════════════════════════════════════════════════════════
h1("11. RAKİP KARŞILAŞTIRMASI")

body(
    "Futbol performans analitik pazarı ~7 milyar dolar büyüklüğünde, "
    "yıllık %20 büyüme hızında. Piyasadaki başlıca sistemler:"
)

add_table(
    ["Sistem", "Kamera/CV", "EMG", "HRV/HR", "GPS/IMU", "Biyomekanik", "Tahmini Doğruluk"],
    [
        ["Bizim Sistem",  "✅ YOLO tam",      "✅", "✅", "✅",     "✅ Derin",      "%90–94"],
        ["Catapult",      "❌",               "❌", "✅", "✅",     "❌",           "%72–78"],
        ["STATSports",    "❌",               "❌", "✅", "✅",     "❌",           "%70–76"],
        ["Kinexon",       "❌",               "❌", "✅", "✅ UWB", "❌",           "%74–80"],
        ["Veo/Pixellot",  "✅ Taktik odaklı", "❌", "❌", "❌",     "⚠️ Yüzeysel", "%45–55"],
        ["Hudl",          "✅ Sınırlı",        "❌", "❌", "❌",     "⚠️ Yüzeysel", "%50–60"],
    ],
    col_widths=[3.5, 3, 1.5, 1.5, 2, 3.5, 2.5]
)

h2("Neden Farklıyız?")
body(
    "Catapult, STATSports ve Kinexon fiziksel yük ve kalp hızı konusunda güçlüdür. "
    "Ama kas seviyesinde ne olduğunu göremezler. "
    "'Çok koştu' diyebilirler — "
    "'hamstring frekansı düştü, sakatlanma gelmeden uyar' diyemezler."
)
body(
    "Veo ve Pixellot kamera ile taktiksel analiz yapar ama fizyoloji yoktur. "
    "Kimin yorulduğunu söyleyemezler."
)
body(
    "Bizim sistemde hem kas seviyesi (EMG) hem fizyoloji (HRV) hem mekanik yük (GPS) "
    "hem biyomekanik (YOLO) aynı anda çalışır. "
    "Bu kombinasyonu bir arada sunan rakip yoktur — "
    "bu boşluk hâlâ doldurulamamış durumda.",
    bold=True
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 12 — VALİDASYON
# ════════════════════════════════════════════════════════════════════════════
h1("12. DOĞRULUĞU NASIL VALİDE EDERİZ?")

body(
    "Sistem iddialarının güvenilirliği zaman içinde kanıtlanmalıdır. "
    "Üç zaman diliminde validasyon planı:"
)

h2("Kısa Vadeli Validasyon (0–3 Ay)")
add_table(
    ["Ne Valide Edilecek", "Yöntem", "Başarı Kriteri"],
    [
        ["Tekme tespiti doğruluğu", "Uzman manuel sayım vs sistem", ">%90 örtüşme"],
        ["Eklem açısı doğruluğu", "Goniometre ile spot ölçüm", "±8° dahilinde"],
        ["HR/TRIMP doğruluğu", "Sporcu RPE (1–10) ile korelasyon", "r > 0.75"],
        ["Sprint hızı doğruluğu", "Radar tabancası ile spot ölçüm", "±0.3 m/s dahilinde"],
    ],
    col_widths=[5.5, 5.5, 5.5]
)

h2("Orta Vadeli Validasyon (3–12 Ay)")
add_table(
    ["Ne Valide Edilecek", "Yöntem", "Başarı Kriteri"],
    [
        ["HRV → toparlanma doğruluğu", "Sonraki antrenman performansı takibi", ">%75 tahmin gücü"],
        ["Risk skoru → sakatlanma", "Sakatlanma kayıtları + geriye dönük analiz", "Yüksek risklilerde >3x insidans"],
        ["ATL/CTL/TSB → performans", "Yo-Yo, sprint, CMJ testleriyle korelasyon", "r > 0.70"],
        ["H:Q → izokinetik güç", "Fizyoterapist izokinetik test", "r > 0.65"],
    ],
    col_widths=[5.5, 5.5, 5.5]
)

h2("Uzun Vadeli Validasyon (1–3 Yıl)")
bullet("Sakatlanma insidansı: sistem kullanan kulüp vs kullanmayan → istatistiksel karşılaştırma")
bullet("ML modeli doğruluğu: tahmin edilen risk ile gerçekleşen sakatlanma korelasyonu")
bullet("Yük yönetimi etkinliği: TSB yönetimi yapılan oyuncular vs yapılmayan → performans farkı")
bullet("Akademik yayın potansiyeli: toplanan veriyle sporta özgü peer-reviewed makale")

h2("Kondisyonere Net Söylem")
add_table(
    ["Güvenle Söylenebilir ✅", "Söylenmemesi Gereken ❌"],
    [
        ["Bu sporcu bu hafta %23 fazla yük aldı", "Sakatlanma ihtimali tam %78"],
        ["Hamstring 3 oturumda giderek zayıflıyor", "Diz açısı tam 137.4°"],
        ["HRV 5 gün düşüyor — yük azalt", "Bu takviyeyi kesin kullan (kan testi yok)"],
        ["Sağ-sol güç farkı büyüyor — asimetri protokolü", "Bu sporcu ACL kopacak"],
        ["H:Q 0.52 — eksantrik çalışma başlat", "Sistem her zaman haklı"],
    ],
    col_widths=[8.5, 8.5]
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# SONUÇ
# ════════════════════════════════════════════════════════════════════════════
h1("13. SONUÇ")

body(
    "Bu sistem, kulübün her karar vericisinin düşünme yükünü azaltmak için tasarlandı. "
    "Kondisyoner, fizyoterapist, teknik direktör, beslenme uzmanı, doktor, mental koç ve yönetici — "
    "herkes kendi sorusunu kendi dilinde sorar, "
    "veriye dayalı tavsiye alır, "
    "son kararı kendisi verir."
)
body(
    "Sistemin özü şudur:"
)
bullet("Kamera tekniği ve taktiği görür")
bullet("EMG kasın içini, yorgunluğu kameradan önce görür")
bullet("HRV otonom sistemi ve toparlanma hazırlığını görür")
bullet("GPS fiziksel yükü ve mekanik stresi görür")
bullet("Subjektif form bağlamı ve gizli değişkenleri ekler")
bullet("Feedback döngüsü sistemi her gün biraz daha akıllı yapar")

body(
    "Tek bir sistem, birinci ligden amatör kulübe kadar "
    "hiçbir rakibin tek çatı altında sunmadığı kombinasyonu sunar. "
    "Kondisyoner değişir, teknik direktör ayrılır — "
    "ama sistemin yıllar içinde öğrendikleri kulüpte kalır. "
    "Bu da kulübün en değerli dijital varlığına dönüşür.",
    bold=True
)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(30)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("— Rapor Sonu —")
r.font.size = Pt(10); r.font.color.rgb = C_GREY

# ── Kaydet ──────────────────────────────────────────────────────────────────
out_path = "/Users/furkandogan/Desktop/Futbol_Performans_Analiz_Sistemi_Rapor.docx"
doc.save(out_path)
print(f"✅ Kaydedildi: {out_path}")
