# Spor video analizi

Python, Streamlit ve MediaPipe ile yerel video analizi. Mevcut çalışan branş taekwondo; voleybol sıradaki öncelik, basketbol daha sonra. Gerçek video ölçüm doğrulaması henüz yapılmadı.

## Çalıştırma

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

CLI: `.venv/bin/python main.py --help`. Varsayılan giriş `data/videos/sample.mp4`, çıktılar `data/output/`. Arayüz analizleri şimdilik geçici yerel oturumlarda tutulur; kalıcı kayıt 4. adımda yapılacak.

## Yapı

- `src/core/`: bağımsız sayısal araçlar ve veri tipleri.
- `src/adapters/`: poz kestirimi, çizim ve dosya erişimi.
- `src/sports/`: branşa özel analiz ve rapor hazırlığı.
- `ui/components/`: ortak yükleyici, video, zaman çizelgesi, metrik, kalite ve karşılaştırma bileşenleri.
- `ui/sports/`: branş ekranları ve oturum akışları.
- `cli/`: komut satırı girişleri.
- `tests/`: mimari, regresyon ve arayüz kontrolleri.
- `data/`: Git dışında tutulan yerel videolar ve çıktılar.

Çalışma planı ve devam notları: [upgrade.md](upgrade.md). Mimari sınırlar: [docs/architecture.md](docs/architecture.md).

## Kontrol

```sh
.venv/bin/python -m unittest discover -v
```

Kod testleri ölçüm doğruluğunu kanıtlamaz. Görsel asimetri kuvvet farkı değildir; açı hızları sprint hızı değildir.
