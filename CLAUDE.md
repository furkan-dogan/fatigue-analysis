# Proje rehberi

Güncel çalışma kuralları [AGENTS.md](AGENTS.md), ayrıntılı plan ve devam durumu
[upgrade.md](upgrade.md) içindedir. Başlamadan ikisini oku.

## Giriş noktaları

```bash
.venv/bin/streamlit run app.py
.venv/bin/python main.py --input videos/pre_video.mp4
.venv/bin/python -m cli.compare --pre-events pre.csv --post-events post.csv
.venv/bin/python -m unittest discover -v
```

## Dosya haritası

- `ui/app.py`, `ui/theme.py`: uygulama kabuğu ve stil.
- `ui/components/`: ortak video/grafik bileşenleri.
- `ui/sports/taekwondo/`: mevcut sayfa, rapor ve tekme sunumu.
- `src/core/`: backend bağımsız pose tipi, geometri, sinyal ve sayısal araçlar.
- `src/adapters/`: MediaPipe, çizim, klip, CSV bağlantıları.
- `src/sports/taekwondo/pipeline.py`: mevcut `run_analysis()` ve `AnalysisResult`.
- `src/sports/taekwondo/`: tekme tespiti, metrikler, yorgunluk ve sensör kodu.
- `src/sports/volleyball/`, `ui/sports/volleyball/`: voleybol için açılan modüller.
- `src/sports/basketball/`, `ui/sports/basketball/`: basketbol için açılan modüller.

Taekwondo davranışı yapısal taşıma sırasında korunmuştur; bilinen ölçüm sorunları çözülmüş değildir.
Eski rehber yalnızca `docs/archive/CLAUDE.taekwondo.md` altında tarihsel kayıt olarak bulunur.
