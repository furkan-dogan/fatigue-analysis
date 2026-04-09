# Taekwondo Fatigue Analysis — Proje Rehberi

## Proje Ne Yapıyor
Video üzerinden eklem açısı, hız, tekme tespiti ve yorgunluk analizi.
Pre/post antrenman videosu karşılaştırarak yorgunluk indeksi çıkarıyor.

**Stack:** Python 3.12 · MediaPipe · YOLOv8/v11-pose · OpenCV · Streamlit · Plotly · SciPy

```bash
.venv/bin/streamlit run app.py   # dashboard
python main.py --input video.mp4  # CLI
```

---

## Dosya Haritası

| Dosya | Ne yapar |
|-------|----------|
| `app.py` | Streamlit — 3 sayfa: Tek Video / Çift Video / EMG Sync |
| `main.py` | CLI — tek video analizi |
| `compare_pre_post.py` | CLI — pre/post CSV karşılaştırma |
| `src/pipeline.py` | `run_analysis()` — her şeyi birleştiren ana fonksiyon |
| `src/pose_runner.py` | `MediaPipePoseRunner` + `YOLOPoseRunner` — aynı interface |
| `src/metrics.py` | Eklem açısı, Savitzky-Golay hız/ivme, normalize ayak hızı |
| `src/events.py` | Kick tespiti (peak tabanlı) + 3 fazlı segmentasyon |
| `src/draw.py` | Video üzerine iskelet + açı paneli çizimi |
| `src/stats.py` | Cohen's d, %95 CI, effect label |
| `src/exporter.py` | Frame + event CSV yazıcı |
| `src/emg_sync.py` | EMG CSV → video frame zamanına resample |

---

## Veri Akışı

```
Video → pose_runner → Keypoints2D
                    → metrics.py  → açı, hız, ivme, ayak hızı (frame bazlı)
                    → events.py   → kick listesi (start/peak/end frame, faz süreleri, ASI)
                    → draw.py     → annotated video
                    → exporter.py → frame_metrics.csv + kick_events.csv
```

`run_analysis()` tek çağrıyla hepsini yapar, `AnalysisResult` döner.

---

## Önemli Teknik Detaylar

**Pose backend:**
- `MediaPipePoseRunner` → 33 landmark, visibility skoru
- `YOLOPoseRunner` → COCO 17 keypoint, foot_index yok → ankle fallback
- Her ikisi `process_frame(bgr) → (Keypoints2D | None, raw)` ve `get_confidence(raw) → float` interface'i

**Kick tespiti (`events.py`):**
- Normalize ayak yüksekliği (torso uzunluğuna göre) peak'i bul
- Filtreler: min diz ROM (≥20°), min peak yükseklik (≥-0.3), min/max süre
- 3 faz: yüklenme (start→chamber) / uzatma (chamber→extension) / geri çekim (extension→end)

**Yorgunluk indeksi:**
- 7 metriğin ağırlıklı ortalaması → 0-100 arası skor
- Ağırlıklar: diz ROM ×0.25, peak hız ×0.25, peak hıza süre ×0.15, tekme yüksekliği ×0.15, ayak hızı ×0.10, tekme süresi ×0.05, ort. hız ×0.05

**app.py sayfa yapısı:**
- Sayfa 1 (Tek Video): 5 tab — Annotated Video / Açı / Hız / Tekme Eventleri / Kick İnceleme
- Sayfa 2 (Çift Video): 9 tab — Videolar / Açı / Hız / Yorgunluk / Tekme Bazlı / Faz / Asimetri / İstatistik / Export
- Sayfa 3 (EMG Sync): EMG CSV + frame CSV → overlay + per-kick RMS

**YOLO kontrolü:** `app.py` başında `_YOLO_AVAILABLE` flag'i var, kurulu değilse seçenek gizlenir.

---

## Kurallar
- Türkçe UI, Türkçe label'lar
- Edit tool kullan, sadece değişen kısım
- Her değişikten sonra `python3 -c "import ast; ast.parse(open('...').read())"` syntax check
- Kısa cevap ver, özet yazma
