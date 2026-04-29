# Taekwondo Fatigue Analysis — Proje Rehberi

## Proje Ne Yapıyor
Video üzerinden eklem açısı, hız, tekme tespiti ve yorgunluk analizi.
Pre/post antrenman videosu karşılaştırarak yorgunluk indeksi çıkarıyor.

**Stack:** Python 3.12 · MediaPipe · OpenCV · Streamlit · Plotly · SciPy

```bash
.venv/bin/streamlit run app.py        # dashboard
python main.py --input video.mp4      # CLI — tek video analizi
python cli/compare.py --pre-events pre.csv --post-events post.csv
```

---

## Dosya Haritası

### Uygulama giriş noktaları
| Dosya | Ne yapar |
|-------|----------|
| `app.py` | Streamlit router — sayfa seçimi, 20 satır |
| `main.py` | CLI — tek video analizi |
| `cli/compare.py` | CLI — pre/post kick CSV karşılaştırma |

### Sayfa modülleri (`pages/`)
| Dosya | Ne yapar |
|-------|----------|
| `pages/video_analysis.py` | Video Analizi sayfası — 11 sekme (pre/post karşılaştırma) |
| `pages/emg_sync.py` | EMG Sync sayfası — harici EMG CSV senkronizasyonu |

### UI bileşenleri (`ui/`)
| Dosya | Ne yapar |
|-------|----------|
| `ui/charts.py` | Tüm Plotly grafik fonksiyonları (overlay, gauge, per_kick_trend…) |
| `ui/components.py` | `video_player`, `kick_video_section`, `phase_bars`, `sensor_stats` |
| `ui/report.py` | `render_athlete_report` — Sporcu Raporu sekmesi |

### Çekirdek analiz (`src/`)
| Dosya | Ne yapar |
|-------|----------|
| `src/pipeline.py` | `run_analysis()` — her şeyi birleştiren ana fonksiyon |
| `src/pose_runner.py` | `MediaPipePoseRunner` — MediaPipe arayüzü |
| `src/metrics.py` | Eklem açısı, Savitzky-Golay hız/ivme, normalize ayak hızı |
| `src/events.py` | Kick tespiti (peak tabanlı) + 3 fazlı segmentasyon |
| `src/fatigue.py` | `FATIGUE_METRICS`, `FATIGUE_WEIGHTS`, `compute_fatigue()` |
| `src/sensors.py` | `generate_emg`, `generate_nirs`, `generate_interpretation` |
| `src/utils.py` | Paylaşımlı: `fill_none_forward`, `moving_average`, `events_mean`, `pct_change`… |
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
                    → sensors.py  → sentetik EMG + NIRS serileri
                    → draw.py     → annotated video
                    → exporter.py → frame_metrics.csv + kick_events.csv
```

`run_analysis()` tek çağrıyla hepsini yapar, `AnalysisResult` döner.

---

## Önemli Teknik Detaylar

**Pose backend:** Sadece `MediaPipePoseRunner` — 33 landmark, visibility skoru.
`process_frame(bgr) → (Keypoints2D | None, raw)` ve `get_confidence(raw) → float` arayüzü.

**Kick tespiti (`events.py`):**
- Normalize ayak yüksekliği (torso uzunluğuna göre) peak'i bul
- Filtreler: min diz ROM (≥12°), min peak yükseklik (≥-0.5), min/max süre
- 3 faz: yüklenme (start→chamber) / uzatma (chamber→extension) / geri çekim (extension→end)

**Yorgunluk indeksi (`src/fatigue.py`):**
- 7 metriğin ağırlıklı ortalaması → 0-100 arası skor
- Ağırlıklar: diz ROM ×0.25, peak hız ×0.25, peak hıza süre ×0.15, tekme yüksekliği ×0.15, ayak hızı ×0.10, tekme süresi ×0.05, ort. hız ×0.05

**Sayfa yapısı:**
- Video Analizi: 11 sekme — Videolar / Açı / Hız / Yorgunluk / Tekme Bazlı / Faz / Asimetri / İstatistik / Export / Sensör / Sporcu Raporu
- EMG Sync: EMG CSV + frame CSV → overlay + per-kick RMS

---

## Kurallar
- Türkçe UI, Türkçe label'lar
- Edit tool kullan, sadece değişen kısım
- Her değişikten sonra `python3 -c "import ast; ast.parse(open('...').read())"` syntax check
- Kısa cevap ver, özet yazma
