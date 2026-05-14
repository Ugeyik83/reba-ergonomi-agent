# 🦺 REBA Ergonomi Risk Analiz Sistemi v5.2

**Rapid Entire Body Assessment** — İSG uzmanları için AI destekli çoklu fotoğraf ve video postür analiz aracı.

## Özellikler

| Özellik | Açıklama |
|---------|----------|
| **Çoklu Fotoğraf** | Birden fazla fotoğraf yükle, her biri ayrı analiz edilir |
| **Segment Renklendirme** | Her vücut segmenti risk seviyesine göre renklendirilir |
| **Bilateral Seçim** | En görünür taraf otomatik seçilir |
| **Annotation Modları** | Minimal / Standard / Debug / Expert |
| **Explainable AI** | "Neden bu skor?" detaylı açıklama |
| **Extension Kontrolü** | Boyun ve gövde geriye eğilme tespiti |
| **Worksheet PDF** | REBA formu formatında profesyonel rapor |
| **Aksiyon Önerileri** | Segment bazlı mühendislik kontrol önerileri |

## REBA Risk Skalası

| Skor | Risk | Aksiyon |
|------|------|---------|
| 1 | Önemsiz | Önlem gerekmez |
| 2-3 | Düşük | Gerekirse iyileştirme |
| 4-7 | Orta | Ayrıntılı incele, değişiklik planla |
| 8-10 | Yüksek | Araştırma yap ve aksiyon al |
| 11+ | Çok Yüksek | Derhal revize et |

## Modüler Yapı

```
reba_core.py      → REBA hesaplama motoru (açılar, skorlama, tablolar)
reba_visual.py    → İskelet overlay + PDF rapor
reba_agent.py     → Streamlit arayüzü
```

## Kurulum

### Streamlit Cloud
1. GitHub'a push et
2. [share.streamlit.io](https://share.streamlit.io) → Deploy
3. Advanced Settings → Python 3.11 seç

### Lokal
```bash
pip install -r requirements.txt
streamlit run reba_agent.py
```

## Sınırlılıklar

- 2D görüntüden açı tahmini ±3-5° doğruluk payı içerir
- Kamera açısı sonucu etkiler — ideal: çalışanın yan profilinden, 2m mesafe
- Bilek pronasyon/supinasyon 2D'de güvenilir değildir
- Profesyonel ergonomi değerlendirmesinin yerini tutmaz

## Referans

Hignett, S. & McAtamney, L. (2000). Rapid Entire Body Assessment (REBA).
*Applied Ergonomics*, 31(2), 201-205.
