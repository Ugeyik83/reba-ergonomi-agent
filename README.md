# 🦺 REBA Ergonomi Risk Analiz Sistemi v5.3

**Rapid Entire Body Assessment** — İSG uzmanları için AI destekli postür analiz aracı.

## Özellikler

| Özellik | Açıklama |
|---------|----------|
| **Çoklu Fotoğraf** | Birden fazla fotoğraf, her biri ayrı analiz |
| **Video Analizi** | 2fps örnekleme, max 30sn, risk dağılımı |
| **Manuel Modifier'lar** | 10 checkbox — uzman kararı |
| **Segment Renklendirme** | Her segment risk seviyesine göre renkli |
| **Bilateral Seçim** | En görünür taraf otomatik seçilir |
| **Kamera Sınıflandırma** | Frontal/lateral/oblique tespiti |
| **Yüz Anonimleştirme** | KVKK uyumu — default açık |
| **Explainable AI** | "Neden bu skor?" açıklaması |
| **Worksheet PDF** | REBA formu formatında rapor |
| **JSON Export** | Random Forest modeli için ham veri |
| **pytest** | Tablo + açı + scoring testleri |

## Neden Manuel Modifier?

Aşağıdaki modifier'lar **otomatik tespit edilmez**, kullanıcı tarafından sidebar'dan girilir:

| Modifier | Neden Manuel? |
|----------|---------------|
| Boyun extension | 2D görüntüde boyun geriye eğilmesi güvenilir biçimde ayırt edilemiyor |
| Gövde extension | Kamera açısı bağımlılığı çok yüksek — dik duruş ile hafif geriye eğilme karıştırılıyor |
| Boyun/gövde yan eğilme | Frontal çekimlerde görünür ama lateral çekimlerde kaybolur |
| Boyun/gövde dönüş | Omuz-kalça oranı yöntemi kamera açısında %30+ hata verebilir |
| Omuz kalkış, abdüksiyon | Giysi, ekipman ve kamera açısı bu tespiti bozuyor |
| Bilek dönüş/pronasyon | 2D landmark ile supinasyon/pronasyon güvenilir değil |

**Temel açılar** (fleksiyon dereceleri) AI tarafından hesaplanır.
**Modifier'lar** her zaman uzmanın gözlemiyle girilir.

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
reba_core.py      → REBA hesaplama motoru
reba_visual.py    → İskelet overlay + PDF rapor
reba_agent.py     → Streamlit arayüzü
data_export.py    → JSON dışa aktarım
tests/
  test_reba_core.py → pytest unit testler
```

## Test

```bash
pytest tests/ -v --cov=reba_core --cov-report=term-missing
```

Hedef: %60+ coverage

## Kurulum

### Streamlit Cloud
1. GitHub'a push et
2. [share.streamlit.io](https://share.streamlit.io) → Deploy
3. Advanced Settings → Python 3.11

### Lokal
```bash
pip install -r requirements.txt
streamlit run reba_agent.py
```

## Sınırlılıklar

- 2D görüntüden açı tahmini ±3-5° doğruluk payı içerir
- Kamera açısı sonucu etkiler — ideal: 45° oblique, 2m mesafe, bel hizası
- Bilek pronasyon/supinasyon 2D'de güvenilir değil
- Profesyonel ergonomi değerlendirmesinin yerini tutmaz

## Referans

Hignett, S. & McAtamney, L. (2000). Rapid Entire Body Assessment (REBA).
*Applied Ergonomics*, 31(2), 201-205.
