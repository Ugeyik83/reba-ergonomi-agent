# 🦺 REBA Ergonomi Risk Analiz Ajanı v4.0

**İSG Uzmanları için AI Destekli Hızlı Tüm Vücut Değerlendirme Aracı**

---

## 📋 Nedir?

REBA (Rapid Entire Body Assessment), çalışanların kas-iskelet sistemi bozukluklarına yol açan ergonomik risk faktörlerini değerlendiren, uluslararası kabul görmüş bir yöntemdir *(Hignett & McAtamney, 2000)*.

Bu araç, geleneksel kâğıt tabanlı REBA formunu **yapay zeka destekli görüntü analizi** ile birleştirerek İSG uzmanlarının saha değerlendirmelerini hızlandırır.

---

## 🔬 Nasıl Çalışır?

```
Video/Fotoğraf → MediaPipe Pose → Eklem Açıları → REBA Skorlama → Rapor
```

1. **Medya Yükleme** — Video (≤15 sn) veya fotoğraf yüklenir
2. **Manuel Girdi** — Yük, tutma kalitesi ve aktivite bilgileri girilir
3. **Pose Tespiti** — Google MediaPipe ile 33 eklem noktası tespit edilir
4. **Açı Hesaplama** — Trigonometrik yöntemle vücut segment açıları hesaplanır
5. **REBA Puanlama** — Tablo A, B, C üzerinden nihai skor hesaplanır
6. **Raporlama** — Ekranda sonuç gösterilir, PDF/JSON export edilir

---

## 🎯 Özellikler

| Özellik | Detay |
|---------|-------|
| **Video Analizi** | Max 15 saniye, saniyede 3 kare |
| **Görüntü Analizi** | Tekli fotoğraf desteği |
| **REBA Tablo A** | Boyun + Gövde + Bacak |
| **REBA Tablo B** | Üst Kol + Alt Kol + Bilek |
| **Yük Analizi** | kg girişi, otomatik skor, ani kuvvet seçeneği |
| **Tutma Analizi** | 4 seviye açılır liste |
| **Aktivite Analizi** | 3 checkbox (statik, tekrarlı, dengesiz) |
| **En Riskli Kare** | Birden fazla maksimum skor zamanı gösterilir |
| **İskelet Overlay** | Renk kodlu iskelet çizimi |
| **PDF Raporu** | Tam form bilgileri + kare bazlı analiz |
| **JSON Export** | Ham veri indirme |
| **Form Bilgileri** | Bölüm, İş İstasyonu, İş Adımı, Tarih |

---

## 🚦 Risk Skalası

| Skor | Risk Seviyesi | Önlem |
|------|--------------|-------|
| 1 | 🟢 Önemsiz | Herhangi bir önlem gerekmez |
| 2-3 | 🟡 Düşük | Gerekirse iyileştirme yap |
| 4-7 | 🟠 Orta Seviyeli | Ayrıntılı incele, değişiklik planla |
| 8-10 | 🔴 Yüksek | Araştırma yap ve aksiyon al |
| 11+ | ⛔ Çok Yüksek | Derhal revize et |

---

## 🚀 Kurulum ve Çalıştırma

### Lokal (Terminal)
```bash
pip install -r requirements.txt
streamlit run reba_agent.py
```

### Streamlit Cloud (Ücretsiz Deploy)
1. Bu repo'yu fork et
2. [share.streamlit.io](https://share.streamlit.io) → "New app"
3. `reba_agent.py` dosyasını seç
4. Deploy → URL'yi paylaş

---

## 📁 Dosya Yapısı

```
reba-ergonomi-agent/
├── reba_agent.py          # Ana uygulama
├── requirements.txt       # Python bağımlılıkları
├── packages.txt           # Sistem kütüphaneleri (Debian)
└── .streamlit/
    └── config.toml        # Tema ve sunucu ayarları
```

---

## ⚠️ Sınırlılıklar ve Dikkat Edilmesi Gerekenler

- AI açı tahmini **±3-5°** doğruluk payı içerir
- Kamera açısı sonucu doğrudan etkiler — **yan veya 45° açı** ideal
- Düşük ışık, aşırı örtme veya hızlı hareket tespiti zorlaştırır
- Bu araç profesyonel ergonomi uzmanı değerlendirmesinin **yerini tutmaz**; tamamlayıcı niteliktedir
- **Yük, tutma ve aktivite** bilgileri manuel girilmelidir — AI bu parametreleri hesaplamaz

---

## 📚 Referans

> Hignett, S. & McAtamney, L. (2000). Rapid Entire Body Assessment (REBA).
> *Applied Ergonomics*, 31(2), 201-205.

---

## 📄 Lisans

Kurumsal iç kullanım için geliştirilmiştir.
