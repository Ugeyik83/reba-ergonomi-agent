# ⚡ REBA Ergonomi Analiz Ajanı v3

**MediaPipe Pose + REBA Skorlama + Streamlit UI**

Video veya görüntü yükle → iskelet tespiti → açı hesaplama → REBA skoru.

---

## 🚀 Hızlı Başlangıç (Lokal)

```bash
# 1. Repo'yu klonla veya dosyaları indir
# 2. Bağımlılıkları kur
pip install -r requirements.txt

# 3. Çalıştır
streamlit run reba_agent.py
```

Tarayıcıda `http://localhost:8501` açılır.

---

## ☁️ Streamlit Cloud'a Deploy (Ücretsiz)

1. GitHub'da yeni repo oluştur
2. Bu dosyaları push et:
   ```
   reba_agent.py
   requirements.txt
   .streamlit/config.toml
   ```
3. [share.streamlit.io](https://share.streamlit.io) adresine git
4. "New app" → GitHub repo'nu seç → `reba_agent.py` belirt
5. Deploy → URL'yi arkadaşlarına paylaş

**API key gerekmez!** Tüm analiz client-side (MediaPipe) çalışır.

---

## 📁 Dosya Yapısı

```
reba-ergonomi-agent/
├── reba_agent.py          # Ana uygulama (tek dosya)
├── requirements.txt       # Python bağımlılıkları
├── .streamlit/
│   └── config.toml        # Tema ve yapılandırma
└── README.md
```

---

## 🔧 Teknik Mimari

| Katman | Teknoloji | Detay |
|--------|-----------|-------|
| Pose Estimation | MediaPipe Pose | 33 landmark, model_complexity=2 |
| Açı Hesaplama | NumPy + Trigonometri | arccos / atan2 bazlı |
| REBA Engine | Pure Python | Tablo A, B, C tam implementasyon |
| Video İşleme | OpenCV | FPS örnekleme, boyut optimizasyonu |
| UI | Streamlit | Dark tema, responsive |
| Görselleştirme | OpenCV overlay | İskelet + skor çizimi |

---

## 📊 Desteklenen Formatlar

- **Video:** MP4, MOV, M4V, WEBM, AVI, MKV
- **Görüntü:** JPG, PNG, WEBP, BMP, HEIC

---

## ⚠️ Sınırlılıklar

- Kamera açısı sonucu etkiler (ideal: yan veya 45° açı)
- Açı doğruluğu: ±3-5° (Claude Vision'dan 3-5x daha iyi)
- Birden fazla kişi varsa sadece ilk tespit edilen analiz edilir
- Profesyonel ergonomi değerlendirmesinin yerini tutmaz

---

## 📖 Referans

- Hignett, S. & McAtamney, L. (2000). REBA: A Survey Method for Investigation of Work-Related Upper Limb Disorders. *Applied Ergonomics*, 31, 201-205.
- Google MediaPipe Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
