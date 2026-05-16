"""
github_advisor.py — AI Destekli ISG Öneri Motoru v2
GitHub Models GPT-4o Vision üzerinden REBA sonuçlarına
+ fotoğraf analizi ile bağlama duyarlı ISG önerileri üretir.

secrets.toml:
    GITHUB_TOKEN = "ghp_xxxx"
"""

import json
import os
import base64
import cv2
import numpy as np
from openai import OpenAI

GITHUB_BASE_URL = "https://models.inference.ai.azure.com"
DEFAULT_MODEL   = "gpt-4o"  # Vision destekli

SYSTEM_PROMPT = """Sen 6331 sayılı İş Sağlığı ve Güvenliği Kanunu ile
ISO 45001 standardına hâkim, deneyimli bir ISG uzmanısın.

Sana hem REBA ergonomi analiz verileri hem de çalışanın fotoğrafı verilecek.
Fotoğrafa bakarak ne iş yaptığını, hangi postürde durduğunu,
hangi ekipman veya ortamda çalıştığını anla.

ÖNEMLİ KURALLAR:
- Fotoğraftaki gerçek sahneye göre yorum yap (palet, makine, raf, zemin vb.)
- O işe özgü, uygulanabilir öneriler üret — genel klişe yazma
- Gördüğün ortama uymayan öneriler verme (ör: monitör yoksa monitörden bahsetme)
- Her öneri o iş istasyonunda gerçekten yapılabilir olsun
- Türkçe yaz

Yanıtını YALNIZCA aşağıdaki JSON formatında döndür:

{
  "risk_ozeti": "fotoğraftaki sahneye özgü 1-2 cümle değerlendirme",
  "kritik_bulgular": ["görsel bulgulara dayalı madde1", "madde2"],
  "oncelikli_mudahaleler": [
    {
      "mudahale": "o sahneye özgü somut adım",
      "sure": "hemen / 1 hafta / 1 ay",
      "maliyet": "düşük / orta / yüksek"
    }
  ],
  "uzun_vadeli_onlemler": ["önlem1", "önlem2"],
  "tekrar_degerlendirme": "ne zaman yeniden REBA yapılmalı",
  "madde_listesi": ["öneri1", "öneri2", "öneri3"]
}"""


class RebaAdvisor:
    def __init__(self, token: str = None, model: str = DEFAULT_MODEL):
        api_key = token or os.getenv("GITHUB_TOKEN")
        if not api_key:
            raise ValueError("GitHub token bulunamadı. secrets.toml → GITHUB_TOKEN")
        self.client = OpenAI(api_key=api_key, base_url=GITHUB_BASE_URL)
        self.model = model

    def analyze(self, reba_skor, form_bilgi: dict = None, overlay_img=None) -> dict:
        """
        reba_skor  : REBASkoru objesi
        form_bilgi : {'bolum', 'is_adimi', 'is_istasyonu', 'yuk_kg', ...}
        overlay_img: BGR numpy array (OpenCV) — fotoğraf varsa vision kullanılır
        """
        metin_prompt = self._skor_to_prompt(reba_skor, form_bilgi)

        if overlay_img is not None:
            # Vision modu — fotoğrafı base64'e çevir
            messages = self._vision_messages(metin_prompt, overlay_img)
        else:
            # Sadece metin modu
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": metin_prompt},
            ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()

        # Markdown fence temizle
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    def aksiyon_listesi(self, reba_skor, form_bilgi: dict = None, overlay_img=None) -> list:
        """PDF için bullet-point listesi. Hata durumunda fallback devreye girer."""
        try:
            result = self.analyze(reba_skor, form_bilgi, overlay_img)
            return result.get("madde_listesi", [])
        except Exception:
            return _kural_tabanli_oneriler(reba_skor)

    def _vision_messages(self, metin_prompt: str, img: np.ndarray) -> list:
        """BGR görüntüyü base64 JPEG'e çevirip vision mesajı oluştur."""
        # Boyutu küçült — token tasarrufu (max 800px uzun kenar)
        h, w = img.shape[:2]
        max_px = 800
        if max(h, w) > max_px:
            scale = max_px / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        # Yüz blur kaldır — model sahneyi görmeli (isteğe bağlı)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        b64 = base64.b64encode(buf.tobytes()).decode()

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",   # "low" = hızlı + ucuz, "high" = detaylı
                        }
                    },
                    {
                        "type": "text",
                        "text": metin_prompt,
                    }
                ]
            }
        ]

    def _skor_to_prompt(self, skor, form_bilgi: dict = None) -> str:
        a = skor.acılar
        fb = form_bilgi or {}

        # İş bağlamı
        is_adimi     = fb.get('is_adimi') or fb.get('is_adımı') or '—'
        bolum        = fb.get('bolum', '—')
        is_istasyonu = fb.get('is_istasyonu', '—')
        yuk_kg       = fb.get('yuk_kg', 0)

        if a is None:
            return (f"İş Adımı: {is_adimi} | Bölüm: {bolum}\n"
                    f"REBA final skoru: {skor.final_skor}/15. ISG önerisi üret.")

        diz_max = max(a.diz_flexion_sol, a.diz_flexion_sag)

        mod_boyun, mod_govde, mod_ukol = [], [], []
        if a.boyun_extension:     mod_boyun.append("extension")
        if a.boyun_yan_egim > 15: mod_boyun.append("yana eğim")
        if a.boyun_donus:         mod_boyun.append("dönüş")
        if a.govde_extension:     mod_govde.append("extension")
        if a.govde_yan_egim > 10: mod_govde.append("yana eğim")
        if a.govde_donus:         mod_govde.append("dönüş")
        if a.omuz_kalkmis:        mod_ukol.append("omuz kalkış")
        if a.kol_abdukte:         mod_ukol.append("abdüksiyon")
        if a.kol_destekli:        mod_ukol.append("destekli")

        return f"""İŞ BAĞLAMI:
  Bölüm        : {bolum}
  İş İstasyonu : {is_istasyonu}
  İş Adımı     : {is_adimi}
  Yük          : {yuk_kg} kg

REBA ANALİZ SONUCU:
  Nihai Skor   : {skor.final_skor}/15 — {skor.risk_seviyesi}
  Aksiyon      : {skor.aksiyon}

SEGMENT SKORLARI:
  Boyun   : {skor.boyun_skoru} | {a.boyun_flexion:.0f}° | {', '.join(mod_boyun) or '—'}
  Gövde   : {skor.govde_skoru} | {a.govde_flexion:.0f}° | {', '.join(mod_govde) or '—'}
  Bacak   : {skor.bacak_skoru} | diz {diz_max:.0f}°
  Üst Kol : {skor.ust_kol_skoru} | {a.ust_kol_aci:.0f}° | {', '.join(mod_ukol) or '—'}
  Alt Kol : {skor.alt_kol_skoru} | {a.alt_kol_aci:.0f}°
  Bilek   : {skor.bilek_skoru} | {a.bilek_aci:.0f}° | {'dönüş var' if a.bilek_donus else 'dönüş yok'}

YÜKLER:
  Yük: +{skor.yuk_skoru} | Tutma: +{skor.tutma_skoru} | Aktivite: +{skor.aktivite_skoru}

Fotoğraftaki sahneyi de dikkate alarak JSON formatında ISG önerilerini döndür."""


# ── Kural tabanlı fallback ────────────────────────────────────

def _kural_tabanli_oneriler(skor) -> list:
    oneriler = []
    a = skor.acılar
    if a and skor.boyun_skoru >= 3:
        oneriler.append(f"Boyun {a.boyun_flexion:.0f}° fleksiyon — çalışma yüksekliğini artırın.")
    if a and skor.govde_skoru >= 4:
        oneriler.append(f"Gövde {a.govde_flexion:.0f}° eğilim — malzeme erişim mesafesini kısaltın.")
    if a and skor.bacak_skoru >= 3:
        diz = max(a.diz_flexion_sol, a.diz_flexion_sag)
        oneriler.append(f"Diz {diz:.0f}° fleksiyon — kaldırıcı ekipman kullanın.")
    if a and skor.ust_kol_skoru >= 4:
        oneriler.append(f"Üst kol {a.ust_kol_aci:.0f}° — erişim noktasını omuz altına indirin.")
    if a and skor.bilek_skoru >= 3:
        oneriler.append(f"Bilek {a.bilek_aci:.0f}° sapma — ergonomik tutma aparatı kullanın.")
    if skor.yuk_skoru >= 2:
        oneriler.append("10 kg+ taşıma — mekanik yardımcı ekipman ekleyin.")
    if skor.aktivite_skoru >= 2:
        oneriler.append("Tekrarlı hareket — iş rotasyonu veya mikro mola uygulayın.")
    return oneriler or ["Mevcut risk seviyesi için genel izleme yeterlidir."]


# ── Dışa açılan fonksiyonlar ─────────────────────────────────

def get_ai_oneriler(reba_skor, github_token: str,
                    form_bilgi: dict = None, overlay_img=None) -> dict:
    return RebaAdvisor(token=github_token).analyze(reba_skor, form_bilgi, overlay_img)


def get_aksiyon_listesi(reba_skor, github_token: str,
                        form_bilgi: dict = None, overlay_img=None) -> list:
    return RebaAdvisor(token=github_token).aksiyon_listesi(reba_skor, form_bilgi, overlay_img)
