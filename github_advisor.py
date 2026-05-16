"""
github_advisor.py — AI Destekli ISG Öneri Motoru
GitHub Models (ücretsiz) üzerinden REBA sonuçlarına
bağlama duyarlı, segment bazlı ISG önerileri üretir.

reba_visual.py içindeki _aksiyon_onerisi() ile entegre çalışır.

Token alma:
    github.com → Settings → Developer Settings
    → Personal Access Tokens → Generate new token
    (hiçbir scope seçmeye gerek yok)

secrets.toml:
    GITHUB_TOKEN = "ghp_xxxx"
"""

import json
import os
from openai import OpenAI

GITHUB_BASE_URL = "https://models.inference.ai.azure.com"
DEFAULT_MODEL   = "gpt-4o"

SYSTEM_PROMPT = """Sen 6331 sayılı İş Sağlığı ve Güvenliği Kanunu ile
ISO 45001 standardına hâkim, deneyimli bir ISG uzmanısın.

REBA ergonomi analiz sonuçlarını değerlendirip,
sahada uygulanabilir, önceliklendirilmiş öneriler üretirsin.

Yanıtını YALNIZCA aşağıdaki JSON formatında döndür.
Başka metin, açıklama veya markdown ekleme:

{
  "risk_ozeti": "1-2 cümle genel değerlendirme",
  "kritik_bulgular": ["bulgu1", "bulgu2"],
  "oncelikli_mudahaleler": [
    {
      "mudahale": "yapılacak somut işlem",
      "sure": "hemen / 1 hafta / 1 ay",
      "maliyet": "düşük / orta / yüksek"
    }
  ],
  "uzun_vadeli_onlemler": ["önlem1", "önlem2"],
  "tekrar_degerlendirme": "ne zaman yeniden REBA yapılmalı",
  "madde_listesi": ["öneri1", "öneri2", "öneri3"]
}

"madde_listesi" → PDF'de bullet point olarak kullanılacak,
her madde max 15 kelime, somut ve ölçülebilir olsun."""


class RebaAdvisor:
    def __init__(self, token: str = None, model: str = DEFAULT_MODEL):
        api_key = token or os.getenv("GITHUB_TOKEN")
        if not api_key:
            raise ValueError(
                "GitHub token bulunamadı.\n"
                "secrets.toml → GITHUB_TOKEN = 'ghp_xxxx'"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=GITHUB_BASE_URL,
        )
        self.model = model

    def analyze(self, reba_skor) -> dict:
        """
        reba_skor: REBASkoru objesi (reba_core.py)
        Döndürür: ISG önerileri dict
        """
        prompt = self._skor_to_prompt(reba_skor)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
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
        raw = raw.strip()

        return json.loads(raw)

    def aksiyon_listesi(self, reba_skor) -> list:
        """
        PDF ve Streamlit için bullet-point listesi döndürür.
        Hata durumunda kural tabanlı fallback devreye girer.
        """
        try:
            result = self.analyze(reba_skor)
            return result.get("madde_listesi", [])
        except Exception:
            return _kural_tabanli_oneriler(reba_skor)

    def _skor_to_prompt(self, skor) -> str:
        a = skor.acılar
        if a is None:
            return f"REBA final skoru: {skor.final_skor}. Genel ISG önerisi üret."

        diz_max = max(a.diz_flexion_sol, a.diz_flexion_sag)

        mod_boyun = []
        if a.boyun_extension:     mod_boyun.append("extension")
        if a.boyun_yan_egim > 15: mod_boyun.append("yana eğim")
        if a.boyun_donus:         mod_boyun.append("dönüş")

        mod_govde = []
        if a.govde_extension:     mod_govde.append("extension")
        if a.govde_yan_egim > 10: mod_govde.append("yana eğim")
        if a.govde_donus:         mod_govde.append("dönüş")

        mod_ukol = []
        if a.omuz_kalkmis: mod_ukol.append("omuz kalkış")
        if a.kol_abdukte:  mod_ukol.append("abdüksiyon")
        if a.kol_destekli: mod_ukol.append("destekli")

        return f"""REBA Analiz Sonucu:

NİHAİ SKOR     : {skor.final_skor}/15
RİSK SEVİYESİ  : {skor.risk_seviyesi}
AKSIYON        : {skor.aksiyon}
ANALİZ TARAFI  : {a.analiz_tarafi}

SEGMENT SKORLARI:
  Boyun   : {skor.boyun_skoru} | açı: {a.boyun_flexion:.0f}° | mod: {', '.join(mod_boyun) or '—'}
  Gövde   : {skor.govde_skoru} | açı: {a.govde_flexion:.0f}° | mod: {', '.join(mod_govde) or '—'}
  Bacak   : {skor.bacak_skoru} | diz: {diz_max:.0f}°
  Üst Kol : {skor.ust_kol_skoru} | açı: {a.ust_kol_aci:.0f}° | mod: {', '.join(mod_ukol) or '—'}
  Alt Kol : {skor.alt_kol_skoru} | açı: {a.alt_kol_aci:.0f}°
  Bilek   : {skor.bilek_skoru} | açı: {a.bilek_aci:.0f}° | dönüş: {'evet' if a.bilek_donus else 'hayır'}

YÜKLER:
  Yük skoru     : +{skor.yuk_skoru}
  Tutma skoru   : +{skor.tutma_skoru}
  Aktivite skoru: +{skor.aktivite_skoru}

Tablo A: {skor.tablo_a} → Skor A: {skor.skor_a}
Tablo B: {skor.tablo_b} → Skor B: {skor.skor_b}
Tablo C: {skor.skor_c} → Final: {skor.final_skor}

JSON formatında ISG önerilerini döndür."""


# ── Kural tabanlı fallback (API çalışmazsa) ──────────────────

def _kural_tabanli_oneriler(skor) -> list:
    """GitHub Models erişilemediğinde devreye giren deterministik fallback."""
    oneriler = []
    a = skor.acılar

    if a and skor.boyun_skoru >= 3:
        oneriler.append(
            f"Boyun {a.boyun_flexion:.0f}° fleksiyon — monitörü göz hizasına getirin.")
    if a and skor.govde_skoru >= 4:
        oneriler.append(
            f"Gövde {a.govde_flexion:.0f}° eğilim — çalışma yüzeyini yükseltin.")
    if a and skor.bacak_skoru >= 3:
        diz = max(a.diz_flexion_sol, a.diz_flexion_sag)
        oneriler.append(
            f"Diz {diz:.0f}° fleksiyon — platform veya kaldırıcı ekipman kullanın.")
    if a and skor.ust_kol_skoru >= 4:
        oneriler.append(
            f"Üst kol {a.ust_kol_aci:.0f}° — alet erişimini omuz altına indirin.")
    if a and skor.bilek_skoru >= 3:
        oneriler.append(
            f"Bilek {a.bilek_aci:.0f}° sapma — ergonomik tutma aparatı kullanın.")
    if skor.yuk_skoru >= 2:
        oneriler.append("10 kg+ taşıma — mekanik yardımcı ekipman ekleyin.")
    if skor.aktivite_skoru >= 2:
        oneriler.append("Tekrarlı hareket — iş rotasyonu veya mikro mola uygulayın.")

    return oneriler or ["Mevcut risk seviyesi için genel izleme yeterlidir."]


# ── Dışa açılan fonksiyonlar ─────────────────────────────────

def get_ai_oneriler(reba_skor, github_token: str) -> dict:
    """
    Streamlit'te tam öneri sözlüğü:

        from github_advisor import get_ai_oneriler
        oneriler = get_ai_oneriler(skor, st.secrets["GITHUB_TOKEN"])
        st.write(oneriler["risk_ozeti"])
        for m in oneriler["oncelikli_mudahaleler"]:
            st.write(f"• {m['mudahale']} ({m['sure']}, {m['maliyet']})")
    """
    return RebaAdvisor(token=github_token).analyze(reba_skor)


def get_aksiyon_listesi(reba_skor, github_token: str) -> list:
    """
    reba_visual.py'da _aksiyon_onerisi() yerine kullanın:

        # ESKİ:
        aksiyonlar = _aksiyon_onerisi(s)

        # YENİ (pdf_olustur fonksiyonuna github_token parametresi ekleyin):
        from github_advisor import get_aksiyon_listesi
        aksiyonlar = get_aksiyon_listesi(s, github_token)
    """
    return RebaAdvisor(token=github_token).aksiyon_listesi(reba_skor)
