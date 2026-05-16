"""
data_export.py — Veri Dışa Aktarım Modülü v5.2
Analiz sonuçlarını JSON formatında dışa aktar.
Çıktı: Random Forest kaza tahmin modeli için feature kaynağı.
"""

import json
from datetime import datetime
from typing import List, Optional

from reba_core import FotoSonuc, REBASkoru


def skor_to_dict(s: REBASkoru) -> dict:
    """REBASkoru → JSON-serializable dict."""
    a = s.acılar
    return {
        "final_skor": s.final_skor,
        "risk_seviyesi": s.risk_seviyesi,
        "segment_skorlari": {
            "boyun": s.boyun_skoru,
            "govde": s.govde_skoru,
            "bacak": s.bacak_skoru,
            "ust_kol": s.ust_kol_skoru,
            "alt_kol": s.alt_kol_skoru,
            "bilek": s.bilek_skoru,
        },
        "ara_skorlar": {
            "tablo_a": s.tablo_a,
            "skor_a": s.skor_a,
            "tablo_b": s.tablo_b,
            "skor_b": s.skor_b,
            "skor_c": s.skor_c,
        },
        "manuel_girdiler": {
            "yuk_skoru": s.yuk_skoru,
            "tutma_skoru": s.tutma_skoru,
            "aktivite_skoru": s.aktivite_skoru,
        },
        "acilar": {
            "boyun_flexion": round(a.boyun_flexion, 1) if a else None,
            "govde_flexion": round(a.govde_flexion, 1) if a else None,
            "diz_flexion_max": round(max(a.diz_flexion_sol, a.diz_flexion_sag), 1) if a else None,
            "ust_kol_aci": round(a.ust_kol_aci, 1) if a else None,
            "alt_kol_aci": round(a.alt_kol_aci, 1) if a else None,
            "bilek_aci": round(a.bilek_aci, 1) if a else None,
        },
        "modifier_flags": {
            "boyun_extension": a.boyun_extension if a else False,
            "boyun_yan_egim": a.boyun_yan_egim > 0 if a else False,
            "boyun_donus": a.boyun_donus if a else False,
            "govde_extension": a.govde_extension if a else False,
            "govde_yan_egim": a.govde_yan_egim > 0 if a else False,
            "govde_donus": a.govde_donus if a else False,
            "omuz_kalkmis": a.omuz_kalkmis if a else False,
            "kol_abdukte": a.kol_abdukte if a else False,
            "bilek_donus": a.bilek_donus if a else False,
            "bilateral_destek": a.bilateral_destek if a else True,
        },
        "meta": {
            "guven": round(a.guven, 3) if a else None,
            "analiz_tarafi": a.analiz_tarafi if a else None,
            "kamera_acisi": a.kamera_acisi if a else None,
            "bacak_gozukuyor": a.bacak_gozukuyor if a else True,
        },
    }


def foto_analiz_export(
    form_bilgi: dict,
    foto_sonuclari: List[FotoSonuc],
) -> bytes:
    """
    Fotoğraf analizi sonuçlarını JSON olarak dışa aktar.
    Her kayıt bir fotoğrafı temsil eder.
    Random Forest için feature vector'ü içerir.
    """
    kayitlar = []
    for fs in foto_sonuclari:
        if fs.skor is None:
            continue
        kayit = {
            "analiz_turu": "fotograf",
            "zaman_damgasi": datetime.now().isoformat(),
            "form": {
                "bolum": form_bilgi.get("bolum", ""),
                "is_istasyonu": form_bilgi.get("is_istasyonu", ""),
                "is_adimi": form_bilgi.get("is_adimi", ""),
                "analist": form_bilgi.get("analist", ""),
                "tarih": form_bilgi.get("tarih", ""),
            },
            "dosya_adi": fs.dosya_adi,
            "reba": skor_to_dict(fs.skor),
        }
        kayitlar.append(kayit)

    return json.dumps(kayitlar, ensure_ascii=False, indent=2).encode("utf-8")


def video_analiz_export(
    form_bilgi: dict,
    video_sonuclari: list,
    sure_sn: float,
    video_adi: str = "",
) -> bytes:
    """
    Video analizi sonuçlarını JSON olarak dışa aktar.
    Her kayıt bir frame'i temsil eder.
    Ek olarak video bazlı özet istatistikler içerir.
    """
    tum_skorlar = [v["skor"] for v in video_sonuclari]
    if not tum_skorlar:
        return b"[]"

    ort = sum(tum_skorlar) / len(tum_skorlar)
    sorted_s = sorted(tum_skorlar)
    p90 = sorted_s[min(int(len(sorted_s) * 0.9), len(sorted_s) - 1)]
    yuksek_risk_kare = sum(1 for s in tum_skorlar if s >= 8)
    yuksek_risk_sure = yuksek_risk_kare / 2  # 2fps

    ozet = {
        "analiz_turu": "video",
        "zaman_damgasi": datetime.now().isoformat(),
        "form": {
            "bolum": form_bilgi.get("bolum", ""),
            "is_istasyonu": form_bilgi.get("is_istasyonu", ""),
            "is_adimi": form_bilgi.get("is_adimi", ""),
            "analist": form_bilgi.get("analist", ""),
            "tarih": form_bilgi.get("tarih", ""),
        },
        "video_adi": video_adi,
        "sure_sn": round(sure_sn, 1),
        "toplam_kare": len(tum_skorlar),
        "istatistik": {
            "ortalama_reba": round(ort, 2),
            "p90_reba": p90,
            "en_yuksek": max(tum_skorlar),
            "en_dusuk": min(tum_skorlar),
            "yuksek_risk_kare": yuksek_risk_kare,
            "yuksek_risk_sure_sn": round(yuksek_risk_sure, 1),
            "yuksek_risk_yuzde": round(yuksek_risk_kare / len(tum_skorlar) * 100, 1),
        },
        "kareler": [
            {
                "zaman_sn": round(v["zaman"], 2),
                "reba": skor_to_dict(v["skor_obj"]) if "skor_obj" in v else {"final_skor": v["skor"]},
            }
            for v in video_sonuclari
        ],
    }

    return json.dumps(ozet, ensure_ascii=False, indent=2).encode("utf-8")
