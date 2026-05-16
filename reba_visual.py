"""
reba_visual.py — Görsel Çıktı Modülü v5.3
İskelet overlay çizimi ve PDF rapor oluşturma.
#7: Segment bazlı risk renklendirme
#6: Adaptive annotation mode (minimal/standard/debug/expert)
#8: Bilateral — en görünür taraf etiketlenir
#9: Explainable AI tablosu
#10: Worksheet formatı PDF
#12: Önerilen aksiyonlar bölümü
#13: GitHub Models AI destekli öneri motoru
"""

import cv2
import numpy as np
import io
import os
import glob
from datetime import datetime
from typing import List

import mediapipe as mp

from reba_core import (
    AcilarObj, REBASkoru, FotoSonuc, risk_info,
    segment_risk_renk, SEGMENT_MAX,
    TABLO_A, TABLO_B, TABLO_C,
)

LM = mp.solutions.pose.PoseLandmark
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

ANNOTATION_MODES = {
    "minimal":  "Sadece REBA final skoru",
    "standard": "Kritik segmentler + skor",
    "debug":    "Tüm açılar + modifier'lar",
    "expert":   "Tüm biyomekanik veriler + güven",
}

SEGMENT_CONNECTIONS = {
    'boyun': [(LM.LEFT_EAR, LM.LEFT_SHOULDER), (LM.RIGHT_EAR, LM.RIGHT_SHOULDER),
              (LM.NOSE, LM.LEFT_SHOULDER), (LM.NOSE, LM.RIGHT_SHOULDER)],
    'govde': [(LM.LEFT_SHOULDER, LM.LEFT_HIP), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
              (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER), (LM.LEFT_HIP, LM.RIGHT_HIP)],
    'bacak': [(LM.LEFT_HIP, LM.LEFT_KNEE), (LM.LEFT_KNEE, LM.LEFT_ANKLE),
              (LM.RIGHT_HIP, LM.RIGHT_KNEE), (LM.RIGHT_KNEE, LM.RIGHT_ANKLE)],
    'ust_kol': [(LM.LEFT_SHOULDER, LM.LEFT_ELBOW), (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW)],
    'alt_kol': [(LM.LEFT_ELBOW, LM.LEFT_WRIST), (LM.RIGHT_ELBOW, LM.RIGHT_WRIST)],
    'bilek': [(LM.LEFT_WRIST, LM.LEFT_INDEX), (LM.RIGHT_WRIST, LM.RIGHT_INDEX),
              (LM.LEFT_WRIST, LM.LEFT_PINKY), (LM.RIGHT_WRIST, LM.RIGHT_PINKY)],
}


def _hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _etiket_yaz(img, nokta, metin, renk=(20, 20, 80)):
    x, y = nokta
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thick = 0.4, 1
    (tw, th), _ = cv2.getTextSize(metin, font, fs, thick)
    x = max(2, min(x, w - tw - 8))
    y = max(th + 4, min(y, h - 4))
    cv2.rectangle(img, (x-2, y-th-4), (x+tw+4, y+4), (255, 255, 240), -1)
    cv2.rectangle(img, (x-2, y-th-4), (x+tw+4, y+4), renk, 1)
    cv2.putText(img, metin, (x, y), font, fs, renk, thick, cv2.LINE_AA)


def yuz_blur_uygula(img: np.ndarray, landmarks, genislik: int = 60) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    yuz_noktalari = [LM.NOSE, LM.LEFT_EAR, LM.RIGHT_EAR,
                     LM.LEFT_EYE, LM.RIGHT_EYE]
    gorunen = [(int(landmarks[lm].x * w), int(landmarks[lm].y * h))
               for lm in yuz_noktalari if landmarks[lm].visibility > 0.3]
    if not gorunen:
        return out
    xs = [p[0] for p in gorunen]
    ys = [p[1] for p in gorunen]
    x1 = max(0, min(xs) - genislik)
    y1 = max(0, min(ys) - genislik)
    x2 = min(w, max(xs) + genislik)
    y2 = min(h, max(ys) + genislik)
    if x2 <= x1 or y2 <= y1:
        return out
    roi = out[y1:y2, x1:x2]
    blur_size = max(21, genislik | 1)
    blurred = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cx, cy = (x2-x1)//2, (y2-y1)//2
    cv2.ellipse(mask, (cx, cy), (cx, cy), 0, 0, 360, 255, -1)
    mask_3ch = cv2.merge([mask, mask, mask])
    out[y1:y2, x1:x2] = np.where(mask_3ch > 0, blurred, roi)
    return out


def overlay_ciz(
    img: np.ndarray,
    landmarks,
    skor: REBASkoru,
    mode: str = "standard",
    yuz_blur: bool = True,
) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    if yuz_blur:
        out = yuz_blur_uygula(out, landmarks)
    a = skor.acılar
    if a is None:
        return out

    def p(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    def v(idx):
        return landmarks[idx].visibility

    seg_renk = {
        'boyun':   _hex_to_bgr(segment_risk_renk(skor.boyun_skoru, SEGMENT_MAX['boyun'])),
        'govde':   _hex_to_bgr(segment_risk_renk(skor.govde_skoru, SEGMENT_MAX['govde'])),
        'bacak':   _hex_to_bgr(segment_risk_renk(skor.bacak_skoru, SEGMENT_MAX['bacak'])),
        'ust_kol': _hex_to_bgr(segment_risk_renk(skor.ust_kol_skoru, SEGMENT_MAX['ust_kol'])),
        'alt_kol': _hex_to_bgr(segment_risk_renk(skor.alt_kol_skoru, SEGMENT_MAX['alt_kol'])),
        'bilek':   _hex_to_bgr(segment_risk_renk(skor.bilek_skoru, SEGMENT_MAX['bilek'])),
    }

    for seg_name, conns in SEGMENT_CONNECTIONS.items():
        col = seg_renk[seg_name]
        for c in conns:
            if landmarks[c[0]].visibility > 0.4 and landmarks[c[1]].visibility > 0.4:
                cv2.line(out, p(c[0]), p(c[1]), col, 3, cv2.LINE_AA)

    for lm_idx, lm in enumerate(landmarks):
        if lm.visibility > 0.4:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(out, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, pt, 5, (80, 80, 80), 1, cv2.LINE_AA)

    s = skor.final_skor
    col_final = _hex_to_bgr(skor.renk)
    cv2.rectangle(out, (8, 8), (200, 70), (255, 255, 255), -1)
    cv2.rectangle(out, (8, 8), (200, 70), col_final, 2)
    cv2.rectangle(out, (8, 8), (200, 30), col_final, -1)
    cv2.putText(out, "REBA SKORU", (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, str(s), (14, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, col_final, 3, cv2.LINE_AA)
    cv2.putText(out, "/15", (58, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    if mode == "minimal":
        return out

    mid_omuz_x = int((landmarks[LM.LEFT_SHOULDER].x + landmarks[LM.RIGHT_SHOULDER].x) / 2 * w)
    mid_omuz_y = int((landmarks[LM.LEFT_SHOULDER].y + landmarks[LM.RIGHT_SHOULDER].y) / 2 * h)
    mid_kalca_y = int((landmarks[LM.LEFT_HIP].y + landmarks[LM.RIGHT_HIP].y) / 2 * h)

    if a.analiz_tarafi == "sol":
        elbow_lm, wrist_lm = LM.LEFT_ELBOW, LM.LEFT_WRIST
    else:
        elbow_lm, wrist_lm = LM.RIGHT_ELBOW, LM.RIGHT_WRIST

    if v(LM.NOSE) > 0.4:
        nx, ny = p(LM.NOSE)
        if mode in ("standard", "debug", "expert"):
            mod = ""
            if a.boyun_extension: mod += "+Ext"
            if a.boyun_yan_egim > 15: mod += "+YE"
            if a.boyun_donus: mod += "+D"
            _etiket_yaz(out, (nx + 8, ny - 6),
                        f"Boyun:{a.boyun_flexion:.0f}{mod} [{skor.boyun_skoru}]",
                        renk=seg_renk['boyun'])

    if mode in ("standard", "debug", "expert"):
        gy = (mid_omuz_y + mid_kalca_y) // 2
        mod = ""
        if a.govde_extension: mod += "+Ext"
        if a.govde_yan_egim > 10: mod += "+YE"
        if a.govde_donus: mod += "+D"
        _etiket_yaz(out, (mid_omuz_x + 10, gy),
                    f"Govde:{a.govde_flexion:.0f}{mod} [{skor.govde_skoru}]",
                    renk=seg_renk['govde'])

    if mode in ("debug", "expert"):
        if v(LM.LEFT_KNEE) > 0.4:
            kx, ky = p(LM.LEFT_KNEE)
            diz_max = max(a.diz_flexion_sol, a.diz_flexion_sag)
            _etiket_yaz(out, (kx + 6, ky),
                        f"Diz:{diz_max:.0f} [{skor.bacak_skoru}]",
                        renk=seg_renk['bacak'])

    if v(elbow_lm) > 0.4:
        ex, ey = p(elbow_lm)
        if mode in ("standard", "debug", "expert"):
            mod = ""
            if a.omuz_kalkmis: mod += "+OK"
            if a.kol_abdukte: mod += "+AB"
            _etiket_yaz(out, (ex + 6, ey - 8),
                        f"UKol:{a.ust_kol_aci:.0f}{mod} [{skor.ust_kol_skoru}]",
                        renk=seg_renk['ust_kol'])

    if v(wrist_lm) > 0.4:
        wx, wy = p(wrist_lm)
        if mode in ("debug", "expert"):
            _etiket_yaz(out, (wx + 6, wy - 16),
                        f"AKol:{a.alt_kol_aci:.0f} [{skor.alt_kol_skoru}]",
                        renk=seg_renk['alt_kol'])
        if mode in ("standard", "debug", "expert"):
            mod = "+D" if a.bilek_donus else ""
            _etiket_yaz(out, (wx + 6, wy + 4),
                        f"Bilek:{a.bilek_aci:.0f}{mod} [{skor.bilek_skoru}]",
                        renk=seg_renk['bilek'])

    if mode == "expert":
        _etiket_yaz(out, (w - 200, h - 30),
                    f"Guven:{a.guven:.0%} Taraf:{a.analiz_tarafi}",
                    renk=(60, 60, 60))

    return out


def overlay_to_bytes(overlay_img: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode('.jpg', overlay_img,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _turkce_font_yukle():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    FONT, FONT_BOLD = 'Helvetica', 'Helvetica-Bold'
    try:
        candidates = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
        normal = [f for f in candidates
                  if ('DejaVuSans' in f or 'LiberationSans' in f)
                  and 'Bold' not in f and 'Italic' not in f]
        bold = [f for f in candidates
                if ('DejaVuSans-Bold' in f or 'LiberationSans-Bold' in f)
                and 'Italic' not in f]
        if normal:
            pdfmetrics.registerFont(TTFont('TRF', normal[0]))
            FONT = 'TRF'
        if bold:
            pdfmetrics.registerFont(TTFont('TRFB', bold[0]))
            FONT_BOLD = 'TRFB'
        elif normal:
            pdfmetrics.registerFont(TTFont('TRFB', normal[0]))
            FONT_BOLD = 'TRFB'
    except Exception:
        pass
    return FONT, FONT_BOLD


# ════════════════════════════════════════════════════════
# #12 + #13: AKSİYON ÖNERİ MOTORU
# github_token verilirse → GitHub Models AI
# verilmezse → kural tabanlı fallback
# ════════════════════════════════════════════════════════

def _aksiyon_onerisi(skor: REBASkoru, github_token: str = None) -> List[str]:
    """
    github_token verilirse GitHub Models (GPT-4o) üzerinden
    bağlama duyarlı ISG önerileri üretir.
    Token yoksa deterministik kural motoru devreye girer.
    """
    if github_token:
        try:
            from github_advisor import get_aksiyon_listesi
            return get_aksiyon_listesi(skor, github_token)
        except Exception:
            pass  # fallback'e geç

    # Kural tabanlı fallback
    oneriler = []
    a = skor.acılar

    if a and skor.boyun_skoru >= 3:
        oneriler.append(
            f"Boyun: {a.boyun_flexion:.0f}° fleksiyon tespit edildi. "
            "Monitör/çalışma yüzeyini göz hizasına getirin.")
    if a and skor.govde_skoru >= 4:
        oneriler.append(
            f"Gövde: {a.govde_flexion:.0f}° öne eğilme. "
            "Çalışma yüksekliğini artırın veya malzeme erişim mesafesini kısaltın.")
    if a and skor.bacak_skoru >= 3:
        diz = max(a.diz_flexion_sol, a.diz_flexion_sag) if a else 0
        oneriler.append(
            f"Bacak: Diz {diz:.0f}° fleksiyon. "
            "Çömelme gerektiren işlemler için platform/kaldırıcı ekipman kullanın.")
    if a and skor.ust_kol_skoru >= 4:
        oneriler.append(
            f"Üst Kol: {a.ust_kol_aci:.0f}° açı. "
            "Malzeme/alet erişim noktasını omuz seviyesinin altına indirin.")
    if a and skor.bilek_skoru >= 3:
        oneriler.append(
            f"Bilek: {a.bilek_aci:.0f}° sapma + dönüş. "
            "Ergonomik tutma aparatı veya açılı alet kullanımını değerlendirin.")
    if skor.yuk_skoru >= 2:
        oneriler.append(
            "Yük: 10 kg üstü taşıma. Mekanik kaldırma yardımcısı kullanın.")
    if skor.aktivite_skoru >= 2:
        oneriler.append(
            "Aktivite: Tekrarlı hareket + statik pozisyon. "
            "İş rotasyonu veya mikro mola programı uygulayın.")

    return oneriler or ["Mevcut risk seviyesi için genel izleme yeterlidir."]


# ════════════════════════════════════════════════════════
# PDF RAPORU
# ════════════════════════════════════════════════════════

def pdf_olustur(
    form_bilgi: dict,
    foto_sonuclari: List[FotoSonuc],
    github_token: str = None,        # #13: AI öneri motoru için
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable, PageBreak, Image as RLImage,
    )

    gecerli = [s for s in foto_sonuclari if s.skor is not None]
    if not gecerli:
        return b""

    FONT, FONT_BOLD = _turkce_font_yukle()
    skorlar = [s.skor.final_skor for s in gecerli]
    ort = sum(skorlar) / len(skorlar)
    en_yuk_obj = max(gecerli, key=lambda x: x.skor.final_skor).skor

    KOYU  = colors.HexColor('#1e3a5f')
    ACIK  = colors.HexColor('#f0f4f8')
    CIZGI = colors.HexColor('#cbd5e1')

    def rk(skor):
        if skor <= 1:    return colors.HexColor('#dcfce7')
        elif skor <= 3:  return colors.HexColor('#ecfccb')
        elif skor <= 7:  return colors.HexColor('#fef9c3')
        elif skor <= 10: return colors.HexColor('#fee2e2')
        else:            return colors.HexColor('#ede9fe')

    def rt(skor):
        if skor <= 1:    return colors.HexColor('#16a34a')
        elif skor <= 3:  return colors.HexColor('#65a30d')
        elif skor <= 7:  return colors.HexColor('#d97706')
        elif skor <= 10: return colors.HexColor('#dc2626')
        else:            return colors.HexColor('#7c3aed')

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=1.5*cm, leftMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    ss = getSampleStyleSheet()

    def P(txt, bold=False, size=9, color=colors.black, space_after=0):
        fn = FONT_BOLD if bold else FONT
        st = ParagraphStyle('x', parent=ss['Normal'],
                            fontName=fn, fontSize=size,
                            textColor=color, spaceAfter=space_after)
        return Paragraph(txt, st)

    def ts_base():
        return TableStyle([
            ('FONTNAME',  (0,0), (-1,-1), FONT),
            ('FONTNAME',  (0,0), (-1, 0), FONT_BOLD),
            ('FONTSIZE',  (0,0), (-1,-1), 8),
            ('BACKGROUND',(0,0), (-1, 0), KOYU),
            ('TEXTCOLOR', (0,0), (-1, 0), colors.white),
            ('GRID',      (0,0), (-1,-1), 0.4, CIZGI),
            ('PADDING',   (0,0), (-1,-1), 4),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, ACIK]),
            ('VALIGN',    (0,0), (-1,-1), 'MIDDLE'),
        ])

    story = []

    story.append(P("REBA Ergonomi Risk Analiz Raporu",
                   bold=True, size=18, color=KOYU, space_after=6))
    story.append(Spacer(1, 0.1*cm))
    story.append(P(
        "Rapid Entire Body Assessment  |  AI Destekli Postür Analizi  |  v5.3",
        size=8, color=colors.HexColor('#64748b'), space_after=10))
    story.append(HRFlowable(width="100%", thickness=2, color=KOYU))
    story.append(Spacer(1, 0.3*cm))

    story.append(P("Form Bilgileri", bold=True, size=10, color=KOYU, space_after=4))
    fr = [
        [P("Bölüm", bold=True),       form_bilgi.get('bolum', '—'),
         P("Tarih", bold=True),        form_bilgi.get('tarih', '—')],
        [P("İş İstasyonu", bold=True), form_bilgi.get('is_istasyonu', '—'),
         P("Analist", bold=True),      form_bilgi.get('analist', '—')],
        [P("İş Adımı", bold=True),     form_bilgi.get('is_adimi', '—'),
         P("Oluşturma", bold=True),    datetime.now().strftime('%d.%m.%Y %H:%M')],
    ]
    tf = Table(fr, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    tf.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), FONT),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,0), (-1,-1), ACIK),
        ('GRID', (0,0), (-1,-1), 0.4, CIZGI),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(tf)
    story.append(Spacer(1, 0.3*cm))

    story.append(P("Manuel Parametreler", bold=True, size=10, color=KOYU, space_after=3))
    mp_rows = [
        ["Yük", f"+{form_bilgi.get('yuk_skoru',0)} ({form_bilgi.get('yuk_kg',0)} kg)",
         "Tutma", f"+{form_bilgi.get('tutma',0)} ({form_bilgi.get('tutma_label','')})",
         "Aktivite", f"+{form_bilgi.get('aktivite',0)}"],
    ]
    tmp = Table(mp_rows, colWidths=[2*cm, 4.5*cm, 2*cm, 5*cm, 2.5*cm, 2*cm])
    tmp.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), FONT),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('BACKGROUND', (0,0), (-1,-1), ACIK),
        ('GRID', (0,0), (-1,-1), 0.3, CIZGI),
        ('PADDING', (0,0), (-1,-1), 3),
        ('FONTNAME', (0,0), (0,0), FONT_BOLD),
        ('FONTNAME', (2,0), (2,0), FONT_BOLD),
        ('FONTNAME', (4,0), (4,0), FONT_BOLD),
    ]))
    story.append(tmp)
    story.append(Spacer(1, 0.3*cm))

    en_yuk_fs = max(gecerli, key=lambda x: x.skor.final_skor)
    sk = en_yuk_fs.skor
    ac = sk.acılar

    story.append(P("REBA Worksheet — En Yüksek Skorlu Analiz",
                   bold=True, size=10, color=KOYU, space_after=4))

    boyun_mod = []
    if ac and ac.boyun_extension: boyun_mod.append("Extension +1")
    if ac and ac.boyun_yan_egim > 15: boyun_mod.append("Yana eğme +1")
    if ac and ac.boyun_donus: boyun_mod.append("Dönüş +1")

    govde_mod = []
    if ac and ac.govde_extension: govde_mod.append("Extension +1")
    if ac and ac.govde_yan_egim > 10: govde_mod.append("Yana eğme +1")
    if ac and ac.govde_donus: govde_mod.append("Dönüş +1")

    ukol_mod = []
    if ac and ac.omuz_kalkmis: ukol_mod.append("Omuz kalkış +1")
    if ac and ac.kol_abdukte: ukol_mod.append("Abdüksiyon +1")
    if ac and ac.kol_destekli: ukol_mod.append("Destekli -1")

    taraf = ac.analiz_tarafi if ac else "—"
    diz_max = max(ac.diz_flexion_sol, ac.diz_flexion_sag) if ac else 0

    story.append(P("A Grubu: Boyun — Gövde — Bacak",
                   bold=True, size=9, color=KOYU, space_after=3))
    a_rows = [
        ["Segment", "Açı", "Skor", "Modifikatör"],
        ["Boyun",
         f"{ac.boyun_flexion:.0f}°" if ac else "—",
         str(sk.boyun_skoru),
         ", ".join(boyun_mod) or "—"],
        ["Gövde",
         f"{ac.govde_flexion:.0f}°" if ac else "—",
         str(sk.govde_skoru),
         ", ".join(govde_mod) or "—"],
        ["Bacak",
         f"Diz {diz_max:.0f}°",
         str(sk.bacak_skoru),
         "Bilateral" if (ac and ac.bilateral_destek) else "Tek ayak"],
        [P("Tablo A", bold=True), "", P(str(sk.tablo_a), bold=True), ""],
        [P("+ Yük Skoru", bold=True), "", f"+{sk.yuk_skoru}", ""],
        [P("= Skor A", bold=True), "", P(str(sk.skor_a), bold=True), ""],
    ]
    ta = Table(a_rows, colWidths=[4*cm, 3*cm, 2.5*cm, 8.5*cm])
    tsa = ts_base()
    tsa.add('TEXTCOLOR', (2,-1), (2,-1), rt(sk.skor_a))
    tsa.add('FONTSIZE', (2,-1), (2,-1), 11)
    tsa.add('BACKGROUND', (0,-1), (-1,-1), ACIK)
    ta.setStyle(tsa)
    story.append(ta)
    story.append(Spacer(1, 0.25*cm))

    story.append(P("B Grubu: Üst Kol — Alt Kol — Bilek",
                   bold=True, size=9, color=KOYU, space_after=3))
    b_rows = [
        ["Segment", "Açı", "Skor", "Modifikatör"],
        [f"Üst Kol ({taraf})",
         f"{ac.ust_kol_aci:.0f}°" if ac else "—",
         str(sk.ust_kol_skoru),
         ", ".join(ukol_mod) or "—"],
        [f"Alt Kol ({taraf})",
         f"{ac.alt_kol_aci:.0f}°" if ac else "—",
         str(sk.alt_kol_skoru),
         "60-100° arası" if sk.alt_kol_skoru == 1 else "Aralık dışı"],
        [f"Bilek ({taraf})",
         f"{ac.bilek_aci:.0f}°" if ac else "—",
         str(sk.bilek_skoru),
         "Dönüş +1" if (ac and ac.bilek_donus) else "—"],
        [P("Tablo B", bold=True), "", P(str(sk.tablo_b), bold=True), ""],
        [P("+ Tutma Skoru", bold=True), "", f"+{sk.tutma_skoru}", ""],
        [P("= Skor B", bold=True), "", P(str(sk.skor_b), bold=True), ""],
    ]
    tb = Table(b_rows, colWidths=[4*cm, 3*cm, 2.5*cm, 8.5*cm])
    tsb = ts_base()
    tsb.add('TEXTCOLOR', (2,-1), (2,-1), rt(sk.skor_b))
    tsb.add('FONTSIZE', (2,-1), (2,-1), 11)
    tsb.add('BACKGROUND', (0,-1), (-1,-1), ACIK)
    tb.setStyle(tsb)
    story.append(tb)
    story.append(Spacer(1, 0.25*cm))

    story.append(P("Final Hesaplama", bold=True, size=9, color=KOYU, space_after=3))
    final_rows = [
        ["Adım", "Hesaplama", "Sonuç"],
        ["Tablo C", f"Skor A({sk.skor_a}) × Skor B({sk.skor_b})", str(sk.skor_c)],
        ["+ Aktivite", f"+{sk.aktivite_skoru}", ""],
        [P("NİHAİ REBA", bold=True), "", P(str(sk.final_skor), bold=True)],
    ]
    tfin = Table(final_rows, colWidths=[4*cm, 9*cm, 5*cm])
    stfin = ts_base()
    stfin.add('FONTSIZE', (2,-1), (2,-1), 14)
    stfin.add('TEXTCOLOR', (2,-1), (2,-1), rt(sk.final_skor))
    stfin.add('BACKGROUND', (0,-1), (-1,-1), rk(sk.final_skor))
    tfin.setStyle(stfin)
    story.append(tfin)
    story.append(Spacer(1, 0.3*cm))

    story.append(P("Genel Değerlendirme", bold=True, size=10, color=KOYU, space_after=3))
    oz = [
        ["Fotoğraf", "Ortalama REBA", "En Yüksek", "Risk Seviyesi", "Aksiyon"],
        [str(len(gecerli)), f"{ort:.1f}", str(max(skorlar)),
         en_yuk_obj.risk_seviyesi, en_yuk_obj.aksiyon],
    ]
    to = Table(oz, colWidths=[2.5*cm, 3.5*cm, 3*cm, 4*cm, 5*cm])
    sto = ts_base()
    sto.add('ALIGN', (0,0), (-1,-1), 'CENTER')
    sto.add('TEXTCOLOR', (2,1), (2,1), rt(max(skorlar)))
    sto.add('FONTNAME', (2,1), (2,1), FONT_BOLD)
    to.setStyle(sto)
    story.append(to)
    story.append(Spacer(1, 0.3*cm))

    story.append(P("REBA Risk Skalası", bold=True, size=9, color=KOYU, space_after=3))
    rsr = [
        ["Skor", "Risk", "Önlem"],
        ["1", "Önemsiz", "Önlem gerekmez"],
        ["2-3", "Düşük", "Gerekirse iyileştirme"],
        ["4-7", "Orta", "Ayrıntılı incele, değişiklik planla"],
        ["8-10", "Yüksek", "Araştırma yap, aksiyon al"],
        ["11+", "Çok Yüksek", "Derhal revize et"],
    ]
    trs = Table(rsr, colWidths=[2*cm, 3.5*cm, 12.5*cm])
    strs = ts_base()
    strs.add('FONTSIZE', (0,0), (-1,-1), 7.5)
    for i, c in enumerate([
        colors.HexColor('#dcfce7'), colors.HexColor('#ecfccb'),
        colors.HexColor('#fef9c3'), colors.HexColor('#fee2e2'),
        colors.HexColor('#ede9fe'),
    ], 1):
        strs.add('BACKGROUND', (0,i), (-1,i), c)
    trs.setStyle(strs)
    story.append(trs)

    # ── Foto sayfaları ──
    img_buffers = {}
    for fs in gecerli:
        if fs.overlay_img is not None:
            jpg = overlay_to_bytes(fs.overlay_img, quality=88)
            if jpg:
                img_buffers[fs.idx] = io.BytesIO(jpg)

    for idx, fs in enumerate(gecerli, 1):
        story.append(PageBreak())
        s = fs.skor
        a = s.acılar

        story.append(P(f"Fotoğraf {idx}  —  Detaylı REBA Analizi",
                       bold=True, size=14, color=KOYU, space_after=6))
        story.append(Spacer(1, 0.1*cm))
        story.append(P(
            f"REBA: {s.final_skor}/15  |  {s.risk_seviyesi}  |  {s.aksiyon}",
            size=8, color=colors.HexColor('#64748b'), space_after=8))
        story.append(HRFlowable(width="100%", thickness=1.5, color=KOYU))
        story.append(Spacer(1, 0.2*cm))

        if fs.idx in img_buffers:
            img_buf = img_buffers[fs.idx]
            img_buf.seek(0)
            img_w = 12*cm
            oh, ow = fs.overlay_img.shape[:2]
            img_h = img_w * (oh / ow)
            max_h = 14*cm
            if img_h > max_h:
                img_h = max_h
                img_w = img_h * (ow / oh)
            story.append(RLImage(img_buf, width=img_w, height=img_h))
            story.append(Spacer(1, 0.2*cm))

        if s.aciklama:
            story.append(P("Neden Bu Skor? — Segment Analizi",
                           bold=True, size=10, color=KOYU, space_after=3))
            exp_rows = [["Segment", "Açı", "Skor", "Açıklama"]]
            for item in s.aciklama:
                exp_rows.append([
                    str(item.get('segment', '')),
                    str(item.get('aci', '')),
                    str(item.get('temel', '')),
                    str(item.get('aciklama', '')),
                ])
            te = Table(exp_rows, colWidths=[4*cm, 3*cm, 2*cm, 9*cm])
            te.setStyle(ts_base())
            story.append(te)
            story.append(Spacer(1, 0.2*cm))

        story.append(P("Skor Hesaplama", bold=True, size=10, color=KOYU, space_after=3))
        hr = [
            ["Adım", "Hesaplama", "Sonuç"],
            ["Tablo A", f"Gövde({s.govde_skoru})×Boyun({s.boyun_skoru})×Bacak({s.bacak_skoru})", str(s.tablo_a)],
            ["Skor A", f"Tablo A({s.tablo_a}) + Yük({s.yuk_skoru})", str(s.skor_a)],
            ["Tablo B", f"ÜstKol({s.ust_kol_skoru})×AltKol({s.alt_kol_skoru})×Bilek({s.bilek_skoru})", str(s.tablo_b)],
            ["Skor B", f"Tablo B({s.tablo_b}) + Tutma({s.tutma_skoru})", str(s.skor_b)],
            ["Tablo C", f"Skor A({s.skor_a}) × Skor B({s.skor_b})", str(s.skor_c)],
            ["REBA", f"Tablo C({s.skor_c}) + Aktivite({s.aktivite_skoru})", str(s.final_skor)],
        ]
        th = Table(hr, colWidths=[3*cm, 9*cm, 6*cm])
        sth = ts_base()
        sth.add('FONTNAME', (2,-1), (2,-1), FONT_BOLD)
        sth.add('FONTSIZE', (2,-1), (2,-1), 12)
        sth.add('TEXTCOLOR', (2,-1), (2,-1), rt(s.final_skor))
        sth.add('BACKGROUND', (0,-1), (-1,-1), rk(s.final_skor))
        th.setStyle(sth)
        story.append(th)

        # #12 + #13: AI destekli öneriler
        if s.final_skor >= 4:
            story.append(Spacer(1, 0.2*cm))
            oneri_baslik = "Önerilen Aksiyonlar (AI Destekli)" if github_token else "Önerilen Aksiyonlar"
            story.append(P(oneri_baslik, bold=True, size=10, color=KOYU, space_after=3))
            aksiyonlar = _aksiyon_onerisi(s, github_token)
            for aks in aksiyonlar:
                story.append(P(f"• {aks}", size=8, space_after=2))

        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=CIZGI))
        story.append(P(
            "REBA Analiz Ajanı v5.3  |  Hignett & McAtamney (2000)  |  "
            "AI açı tahmini ±3-5° doğruluk payı içerir",
            size=7, color=colors.HexColor('#94a3b8')))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ════════════════════════════════════════════════════════
# VİDEO PDF RAPORU
# ════════════════════════════════════════════════════════

def video_pdf_olustur(
    form_bilgi: dict,
    video_sonuclari: list,
    en_riskli_frame: 'np.ndarray',
    en_riskli_skor: 'REBASkoru',
    sure_sn: float,
    github_token: str = None,        # #13: AI öneri motoru için
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable, PageBreak, Image as RLImage,
    )

    if not video_sonuclari:
        return b""

    FONT, FONT_BOLD = _turkce_font_yukle()

    tum_skorlar = [v['skor'] for v in video_sonuclari]
    ort = sum(tum_skorlar) / len(tum_skorlar)
    en_yuk = max(tum_skorlar)
    en_dus = min(tum_skorlar)
    sorted_s = sorted(tum_skorlar)
    p90 = sorted_s[min(int(len(sorted_s) * 0.9), len(sorted_s)-1)]

    dagilim = {'Önemsiz/Düşük': 0, 'Orta': 0, 'Yüksek': 0, 'Çok Yüksek': 0}
    for s in tum_skorlar:
        if s <= 3:   dagilim['Önemsiz/Düşük'] += 1
        elif s <= 7: dagilim['Orta'] += 1
        elif s <= 10: dagilim['Yüksek'] += 1
        else:        dagilim['Çok Yüksek'] += 1

    KOYU  = colors.HexColor('#1e3a5f')
    ACIK  = colors.HexColor('#f0f4f8')
    CIZGI = colors.HexColor('#cbd5e1')

    def rk(skor):
        if skor <= 1:    return colors.HexColor('#dcfce7')
        elif skor <= 3:  return colors.HexColor('#ecfccb')
        elif skor <= 7:  return colors.HexColor('#fef9c3')
        elif skor <= 10: return colors.HexColor('#fee2e2')
        else:            return colors.HexColor('#ede9fe')

    def rt(skor):
        if skor <= 1:    return colors.HexColor('#16a34a')
        elif skor <= 3:  return colors.HexColor('#65a30d')
        elif skor <= 7:  return colors.HexColor('#d97706')
        elif skor <= 10: return colors.HexColor('#dc2626')
        else:            return colors.HexColor('#7c3aed')

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=1.5*cm, leftMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    ss = getSampleStyleSheet()

    def P(txt, bold=False, size=9, color=colors.black, space_after=0):
        fn = FONT_BOLD if bold else FONT
        st = ParagraphStyle('x', parent=ss['Normal'],
                            fontName=fn, fontSize=size,
                            textColor=color, spaceAfter=space_after)
        return Paragraph(txt, st)

    def ts_base():
        return TableStyle([
            ('FONTNAME',  (0,0), (-1,-1), FONT),
            ('FONTNAME',  (0,0), (-1, 0), FONT_BOLD),
            ('FONTSIZE',  (0,0), (-1,-1), 8),
            ('BACKGROUND',(0,0), (-1, 0), KOYU),
            ('TEXTCOLOR', (0,0), (-1, 0), colors.white),
            ('GRID',      (0,0), (-1,-1), 0.4, CIZGI),
            ('PADDING',   (0,0), (-1,-1), 4),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, ACIK]),
            ('VALIGN',    (0,0), (-1,-1), 'MIDDLE'),
        ])

    story = []

    story.append(P("REBA Video Analiz Raporu",
                   bold=True, size=18, color=KOYU, space_after=6))
    story.append(Spacer(1, 0.1*cm))
    story.append(P(
        "Rapid Entire Body Assessment  |  Video Postür Analizi  |  v5.3",
        size=8, color=colors.HexColor('#64748b'), space_after=10))
    story.append(HRFlowable(width="100%", thickness=2, color=KOYU))
    story.append(Spacer(1, 0.3*cm))

    story.append(P("Form Bilgileri", bold=True, size=10, color=KOYU, space_after=4))
    fr = [
        [P("Bölüm", bold=True),       form_bilgi.get('bolum', '—'),
         P("Tarih", bold=True),        form_bilgi.get('tarih', '—')],
        [P("İş İstasyonu", bold=True), form_bilgi.get('is_istasyonu', '—'),
         P("Analist", bold=True),      form_bilgi.get('analist', '—')],
        [P("İş Adımı", bold=True),     form_bilgi.get('is_adimi', '—'),
         P("Video Süresi", bold=True), f"{sure_sn:.0f} sn · {len(tum_skorlar)} kare"],
    ]
    tf = Table(fr, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    tf.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), FONT),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,0), (-1,-1), ACIK),
        ('GRID', (0,0), (-1,-1), 0.4, CIZGI),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(tf)
    story.append(Spacer(1, 0.3*cm))

    story.append(P("Manuel Parametreler", bold=True, size=10, color=KOYU, space_after=3))
    mp_rows = [[
        "Yük", f"+{form_bilgi.get('yuk_skoru',0)} ({form_bilgi.get('yuk_kg',0)} kg)",
        "Tutma", f"+{form_bilgi.get('tutma',0)} ({form_bilgi.get('tutma_label','')})",
        "Aktivite", f"+{form_bilgi.get('aktivite',0)}",
    ]]
    tmp = Table(mp_rows, colWidths=[2*cm, 4*cm, 2*cm, 5*cm, 2.5*cm, 2.5*cm])
    tmp.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), FONT),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('BACKGROUND', (0,0), (-1,-1), ACIK),
        ('GRID', (0,0), (-1,-1), 0.3, CIZGI),
        ('PADDING', (0,0), (-1,-1), 3),
        ('FONTNAME', (0,0), (0,0), FONT_BOLD),
        ('FONTNAME', (2,0), (2,0), FONT_BOLD),
        ('FONTNAME', (4,0), (4,0), FONT_BOLD),
    ]))
    story.append(tmp)
    story.append(Spacer(1, 0.3*cm))

    story.append(P("Genel Metrikler", bold=True, size=10, color=KOYU, space_after=4))
    met_rows = [
        ["Ortalama REBA", "%90 Persentil", "En Yüksek", "En Düşük"],
        [f"{ort:.1f}", str(p90), str(en_yuk), str(en_dus)],
    ]
    tm = Table(met_rows, colWidths=[4.5*cm, 4.5*cm, 4.5*cm, 4.5*cm])
    stm = ts_base()
    stm.add('ALIGN', (0,0), (-1,-1), 'CENTER')
    stm.add('FONTSIZE', (0,1), (-1,1), 13)
    stm.add('FONTNAME', (0,1), (-1,1), FONT_BOLD)
    stm.add('TEXTCOLOR', (0,1), (0,1), rt(round(ort)))
    stm.add('TEXTCOLOR', (1,1), (1,1), rt(p90))
    stm.add('TEXTCOLOR', (2,1), (2,1), rt(en_yuk))
    stm.add('PADDING', (0,0), (-1,-1), 8)
    tm.setStyle(stm)
    story.append(tm)
    story.append(Spacer(1, 0.3*cm))

    story.append(P("Risk Dağılımı", bold=True, size=10, color=KOYU, space_after=4))
    dag_rows = [["Risk Seviyesi", "Kare Sayısı", "Yüzde"]]
    renk_map_pdf = {
        'Önemsiz/Düşük': colors.HexColor('#dcfce7'),
        'Orta': colors.HexColor('#fef9c3'),
        'Yüksek': colors.HexColor('#fee2e2'),
        'Çok Yüksek': colors.HexColor('#ede9fe'),
    }
    for seviye, sayi in dagilim.items():
        pct = sayi / len(tum_skorlar) * 100 if tum_skorlar else 0
        dag_rows.append([seviye, str(sayi), f"{pct:.0f}%"])
    td = Table(dag_rows, colWidths=[7*cm, 5*cm, 6*cm])
    std = ts_base()
    for i, seviye in enumerate(dagilim.keys(), 1):
        std.add('BACKGROUND', (0,i), (-1,i), renk_map_pdf[seviye])
    std.add('ALIGN', (1,0), (-1,-1), 'CENTER')
    td.setStyle(std)
    story.append(td)
    story.append(Spacer(1, 0.3*cm))

    if en_riskli_frame is not None:
        story.append(P(f"En Riskli Kare — REBA {en_yuk}",
                       bold=True, size=10, color=KOYU, space_after=4))
        en_riskli_zaman = next(
            (v['zaman'] for v in video_sonuclari if v['skor'] == en_yuk), 0)
        story.append(P(f"Zaman: {en_riskli_zaman:.1f}s · {en_riskli_skor.risk_seviyesi}",
                       size=8, color=colors.HexColor('#64748b'), space_after=4))
        jpg = overlay_to_bytes(en_riskli_frame, quality=88)
        if jpg:
            img_buf = io.BytesIO(jpg)
            img_w = 10*cm
            oh, ow = en_riskli_frame.shape[:2]
            img_h = img_w * (oh / ow)
            max_h = 10*cm
            if img_h > max_h:
                img_h = max_h
                img_w = img_h * (ow / oh)
            story.append(RLImage(img_buf, width=img_w, height=img_h))
            story.append(Spacer(1, 0.3*cm))

    story.append(P("Kare Bazlı REBA Skorları", bold=True, size=10, color=KOYU, space_after=4))
    frame_rows = [["Zaman (s)", "REBA Skoru", "Risk Seviyesi"]]
    for v in video_sonuclari:
        frame_rows.append([f"{v['zaman']:.1f}", str(v['skor']), v['risk']])
    tf2 = Table(frame_rows, colWidths=[4*cm, 4*cm, 10*cm])
    stf2 = ts_base()
    stf2.add('ALIGN', (0,0), (1,-1), 'CENTER')
    for i, v in enumerate(video_sonuclari, 1):
        stf2.add('TEXTCOLOR', (1,i), (1,i), rt(v['skor']))
        stf2.add('FONTNAME', (1,i), (1,i), FONT_BOLD)
        stf2.add('BACKGROUND', (0,i), (-1,i), rk(v['skor']))
    tf2.setStyle(stf2)
    story.append(tf2)

    # ── Sayfa 2: Worksheet + Segment + Aksiyon ──
    story.append(PageBreak())
    sk = en_riskli_skor
    ac = sk.acılar

    story.append(P("En Riskli Kare — Detaylı REBA Analizi",
                   bold=True, size=16, color=KOYU, space_after=6))
    story.append(Spacer(1, 0.1*cm))
    story.append(P(
        f"REBA: {sk.final_skor}/15  |  {sk.risk_seviyesi}  |  {sk.aksiyon}",
        size=8, color=colors.HexColor('#64748b'), space_after=8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=KOYU))
    story.append(Spacer(1, 0.25*cm))

    if ac:
        boyun_mod = []
        if ac.boyun_extension:     boyun_mod.append("Extension +1")
        if ac.boyun_yan_egim > 0:  boyun_mod.append("Yana eğme +1")
        if ac.boyun_donus:         boyun_mod.append("Dönüş +1")

        govde_mod = []
        if ac.govde_extension:     govde_mod.append("Extension +1")
        if ac.govde_yan_egim > 0:  govde_mod.append("Yana eğme +1")
        if ac.govde_donus:         govde_mod.append("Dönüş +1")

        ukol_mod = []
        if ac.omuz_kalkmis: ukol_mod.append("Omuz kalkış +1")
        if ac.kol_abdukte:  ukol_mod.append("Abdüksiyon +1")
        if ac.kol_destekli: ukol_mod.append("Destekli -1")

        diz_max = max(ac.diz_flexion_sol, ac.diz_flexion_sag)
        taraf = ac.analiz_tarafi

        story.append(P("A Grubu: Boyun — Gövde — Bacak",
                       bold=True, size=9, color=KOYU, space_after=3))
        a_rows = [
            ["Segment", "Açı", "Skor", "Modifikatör"],
            ["Boyun", f"{ac.boyun_flexion:.0f}°", str(sk.boyun_skoru), ", ".join(boyun_mod) or "—"],
            ["Gövde", f"{ac.govde_flexion:.0f}°", str(sk.govde_skoru), ", ".join(govde_mod) or "—"],
            ["Bacak", f"Diz {diz_max:.0f}°", str(sk.bacak_skoru),
             "Bilateral" if ac.bilateral_destek else "Tek ayak"],
            [P("Tablo A", bold=True), "", P(str(sk.tablo_a), bold=True), ""],
            [P("+ Yük", bold=True), "", f"+{sk.yuk_skoru}", ""],
            [P("= Skor A", bold=True), "", P(str(sk.skor_a), bold=True), ""],
        ]
        ta = Table(a_rows, colWidths=[4*cm, 3*cm, 2.5*cm, 8.5*cm])
        tsa = ts_base()
        tsa.add('TEXTCOLOR', (2,-1), (2,-1), rt(sk.skor_a))
        tsa.add('FONTSIZE', (2,-1), (2,-1), 11)
        tsa.add('BACKGROUND', (0,-1), (-1,-1), ACIK)
        ta.setStyle(tsa)
        story.append(ta)
        story.append(Spacer(1, 0.25*cm))

        story.append(P("B Grubu: Üst Kol — Alt Kol — Bilek",
                       bold=True, size=9, color=KOYU, space_after=3))
        b_rows = [
            ["Segment", "Açı", "Skor", "Modifikatör"],
            [f"Üst Kol ({taraf})", f"{ac.ust_kol_aci:.0f}°", str(sk.ust_kol_skoru), ", ".join(ukol_mod) or "—"],
            [f"Alt Kol ({taraf})", f"{ac.alt_kol_aci:.0f}°", str(sk.alt_kol_skoru),
             "60-100° arası" if sk.alt_kol_skoru==1 else "Aralık dışı"],
            [f"Bilek ({taraf})", f"{ac.bilek_aci:.0f}°", str(sk.bilek_skoru),
             "Dönüş +1" if ac.bilek_donus else "—"],
            [P("Tablo B", bold=True), "", P(str(sk.tablo_b), bold=True), ""],
            [P("+ Tutma", bold=True), "", f"+{sk.tutma_skoru}", ""],
            [P("= Skor B", bold=True), "", P(str(sk.skor_b), bold=True), ""],
        ]
        tb = Table(b_rows, colWidths=[4*cm, 3*cm, 2.5*cm, 8.5*cm])
        tsb = ts_base()
        tsb.add('TEXTCOLOR', (2,-1), (2,-1), rt(sk.skor_b))
        tsb.add('FONTSIZE', (2,-1), (2,-1), 11)
        tsb.add('BACKGROUND', (0,-1), (-1,-1), ACIK)
        tb.setStyle(tsb)
        story.append(tb)
        story.append(Spacer(1, 0.25*cm))

    story.append(P("Final Hesaplama", bold=True, size=9, color=KOYU, space_after=3))
    final_rows = [
        ["Adım", "Hesaplama", "Sonuç"],
        ["Tablo C", f"Skor A({sk.skor_a}) × Skor B({sk.skor_b})", str(sk.skor_c)],
        ["+ Aktivite", f"+{sk.aktivite_skoru}", ""],
        [P("NİHAİ REBA", bold=True), "", P(str(sk.final_skor), bold=True)],
    ]
    tfin = Table(final_rows, colWidths=[4*cm, 9*cm, 5*cm])
    stfin = ts_base()
    stfin.add('FONTSIZE', (2,-1), (2,-1), 14)
    stfin.add('TEXTCOLOR', (2,-1), (2,-1), rt(sk.final_skor))
    stfin.add('BACKGROUND', (0,-1), (-1,-1), rk(sk.final_skor))
    tfin.setStyle(stfin)
    story.append(tfin)
    story.append(Spacer(1, 0.25*cm))

    if sk.aciklama:
        story.append(P("Neden Bu Skor? — Segment Analizi",
                       bold=True, size=10, color=KOYU, space_after=3))
        exp_rows = [["Segment", "Açı", "Skor", "Açıklama"]]
        for item in sk.aciklama:
            exp_rows.append([
                str(item.get('segment', '')),
                str(item.get('aci', '')),
                str(item.get('temel', '')),
                str(item.get('aciklama', '')),
            ])
        te = Table(exp_rows, colWidths=[4*cm, 3*cm, 2*cm, 9*cm])
        te.setStyle(ts_base())
        story.append(te)
        story.append(Spacer(1, 0.25*cm))

    # #12 + #13: AI destekli öneriler
    if sk.final_skor >= 4:
        oneri_baslik = "Önerilen Aksiyonlar (AI Destekli)" if github_token else "Önerilen Aksiyonlar"
        story.append(P(oneri_baslik, bold=True, size=10, color=KOYU, space_after=3))
        for aks in _aksiyon_onerisi(sk, github_token):
            story.append(P(f"• {aks}", size=8, space_after=3))

    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=CIZGI))
    story.append(P(
        "REBA Analiz Ajanı v5.3  |  Hignett & McAtamney (2000)  |  "
        "AI tabanlı açı tahmini ±3-5° doğruluk payı içerir",
        size=7, color=colors.HexColor('#94a3b8')))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
