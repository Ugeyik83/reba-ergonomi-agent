"""
reba_visual.py — Görsel Çıktı Modülü
İskelet overlay çizimi ve PDF rapor oluşturma.
reba_core'a bağımlıdır; Streamlit'e bağımlı değildir.
"""

import cv2
import numpy as np
import io
import os
import glob
from datetime import datetime
from typing import List

import mediapipe as mp

from reba_core import AcilarObj, REBASkoru, FotoSonuc, risk_info

LM = mp.solutions.pose.PoseLandmark
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


# ════════════════════════════════════════════════════════
# İSKELET + AÇI OVERLAY
# ════════════════════════════════════════════════════════

def _skor_renk_bgr(final_skor: int) -> tuple:
    """Risk skoruna göre BGR renk döndür."""
    if final_skor <= 3:   return (22, 163, 74)    # yeşil
    elif final_skor <= 7: return (217, 119, 6)    # turuncu
    elif final_skor <= 10: return (220, 38, 38)   # kırmızı
    else:                  return (124, 58, 237)   # mor


def _etiket_yaz(img: np.ndarray, nokta: tuple, metin: str, renk: tuple):
    """Görüntüye arka planlı açı etiketi yaz."""
    x, y = nokta
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thick = 0.4, 1
    (tw, th), _ = cv2.getTextSize(metin, font, fs, thick)
    # Görüntü sınırı kontrolü
    x = max(2, min(x, w - tw - 8))
    y = max(th + 4, min(y, h - 4))
    # Arka plan kutu
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + 4),
                  (255, 255, 240), -1)
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + 4),
                  renk, 1)
    cv2.putText(img, metin, (x, y), font, fs, renk, thick, cv2.LINE_AA)


def overlay_ciz(img: np.ndarray, landmarks, skor: REBASkoru) -> np.ndarray:
    """
    Fotoğraf üzerine iskelet, eklem noktaları ve açı etiketleri çiz.
    Döndürülen görüntü BGR formatındadır.
    """
    h, w = img.shape[:2]
    out = img.copy()
    col = _skor_renk_bgr(skor.final_skor)

    def p(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    def v(idx):
        return landmarks[idx].visibility

    # İskelet bağlantıları
    for conn in POSE_CONNECTIONS:
        s_lm, e_lm = landmarks[conn[0]], landmarks[conn[1]]
        if s_lm.visibility > 0.4 and e_lm.visibility > 0.4:
            cv2.line(out, p(conn[0]), p(conn[1]), col, 2, cv2.LINE_AA)

    # Eklem noktaları
    for lm in landmarks:
        if lm.visibility > 0.4:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(out, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, pt, 5, col, 1, cv2.LINE_AA)

    a = skor.acılar
    if a is None:
        return out

    # Orta noktalar
    mid_omuz_x = int((landmarks[LM.LEFT_SHOULDER].x + landmarks[LM.RIGHT_SHOULDER].x) / 2 * w)
    mid_omuz_y = int((landmarks[LM.LEFT_SHOULDER].y + landmarks[LM.RIGHT_SHOULDER].y) / 2 * h)
    mid_kalca_y = int((landmarks[LM.LEFT_HIP].y + landmarks[LM.RIGHT_HIP].y) / 2 * h)

    # ── BOYUN ETİKETİ ──
    if v(LM.NOSE) > 0.4:
        nx, ny = p(LM.NOSE)
        mod = ("YE" if a.boyun_yan_egim > 15 else "") + ("D" if a.boyun_donus else "")
        mod_str = f"+{mod}" if mod else ""
        _etiket_yaz(out, (nx + 8, ny - 6),
                    f"Boyun:{a.boyun_flexion:.0f}{mod_str} [{skor.boyun_skoru}]",
                    renk=(15, 80, 160))

    # ── GÖVDE ETİKETİ ──
    gy = (mid_omuz_y + mid_kalca_y) // 2
    mod = ("YE" if a.govde_yan_egim > 10 else "") + ("D" if a.govde_donus else "")
    mod_str = f"+{mod}" if mod else ""
    _etiket_yaz(out, (mid_omuz_x + 10, gy),
                f"Govde:{a.govde_flexion:.0f}{mod_str} [{skor.govde_skoru}]",
                renk=(140, 60, 0))

    # ── BACAK ETİKETİ ──
    if v(LM.LEFT_KNEE) > 0.4:
        kx, ky = p(LM.LEFT_KNEE)
        diz_max = max(a.diz_flexion_sol, a.diz_flexion_sag)
        _etiket_yaz(out, (kx + 6, ky),
                    f"Diz:{diz_max:.0f} [{skor.bacak_skoru}]",
                    renk=(70, 100, 20))

    # ── ÜST KOL ETİKETİ ──
    if v(LM.RIGHT_ELBOW) > 0.4:
        ex, ey = p(LM.RIGHT_ELBOW)
        mod = ("OK" if a.omuz_kalkmis else "") + ("AB" if a.kol_abdukte else "")
        mod_str = f"+{mod}" if mod else ""
        _etiket_yaz(out, (ex + 6, ey - 8),
                    f"UKol:{a.ust_kol_aci:.0f}{mod_str} [{skor.ust_kol_skoru}]",
                    renk=(120, 20, 120))

    # ── ALT KOL + BİLEK ETİKETİ ──
    if v(LM.RIGHT_WRIST) > 0.4:
        wx, wy = p(LM.RIGHT_WRIST)
        _etiket_yaz(out, (wx + 6, wy - 16),
                    f"AKol:{a.alt_kol_aci:.0f} [{skor.alt_kol_skoru}]",
                    renk=(0, 100, 140))
        mod_str = "+D" if a.bilek_donus else ""
        _etiket_yaz(out, (wx + 6, wy + 4),
                    f"Bilek:{a.bilek_aci:.0f}{mod_str} [{skor.bilek_skoru}]",
                    renk=(0, 80, 160))

    # ── REBA SKOR KUTUSU (sol üst) ──
    s = skor.final_skor
    cv2.rectangle(out, (8, 8), (195, 70), (255, 255, 255), -1)
    cv2.rectangle(out, (8, 8), (195, 70), col, 2)
    cv2.rectangle(out, (8, 8), (195, 30), col, -1)
    cv2.putText(out, "REBA SKORU", (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, str(s), (14, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3, cv2.LINE_AA)
    cv2.putText(out, "/15", (56, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    return out


def overlay_to_bytes(overlay_img: np.ndarray, quality: int = 88) -> bytes:
    """OpenCV BGR görüntüsünü JPEG bytes olarak döndür."""
    ok, buf = cv2.imencode('.jpg', overlay_img,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b""
    return buf.tobytes()


# ════════════════════════════════════════════════════════
# FONT YÖNETİMİ (Türkçe karakter desteği)
# ════════════════════════════════════════════════════════

def _turkce_font_yukle():
    """
    Sistemde DejaVu veya Liberation fontunu bul ve reportlab'a kaydet.
    Başarısız olursa Helvetica'ya düşer.
    Tuple döndürür: (FONT_ADI, FONT_BOLD_ADI)
    """
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
# PDF RAPORU
# ════════════════════════════════════════════════════════

def pdf_olustur(form_bilgi: dict, foto_sonuclari: List[FotoSonuc]) -> bytes:
    """
    Tüm foto sonuçları için A4 PDF raporu oluştur.
    Sayfa 1: Özet (form bilgisi, parametreler, genel tablo, risk skalası)
    Sayfa 2+: Her fotoğraf için ayrı sayfa (overlay görsel + segment tablosu + hesaplama)

    Görüntüler BytesIO ile aktarılır — temp dosya kullanılmaz.
    """
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

    # Renk paleti
    KOYU  = colors.HexColor('#1e3a5f')
    ACIK  = colors.HexColor('#f0f4f8')
    CIZGI = colors.HexColor('#cbd5e1')

    def rk(skor: int):
        """Risk arka plan rengi."""
        if skor <= 1:   return colors.HexColor('#dcfce7')
        elif skor <= 3: return colors.HexColor('#ecfccb')
        elif skor <= 7: return colors.HexColor('#fef9c3')
        elif skor <= 10: return colors.HexColor('#fee2e2')
        else:           return colors.HexColor('#ede9fe')

    def rt(skor: int):
        """Risk metin rengi."""
        if skor <= 1:   return colors.HexColor('#16a34a')
        elif skor <= 3: return colors.HexColor('#65a30d')
        elif skor <= 7: return colors.HexColor('#d97706')
        elif skor <= 10: return colors.HexColor('#dc2626')
        else:           return colors.HexColor('#7c3aed')

    # PDF buffer
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    ss = getSampleStyleSheet()

    def P(txt: str, bold=False, size=9, color=colors.black, space_after=0):
        fn = FONT_BOLD if bold else FONT
        st = ParagraphStyle('x', parent=ss['Normal'],
                            fontName=fn, fontSize=size,
                            textColor=color, spaceAfter=space_after)
        return Paragraph(txt, st)

    def ts_base() -> TableStyle:
        return TableStyle([
            ('FONTNAME',  (0,0), (-1,-1), FONT),
            ('FONTNAME',  (0,0), (-1, 0), FONT_BOLD),
            ('FONTSIZE',  (0,0), (-1,-1), 8.5),
            ('BACKGROUND',(0,0), (-1, 0), KOYU),
            ('TEXTCOLOR', (0,0), (-1, 0), colors.white),
            ('GRID',      (0,0), (-1,-1), 0.4, CIZGI),
            ('PADDING',   (0,0), (-1,-1), 4),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, ACIK]),
            ('VALIGN',    (0,0), (-1,-1), 'MIDDLE'),
        ])

    story = []

    # ══════════════════════════════════════════════════
    # SAYFA 1 — ÖZET
    # ══════════════════════════════════════════════════

    story.append(P("REBA Ergonomi Risk Analiz Raporu",
                   bold=True, size=18, color=KOYU, space_after=6))
    story.append(Spacer(1, 0.15*cm))
    story.append(P(
        "Rapid Entire Body Assessment  |  AI Destekli Postur Analizi  |  v5.1",
        size=8, color=colors.HexColor('#64748b'), space_after=10))
    story.append(HRFlowable(width="100%", thickness=2, color=KOYU))
    story.append(Spacer(1, 0.4*cm))

    # Form bilgileri
    story.append(P("Form Bilgileri", bold=True, size=11, color=KOYU, space_after=4))
    fr = [
        [P("Bölüm", bold=True),          form_bilgi.get('bolum', '—'),
         P("Tarih", bold=True),           form_bilgi.get('tarih', '—')],
        [P("İş İstasyonu", bold=True),    form_bilgi.get('is_istasyonu', '—'),
         P("Analist", bold=True),         form_bilgi.get('analist', '—')],
        [P("İş Adımı", bold=True),        form_bilgi.get('is_adimi', '—'),
         P("Oluşturma", bold=True),       datetime.now().strftime('%d.%m.%Y %H:%M')],
    ]
    tf = Table(fr, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm])
    tf.setStyle(TableStyle([
        ('FONTNAME',   (0,0), (-1,-1), FONT),
        ('FONTSIZE',   (0,0), (-1,-1), 8.5),
        ('BACKGROUND', (0,0), (-1,-1), ACIK),
        ('GRID',       (0,0), (-1,-1), 0.4, CIZGI),
        ('PADDING',    (0,0), (-1,-1), 5),
    ]))
    story.append(tf)
    story.append(Spacer(1, 0.4*cm))

    # Manuel parametreler
    story.append(P("Manuel Girdi Parametreleri", bold=True, size=11, color=KOYU, space_after=4))
    mr = [
        ["Parametre", "Değer", "Açıklama"],
        [f"Yük: {form_bilgi.get('yuk_kg', 0)} kg",
         f"+{form_bilgi.get('yuk_skoru', 0)}",
         form_bilgi.get('yuk_aciklama', '')],
        ["Ani/Hızlı Kuvvet",
         "Evet" if form_bilgi.get('shock') else "Hayır",
         "+1 eklendi" if form_bilgi.get('shock') else "—"],
        ["Tutma Kalitesi",
         f"+{form_bilgi.get('tutma', 0)}",
         form_bilgi.get('tutma_label', '')],
        ["Aktivite Skoru",
         f"+{form_bilgi.get('aktivite', 0)}",
         form_bilgi.get('aktivite_aciklama', '')],
    ]
    tm = Table(mr, colWidths=[4.5*cm, 2.5*cm, 11*cm])
    tm.setStyle(ts_base())
    story.append(tm)
    story.append(Spacer(1, 0.4*cm))

    # Genel özet
    story.append(P("Genel Değerlendirme Özeti", bold=True, size=11, color=KOYU, space_after=4))
    oz = [
        ["Fotoğraf Sayısı", "Ortalama REBA", "En Yüksek REBA", "Risk Seviyesi"],
        [str(len(gecerli)), f"{ort:.1f}", str(max(skorlar)), en_yuk_obj.risk_seviyesi],
    ]
    to = Table(oz, colWidths=[4*cm, 4*cm, 4*cm, 6*cm])
    sto = ts_base()
    sto.add('FONTSIZE',  (0,1), (-1,1), 12)
    sto.add('FONTNAME',  (0,1), (-1,1), FONT_BOLD)
    sto.add('ALIGN',     (0,0), (-1,-1), 'CENTER')
    sto.add('TEXTCOLOR', (1,1), (1,1), rt(round(ort)))
    sto.add('TEXTCOLOR', (2,1), (2,1), rt(max(skorlar)))
    sto.add('PADDING',   (0,0), (-1,-1), 8)
    to.setStyle(sto)
    story.append(to)
    story.append(Spacer(1, 0.4*cm))

    # Foto özet tablosu
    story.append(P("Fotoğraf Bazlı REBA Skorları", bold=True, size=11, color=KOYU, space_after=4))
    fh = ["No","REBA","Risk","Boyun","Gövde","Bacak","ÜstKol","AltKol","Bilek","SkorA","SkorB"]
    frows = [fh]
    for i, fs in enumerate(gecerli, 1):
        s = fs.skor
        frows.append([
            str(i), str(s.final_skor), s.risk_seviyesi,
            str(s.boyun_skoru), str(s.govde_skoru), str(s.bacak_skoru),
            str(s.ust_kol_skoru), str(s.alt_kol_skoru), str(s.bilek_skoru),
            str(s.skor_a), str(s.skor_b),
        ])
    cw = [1.2*cm,1.5*cm,3.8*cm,1.4*cm,1.4*cm,1.4*cm,
          1.8*cm,1.8*cm,1.4*cm,1.5*cm,1.5*cm]
    tf4 = Table(frows, colWidths=cw)
    st4 = ts_base()
    st4.add('ALIGN', (0,0), (-1,-1), 'CENTER')
    for i, fs in enumerate(gecerli, 1):
        st4.add('TEXTCOLOR', (1,i), (1,i), rt(fs.skor.final_skor))
        st4.add('FONTNAME',  (1,i), (1,i), FONT_BOLD)
        st4.add('BACKGROUND', (0,i), (-1,i), rk(fs.skor.final_skor))
    tf4.setStyle(st4)
    story.append(tf4)
    story.append(Spacer(1, 0.4*cm))

    # Risk skalası
    story.append(P("REBA Risk Skalası", bold=True, size=11, color=KOYU, space_after=4))
    rsr = [
        ["Skor", "Risk Seviyesi", "Önlem"],
        ["1",    "Önemsiz Risk",       "Herhangi bir önlem gerekmez"],
        ["2-3",  "Düşük Risk",         "Gerekirse iyileştirme yapılabilir"],
        ["4-7",  "Orta Seviyeli Risk", "Daha ayrıntılı incele, değişiklik planla"],
        ["8-10", "Yüksek Risk",        "Araştırma yap ve aksiyon al"],
        ["11+",  "Çok Yüksek Risk",    "Süreç çalışmaya uygun değil, derhal revize et"],
    ]
    trs = Table(rsr, colWidths=[2*cm, 4.5*cm, 11.5*cm])
    strs = ts_base()
    for i, c in enumerate([
        colors.HexColor('#dcfce7'), colors.HexColor('#ecfccb'),
        colors.HexColor('#fef9c3'), colors.HexColor('#fee2e2'),
        colors.HexColor('#ede9fe'),
    ], 1):
        strs.add('BACKGROUND', (0,i), (-1,i), c)
    trs.setStyle(strs)
    story.append(trs)

    # ══════════════════════════════════════════════════
    # SAYFA 2+ — HER FOTO İÇİN
    # ══════════════════════════════════════════════════

    # Tüm overlay'leri önce BytesIO'ya çevir — doc.build() öncesi hazır olsun
    img_buffers = {}
    for fs in gecerli:
        if fs.overlay_img is not None:
            img_buf = io.BytesIO()
            jpg_bytes = overlay_to_bytes(fs.overlay_img, quality=88)
            if jpg_bytes:
                img_buf.write(jpg_bytes)
                img_buf.seek(0)
                img_buffers[fs.idx] = img_buf

    for idx, fs in enumerate(gecerli, 1):
        story.append(PageBreak())
        s = fs.skor
        a = s.acılar

        story.append(P(f"Fotoğraf {idx}  —  Detaylı REBA Analizi",
                       bold=True, size=16, color=KOYU, space_after=6))
        story.append(Spacer(1, 0.1*cm))
        story.append(P(
            f"REBA Skoru: {s.final_skor}/15  |  {s.risk_seviyesi}  |  {s.aksiyon}",
            size=8, color=colors.HexColor('#64748b'), space_after=10))
        story.append(HRFlowable(width="100%", thickness=1.5, color=KOYU))
        story.append(Spacer(1, 0.3*cm))

        # Overlay görsel — BytesIO'dan, temp dosya yok
        if fs.idx in img_buffers:
            img_buf = img_buffers[fs.idx]
            img_buf.seek(0)
            img_w = 12*cm
            rl_img = RLImage(img_buf, width=img_w)
            oh, ow = fs.overlay_img.shape[:2]
            rl_img.height = img_w * (oh / ow)
            story.append(rl_img)
            story.append(Spacer(1, 0.3*cm))

        # Segment tablosu
        story.append(P("Segment Analizi", bold=True, size=11, color=KOYU, space_after=4))

        if a:
            def _temel_boyun(ac):
                return 1 if ac <= 20 else 2 if ac <= 40 else 3

            def _temel_govde(ac):
                return 1 if ac <= 5 else 2 if ac <= 20 else 3 if ac <= 60 else 4

            def _mod_boyun():
                m = []
                if a.boyun_yan_egim > 15: m.append("Yana eğme +1")
                if a.boyun_donus: m.append("Dönüş +1")
                return ", ".join(m) or "—"

            def _mod_govde():
                m = []
                if a.govde_yan_egim > 10: m.append("Yana eğme +1")
                if a.govde_donus: m.append("Dönüş +1")
                return ", ".join(m) or "—"

            def _mod_ustkol():
                m = []
                if a.omuz_kalkmis: m.append("Omuz kalkış +1")
                if a.kol_abdukte: m.append("Abdüksiyon +1")
                if a.kol_destekli: m.append("Destekli -1")
                return ", ".join(m) or "—"

            sr = [["Segment", "Açı (derece)", "Temel", "Modifikatör", "Final"]]
            sr.append(["Boyun",    f"{a.boyun_flexion:.1f}",
                       str(_temel_boyun(a.boyun_flexion)), _mod_boyun(), str(s.boyun_skoru)])
            sr.append(["Gövde",    f"{a.govde_flexion:.1f}",
                       str(_temel_govde(a.govde_flexion)), _mod_govde(), str(s.govde_skoru)])
            sr.append(["Bacak/Diz",
                       f"{max(a.diz_flexion_sol, a.diz_flexion_sag):.1f}",
                       "1" if a.bilateral_destek else "2", "—", str(s.bacak_skoru)])
            sr.append(["Üst Kol",  f"{a.ust_kol_aci:.1f}",
                       str(1 if a.ust_kol_aci<=20 else 2 if a.ust_kol_aci<=45 else 3 if a.ust_kol_aci<=90 else 4),
                       _mod_ustkol(), str(s.ust_kol_skoru)])
            sr.append(["Alt Kol",  f"{a.alt_kol_aci:.1f}",
                       "1" if 60 <= a.alt_kol_aci <= 100 else "2", "—", str(s.alt_kol_skoru)])
            sr.append(["Bilek",    f"{a.bilek_aci:.1f}",
                       "1" if a.bilek_aci <= 15 else "2",
                       "Dönüş +1" if a.bilek_donus else "—", str(s.bilek_skoru)])

            tseg = Table(sr, colWidths=[3.5*cm, 3*cm, 2.5*cm, 4.5*cm, 2.5*cm])
            tseg.setStyle(ts_base())
            story.append(tseg)
            story.append(Spacer(1, 0.3*cm))

        # Skor hesaplama tablosu
        story.append(P("Skor Hesaplama", bold=True, size=11, color=KOYU, space_after=4))
        hr = [
            ["Adım", "Hesaplama", "Sonuç"],
            ["Tablo A",
             f"Gövde({s.govde_skoru}) x Boyun({s.boyun_skoru}) x Bacak({s.bacak_skoru})",
             str(s.tablo_a)],
            ["Skor A",
             f"Tablo A({s.tablo_a}) + Yük Skoru({s.yuk_skoru})",
             str(s.skor_a)],
            ["Tablo B",
             f"ÜstKol({s.ust_kol_skoru}) x AltKol({s.alt_kol_skoru}) x Bilek({s.bilek_skoru})",
             str(s.tablo_b)],
            ["Skor B",
             f"Tablo B({s.tablo_b}) + Tutma Skoru({s.tutma_skoru})",
             str(s.skor_b)],
            ["Tablo C",
             f"Skor A({s.skor_a}) x Skor B({s.skor_b})",
             str(s.skor_c)],
            ["REBA Skoru",
             f"Tablo C({s.skor_c}) + Aktivite({s.aktivite_skoru})",
             str(s.final_skor)],
        ]
        th = Table(hr, colWidths=[3.5*cm, 9*cm, 5.5*cm])
        sth = ts_base()
        sth.add('FONTNAME',   (2,-1), (2,-1), FONT_BOLD)
        sth.add('FONTSIZE',   (2,-1), (2,-1), 12)
        sth.add('TEXTCOLOR',  (2,-1), (2,-1), rt(s.final_skor))
        sth.add('BACKGROUND', (0,-1), (-1,-1), rk(s.final_skor))
        th.setStyle(sth)
        story.append(th)

        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor('#cbd5e1')))
        story.append(P(
            "REBA Analiz Ajani v5.1  |  Hignett & McAtamney (2000), Applied Ergonomics 31(2), 201-205  |  "
            "AI tabanli aci tahmini +/-3-5 derece dogruluk payi icerir",
            size=7, color=colors.HexColor('#94a3b8')))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# Dışarıya açık import
