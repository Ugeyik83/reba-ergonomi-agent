"""
reba_agent.py — Streamlit Arayüzü v5.2
REBA Ergonomi Risk Analiz Sistemi
#5: Düşük güven uyarısı
#6: Adaptive annotation mode
#9: Explainable AI gösterimi
"""

import streamlit as st
import cv2
import numpy as np
from datetime import date
from typing import List
import mediapipe as mp

from reba_core import (
    AcilarObj, REBASkoru, FotoSonuc,
    vucut_acilari_hesapla, reba_skorla, risk_info,
    segment_risk_renk, SEGMENT_MAX,
)
from reba_visual import overlay_ciz, pdf_olustur, ANNOTATION_MODES

# ════════════════════════════════════════════════════════
# SAYFA YAPILANDIRMASI
# ════════════════════════════════════════════════════════

st.set_page_config(
    page_title="REBA Ergonomi Analizi",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, .stApp {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 10px !important; font-weight: 700 !important;
    color: #64748b !important; text-transform: uppercase !important;
    letter-spacing: 0.1em !important; margin: 16px 0 6px !important;
    padding-bottom: 5px !important; border-bottom: 1px solid #e2e8f0 !important;
}
.main .block-container {
    padding: 0.8rem 1rem 2rem !important; max-width: 1200px !important;
}
.app-header {
    background: #fff; border: 1px solid #e2e8f0; border-left: 4px solid #1e3a5f;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;
    display: flex; align-items: center; gap: 12px;
}
.app-header-icon {
    width: 40px; height: 40px; background: #1e3a5f; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}
.app-header h1 { margin: 0; font-size: 16px; font-weight: 700; color: #0f172a; }
.app-header p { margin: 2px 0 0; font-size: 11px; color: #64748b; }
.metric-row {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;
}
@media (max-width: 640px) {
    .metric-row { grid-template-columns: repeat(2, 1fr); }
    .app-header h1 { font-size: 14px; }
    .app-header p { display: none; }
}
.metric-box {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px; text-align: center;
}
.metric-box .val {
    font-size: 28px; font-weight: 700; line-height: 1;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-box .lbl { font-size: 10px; color: #64748b; margin-top: 4px; font-weight: 500; }
.risk-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600; margin-top: 5px;
}
.foto-header {
    padding: 10px 14px; border-radius: 8px 8px 0 0;
    display: flex; justify-content: space-between; align-items: center;
    font-size: 13px; font-weight: 600; color: white;
}
.seg-bar-wrap { margin-bottom: 9px; }
.seg-bar-label {
    display: flex; justify-content: space-between;
    font-size: 12px; color: #374151; margin-bottom: 2px;
}
.seg-bar-track { background: #f1f5f9; border-radius: 3px; height: 7px; overflow: hidden; }
.seg-bar-fill { height: 100%; border-radius: 3px; }
.skor-ozet {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 10px 12px; font-size: 12px; margin-top: 8px;
}
.info-box {
    background: #eff6ff; border: 1px solid #bfdbfe; border-left: 3px solid #2563eb;
    border-radius: 6px; padding: 9px 13px; font-size: 12px; color: #1e40af; margin: 6px 0;
}
.warn-box {
    background: #fffbeb; border: 1px solid #fde68a; border-left: 3px solid #d97706;
    border-radius: 6px; padding: 9px 13px; font-size: 12px; color: #92400e; margin: 6px 0;
}
.error-box {
    background: #fef2f2; border: 1px solid #fecaca; border-left: 3px solid #dc2626;
    border-radius: 6px; padding: 9px 13px; font-size: 12px; color: #991b1b; margin: 6px 0;
}
.explain-table {
    width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 8px;
}
.explain-table th {
    background: #f8fafc; color: #475569; font-weight: 600; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.05em;
    padding: 5px 8px; border-bottom: 2px solid #e2e8f0; text-align: left;
}
.explain-table td {
    padding: 5px 8px; border-bottom: 1px solid #f1f5f9; color: #1e293b;
}
.stButton > button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important; border-radius: 6px !important;
    min-height: 44px !important; font-size: 14px !important;
}
.app-footer {
    text-align: center; font-size: 10px; color: #94a3b8;
    padding: 16px 0 8px; border-top: 1px solid #e2e8f0; margin-top: 24px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# BAŞLIK
# ════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
    <div class="app-header-icon">🦺</div>
    <div>
        <h1>REBA Ergonomi Risk Analiz Sistemi</h1>
        <p>Rapid Entire Body Assessment &nbsp;·&nbsp; Çoklu Fotoğraf &nbsp;·&nbsp;
           AI Postür Analizi &nbsp;·&nbsp; v5.2</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📋 Form Bilgileri")
    bolum        = st.text_input("Bölüm", placeholder="örn: Formasyon")
    is_istasyonu = st.text_input("İş İstasyonu", placeholder="örn: Final 1 Sonu")
    is_adimi     = st.text_input("İş Adımı / Kodu", placeholder="örn: Paletten akü alma")
    analist      = st.text_input("Analist", placeholder="Ad Soyad")
    tarih        = st.date_input("Tarih", value=date.today())

    st.markdown("### ⚖️ 4. Yük Analizi")
    yuk_kg = st.number_input("Yük (kg)", min_value=0.0, max_value=200.0,
                              value=0.0, step=0.5, format="%.1f")
    shock = st.checkbox("Ani / hızlı kuvvet (+1)")

    if yuk_kg < 5:
        yuk_base, yuk_aci = 0, f"{yuk_kg:.1f} kg → +0 (5 kg altı)"
    elif yuk_kg <= 10:
        yuk_base, yuk_aci = 1, f"{yuk_kg:.1f} kg → +1 (5–10 kg)"
    else:
        yuk_base, yuk_aci = 2, f"{yuk_kg:.1f} kg → +2 (10 kg üstü)"
    yuk_skoru_val = yuk_base + (1 if shock else 0)

    st.markdown(f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
                padding:9px 12px;font-size:12px;color:#374151;margin-top:4px">
        Yük Skoru: <strong>+{yuk_skoru_val}</strong><br>
        <span style="color:#64748b">{yuk_aci}{' · Ani kuvvet +1' if shock else ''}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🤜 8. Materyali Tutma")
    tutma_sec = {
        "Uygun tutma yeri mevcut (+0)":              (0, "İdeal kavrama"),
        "Kabul edilebilir ama ideal değil (+1)":     (1, "Yeterli kavrama"),
        "Tutulabilir ama uygun değil (+2)":          (2, "Zor kavrama"),
        "Tutma yeri yok / kaldırmaya uygun değil (+3)": (3, "Kavrama yok"),
    }
    tutma_secim = st.selectbox("Tutma", list(tutma_sec.keys()),
                                label_visibility="collapsed")
    tutma_val, tutma_label = tutma_sec[tutma_secim]

    st.markdown("### 🏃 9. Aktivite Analizi")
    akt1 = st.checkbox("Uzuv 1 dk+ statik pozisyonda (+1)")
    akt2 = st.checkbox("Dakikada 4+ tekrarlı hareket (+1)")
    akt3 = st.checkbox("Stabil olmayan zemin / hızlı değişim (+1)")
    aktivite_val = int(akt1) + int(akt2) + int(akt3)
    akt_aci = [l for c, l in [(akt1,"Statik"),(akt2,"Tekrarlı"),(akt3,"Dengesiz")] if c]

    if aktivite_val > 0:
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
                    padding:9px 12px;font-size:12px;margin-top:4px">
            Aktivite: <strong>+{aktivite_val}</strong>
            <span style="color:#64748b"> ({', '.join(akt_aci)})</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔄 Postür Modifier'ları")
    st.markdown("""
    <div style="font-size:10px;color:#64748b;margin-bottom:8px">
    AI açıları hesaplar, modifier'ları siz belirleyin.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Boyun**")
    boyun_yan_egim  = st.checkbox("Yana eğilme (+1)", key="b_ye")
    boyun_donus     = st.checkbox("Dönüş (+1)", key="b_d")
    boyun_extension = st.checkbox("Geriye eğilme / Extension (+1)", key="b_ext")

    st.markdown("**Gövde**")
    govde_yan_egim  = st.checkbox("Yana eğilme (+1)", key="g_ye")
    govde_donus     = st.checkbox("Dönüş (+1)", key="g_d")
    govde_extension = st.checkbox("Geriye eğilme / Extension (+1)", key="g_ext")

    st.markdown("**Üst Kol**")
    omuz_kalkmis = st.checkbox("Omuz kalkış (+1)", key="uk_ok")
    kol_abdukte  = st.checkbox("Abdüksiyon — dışa açılma (+1)", key="uk_ab")
    kol_destekli = st.checkbox("Kol destekli / yaslı (-1)", key="uk_des")

    st.markdown("**Bilek**")
    bilek_donus = st.checkbox("Dönüş / yana bükülme (+1)", key="bi_d")

    # Modifier özeti
    toplam_mod = (int(boyun_yan_egim) + int(boyun_donus) + int(boyun_extension) +
                  int(govde_yan_egim) + int(govde_donus) + int(govde_extension) +
                  int(omuz_kalkmis) + int(kol_abdukte) + int(bilek_donus) - int(kol_destekli))
    if toplam_mod != 0:
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
                    padding:8px 12px;font-size:12px;margin-top:4px">
            Aktif modifier: <strong>{toplam_mod:+d}</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🎨 Görsel Mod")
    annotation_mode = st.selectbox(
        "Annotation",
        list(ANNOTATION_MODES.keys()),
        index=1,  # default: standard
        format_func=lambda x: f"{x.capitalize()} — {ANNOTATION_MODES[x]}",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px;color:#94a3b8;line-height:2">
    <strong style="color:#475569">Risk Skalası</strong><br>
    🟢 1 Önemsiz &nbsp; 🟡 2–3 Düşük<br>
    🟠 4–7 Orta &nbsp; 🔴 8–10 Yüksek<br>
    🟣 11+ Çok Yüksek
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# ANA ALAN — YÜKLEME
# ════════════════════════════════════════════════════════

col_yuk, col_param = st.columns([3, 2])

with col_yuk:
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#475569;
                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">
        📁 Fotoğraf Yükle
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Bir veya birden fazla fotoğraf yükleyin — her biri için ayrı REBA analizi yapılır.<br>
    Desteklenen: <strong>JPG, PNG, WEBP</strong>
    </div>
    """, unsafe_allow_html=True)
    yuklenen = st.file_uploader(
        "Fotoğraf",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if yuklenen:
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;
                    padding:8px 12px;font-size:12px;color:#166534;margin-top:6px">
        ✓ <strong>{len(yuklenen)} fotoğraf</strong> yüklendi
        </div>
        """, unsafe_allow_html=True)

with col_param:
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#475569;
                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">
        ℹ️ Aktif Parametreler
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-size:12px">
    <tr style="border-bottom:1px solid #f1f5f9">
        <td style="padding:6px 4px;color:#64748b">Yük</td>
        <td style="padding:6px 4px;font-weight:600">{yuk_kg:.1f} kg → +{yuk_skoru_val}</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9">
        <td style="padding:6px 4px;color:#64748b">Tutma</td>
        <td style="padding:6px 4px;font-weight:600">+{tutma_val} — {tutma_label}</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9">
        <td style="padding:6px 4px;color:#64748b">Aktivite</td>
        <td style="padding:6px 4px;font-weight:600">
            +{aktivite_val}{(' — '+', '.join(akt_aci)) if akt_aci else ''}
        </td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9">
        <td style="padding:6px 4px;color:#64748b">Görsel Mod</td>
        <td style="padding:6px 4px;font-weight:600">{annotation_mode.capitalize()}</td>
    </tr>
    <tr>
        <td style="padding:6px 4px;color:#64748b">Fotoğraf</td>
        <td style="padding:6px 4px;font-weight:600">
            {len(yuklenen) if yuklenen else 0} adet
        </td>
    </tr>
    </table>
    """, unsafe_allow_html=True)

# Analiz butonu
form_tamam = bool(bolum or is_istasyonu or is_adimi) and bool(yuklenen)

st.markdown("---")
col_btn, col_uyari = st.columns([1, 4])
with col_btn:
    calistir = st.button("▶  ANALİZİ BAŞLAT",
                          disabled=not form_tamam,
                          use_container_width=True, type="primary")
with col_uyari:
    if not form_tamam:
        eksikler = []
        if not (bolum or is_istasyonu or is_adimi):
            eksikler.append("form bilgisi")
        if not yuklenen:
            eksikler.append("fotoğraf")
        st.markdown(f"""
        <div class="warn-box" style="margin-top:4px">
        ⚠️ Analiz için gerekli: <strong>{' ve '.join(eksikler)}</strong><br>
        <span style="font-size:11px">Sidebar'dan (☰) form bilgilerini doldurun.</span>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# ANALİZ
# ════════════════════════════════════════════════════════

if calistir and form_tamam:
    st.markdown("---")
    pb    = st.progress(0)
    durum = st.empty()
    foto_sonuclari: List[FotoSonuc] = []

    for i, dosya in enumerate(yuklenen):
        durum.text(f"Analiz: {dosya.name} ({i+1}/{len(yuklenen)})")
        pb.progress((i + 0.5) / len(yuklenen))

        fs = FotoSonuc(idx=i+1, dosya_adi=dosya.name)
        try:
            img_bytes = dosya.read()
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                fs.hata = "Görüntü okunamadı"
            else:
                h_i, w_i = img.shape[:2]
                if w_i > 1280:
                    sc = 1280 / w_i
                    img = cv2.resize(img, (1280, int(h_i * sc)))

                pose = mp.solutions.pose.Pose(
                    static_image_mode=True, model_complexity=1,
                    min_detection_confidence=0.5)
                res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                pose.close()

                if not res.pose_landmarks:
                    fs.hata = "Kişi tespit edilemedi"
                else:
                    lms = res.pose_landmarks.landmark
                    h2, w2 = img.shape[:2]
                    a_obj = vucut_acilari_hesapla(lms, w2, h2)
                    skor  = reba_skorla(
                        a_obj, yuk_skoru_val, tutma_val, aktivite_val,
                        boyun_yan_egim=boyun_yan_egim,
                        boyun_donus=boyun_donus,
                        boyun_extension=boyun_extension,
                        govde_yan_egim=govde_yan_egim,
                        govde_donus=govde_donus,
                        govde_extension=govde_extension,
                        omuz_kalkmis=omuz_kalkmis,
                        kol_abdukte=kol_abdukte,
                        kol_destekli=kol_destekli,
                        bilek_donus=bilek_donus,
                    )
                    fs.skor = skor
                    fs.overlay_img = overlay_ciz(img.copy(), lms, skor, mode=annotation_mode)
        except Exception as e:
            fs.hata = str(e)

        foto_sonuclari.append(fs)
        pb.progress((i + 1) / len(yuklenen))

    durum.empty()
    pb.empty()

    gecerli = [s for s in foto_sonuclari if s.skor is not None]

    if not gecerli:
        st.markdown("""
        <div class="error-box">
        ❌ Hiçbir fotoğrafta kişi tespit edilemedi.<br>
        Fotoğrafların tam vücut, net ve iyi aydınlatılmış olduğundan emin olun.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── ÖZET METRİKLER ──
    skorlar = [s.skor.final_skor for s in gecerli]
    ort     = sum(skorlar) / len(skorlar)
    en_yuk  = max(skorlar)
    en_dus  = min(skorlar)

    ort_r, ort_c = risk_info(round(ort))
    yuk_r, yuk_c = risk_info(en_yuk)

    st.markdown(f"""
    <div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:10px">
        📊 Genel Özet — {len(gecerli)} Fotoğraf
    </div>
    <div class="metric-row">
        <div class="metric-box">
            <div class="val" style="color:{ort_c}">{ort:.1f}</div>
            <div class="lbl">Ortalama REBA</div>
            <span class="risk-badge"
                  style="background:{ort_c}18;color:{ort_c};border:1px solid {ort_c}40">
                {ort_r}
            </span>
        </div>
        <div class="metric-box">
            <div class="val" style="color:{yuk_c}">{en_yuk}</div>
            <div class="lbl">En Yüksek REBA</div>
            <span class="risk-badge"
                  style="background:{yuk_c}18;color:{yuk_c};border:1px solid {yuk_c}40">
                {yuk_r}
            </span>
        </div>
        <div class="metric-box">
            <div class="val" style="color:#475569">{en_dus}</div>
            <div class="lbl">En Düşük REBA</div>
        </div>
        <div class="metric-box">
            <div class="val" style="color:#0f172a">{len(gecerli)}</div>
            <div class="lbl">Analiz Edilen</div>
            {f'<div style="font-size:10px;color:#dc2626;margin-top:2px">{len(foto_sonuclari)-len(gecerli)} başarısız</div>'
             if len(foto_sonuclari) > len(gecerli) else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Başarısız + düşük güven uyarıları
    for fs in foto_sonuclari:
        if fs.hata:
            st.markdown(f"""
            <div class="warn-box">⚠️ <strong>{fs.dosya_adi}</strong>: {fs.hata}</div>
            """, unsafe_allow_html=True)

    # #5: Düşük güven uyarısı
    for fs in gecerli:
        if fs.skor and fs.skor.acılar and fs.skor.acılar.guven < 0.5:
            st.markdown(f"""
            <div class="warn-box">
            ⚠️ <strong>{fs.dosya_adi}</strong>: AI güven skoru düşük ({fs.skor.acılar.guven:.0%}).
            Bazı eklemler net görünmüyor — sonuç güvenilirliği sınırlı olabilir.
            Fotoğrafı farklı açıdan tekrar çekmeyi deneyin.
            </div>
            """, unsafe_allow_html=True)

        # Bacak görünmüyor uyarısı
        if fs.skor and fs.skor.acılar and not fs.skor.acılar.bacak_gozukuyor:
            st.markdown(f"""
            <div class="warn-box">
            ⚠️ <strong>{fs.dosya_adi}</strong>: Bacak (diz/ayak) görünmüyor.
            Bacak skoru varsayılan olarak hesaplandı (dik duruş). 
            Daha doğru analiz için tam vücut fotoğrafı çekin.
            </div>
            """, unsafe_allow_html=True)

    # ── FOTO DETAYLARI ──
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;
                letter-spacing:0.1em;margin:14px 0 10px">
        🔍 Fotoğraf Bazlı Analiz
    </div>
    """, unsafe_allow_html=True)

    for fs in gecerli:
        s = fs.skor
        a = s.acılar
        _, renk = risk_info(s.final_skor)

        st.markdown(f"""
        <div class="foto-header" style="background:{renk}">
            <span>📷 Fotoğraf {fs.idx} · {fs.dosya_adi}
                  {f' · Taraf: {a.analiz_tarafi}' if a else ''}</span>
            <span>REBA {s.final_skor}/15 · {s.risk_seviyesi}</span>
        </div>
        """, unsafe_allow_html=True)

        col_img, col_analiz = st.columns([2, 3])

        with col_img:
            if fs.overlay_img is not None:
                rgb = cv2.cvtColor(fs.overlay_img, cv2.COLOR_BGR2RGB)
                st.image(rgb, use_container_width=True)

        with col_analiz:
            st.markdown("**Segment Skorları**")

            segs = [
                ("Boyun", s.boyun_skoru, 6,
                 f"{a.boyun_flexion:.0f}°"
                 + ("+Ext" if a.boyun_extension else "")
                 + ("+YanEğ" if a.boyun_yan_egim > 15 else "")
                 + ("+Dön" if a.boyun_donus else "")),
                ("Gövde", s.govde_skoru, 5,
                 f"{a.govde_flexion:.0f}°"
                 + ("+Ext" if a.govde_extension else "")
                 + ("+YanEğ" if a.govde_yan_egim > 10 else "")
                 + ("+Dön" if a.govde_donus else "")),
                ("Bacak/Diz", s.bacak_skoru, 4,
                 f"Diz {max(a.diz_flexion_sol, a.diz_flexion_sag):.0f}°"),
                (f"Üst Kol ({a.analiz_tarafi})", s.ust_kol_skoru, 6,
                 f"{a.ust_kol_aci:.0f}°"
                 + ("+OmKalK" if a.omuz_kalkmis else "")
                 + ("+Abd" if a.kol_abdukte else "")),
                (f"Alt Kol ({a.analiz_tarafi})", s.alt_kol_skoru, 2,
                 f"{a.alt_kol_aci:.0f}°"),
                (f"Bilek ({a.analiz_tarafi})", s.bilek_skoru, 3,
                 f"{a.bilek_aci:.0f}°" + ("+Dön" if a.bilek_donus else "")),
            ]

            for seg_ad, seg_val, seg_max, seg_aci in segs:
                seg_renk = segment_risk_renk(seg_val, seg_max)
                pct = seg_val / seg_max
                st.markdown(f"""
                <div class="seg-bar-wrap">
                    <div class="seg-bar-label">
                        <span>{seg_ad}
                            <span style="color:#94a3b8;font-size:11px">&nbsp;{seg_aci}</span>
                        </span>
                        <span style="font-weight:700;color:{seg_renk};
                                     font-family:'IBM Plex Mono',monospace">
                            {seg_val}/{seg_max}
                        </span>
                    </div>
                    <div class="seg-bar-track">
                        <div class="seg-bar-fill"
                             style="width:{pct*100:.0f}%;background:{seg_renk}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Skor hesaplama
            st.markdown(f"""
            <div class="skor-ozet">
                <div style="display:flex;gap:14px;flex-wrap:wrap;color:#374151">
                    <span>Tablo A: <strong>{s.tablo_a}</strong>
                          + Yük <strong>+{s.yuk_skoru}</strong>
                          = Skor A: <strong style="color:#1d4ed8">{s.skor_a}</strong></span>
                    <span>Tablo B: <strong>{s.tablo_b}</strong>
                          + Tutma <strong>+{s.tutma_skoru}</strong>
                          = Skor B: <strong style="color:#1d4ed8">{s.skor_b}</strong></span>
                </div>
                <div style="margin-top:7px;border-top:1px solid #e2e8f0;padding-top:7px">
                    Tablo C: <strong>{s.skor_c}</strong>
                    + Aktivite: <strong>+{s.aktivite_skoru}</strong>
                    &nbsp;=&nbsp;
                    <strong style="font-size:15px;color:{renk}">REBA {s.final_skor}</strong>
                    &nbsp;
                    <span style="color:{renk};font-weight:600">{s.risk_seviyesi}</span>
                    <br>
                    <span style="color:#64748b;font-size:11px">
                        → {s.aksiyon} · AI Güven: {a.guven:.0%}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # #9: Explainable AI — UI'da
            if s.aciklama and s.final_skor >= 4:
                rows_html = ""
                for item in s.aciklama:
                    rows_html += f"""
                    <tr>
                        <td>{item.get('segment','')}</td>
                        <td>{item.get('aci','')}</td>
                        <td><strong>{item.get('temel','')}</strong></td>
                        <td>{item.get('aciklama','')}</td>
                    </tr>"""
                st.markdown(f"""
                <details style="margin-top:8px">
                    <summary style="font-size:12px;font-weight:600;color:#1e3a5f;cursor:pointer">
                        🔎 Neden Bu Skor?
                    </summary>
                    <table class="explain-table">
                        <tr><th>Segment</th><th>Açı</th><th>Skor</th><th>Açıklama</th></tr>
                        {rows_html}
                    </table>
                </details>
                """, unsafe_allow_html=True)

        st.markdown(
            '<div style="border-bottom:1px solid #e2e8f0;margin:8px 0 18px"></div>',
            unsafe_allow_html=True)

    # ── PDF ──
    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:10px">📄 Rapor İndir</div>
    """, unsafe_allow_html=True)

    akt_aciklama_str = ', '.join(akt_aci) if akt_aci else "Yok"
    form_bilgi = {
        'bolum': bolum, 'is_istasyonu': is_istasyonu,
        'is_adimi': is_adimi, 'analist': analist, 'tarih': str(tarih),
        'yuk_kg': yuk_kg, 'shock': shock, 'yuk_skoru': yuk_skoru_val,
        'yuk_aciklama': yuk_aci + (" + Ani kuvvet +1" if shock else ""),
        'tutma': tutma_val, 'tutma_label': tutma_label,
        'aktivite': aktivite_val, 'aktivite_aciklama': akt_aciklama_str,
    }

    col_p, col_b = st.columns([1, 3])
    with col_p:
        with st.spinner("PDF hazırlanıyor..."):
            try:
                pdf_bytes = pdf_olustur(form_bilgi, foto_sonuclari)
            except Exception as e:
                pdf_bytes = b""
                st.error(f"PDF hatası: {e}")

        if pdf_bytes:
            ad = f"REBA_{(bolum or is_istasyonu or 'analiz').replace(' ','_')}_{tarih}.pdf"
            st.download_button("⬇️  PDF Raporu İndir",
                               data=pdf_bytes, file_name=ad,
                               mime="application/pdf",
                               use_container_width=True)

# ════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════

st.markdown("""
<div class="app-footer">
    REBA Ergonomi Risk Analiz Sistemi v5.2 &nbsp;·&nbsp;
    Hignett & McAtamney (2000), Applied Ergonomics 31(2), 201–205 &nbsp;·&nbsp;
    MediaPipe Pose (Google LLC) &nbsp;·&nbsp;
    AI tabanlı açı tahmini ±3–5° doğruluk payı içerir —
    profesyonel değerlendirmenin yerini tutmaz
</div>
""", unsafe_allow_html=True)
