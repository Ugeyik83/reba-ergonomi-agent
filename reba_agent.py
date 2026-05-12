"""
<<<<<<< HEAD
REBA Ergonomi Analiz Ajanı v5.0
İSG Uzmanları için Profesyonel Çok Fotoğraflı Değerlendirme Aracı
=======
REBA Ergonomi Analiz Ajanı v4.0
================================
İSG Uzmanları için Profesyonel REBA Değerlendirme Aracı
MediaPipe Pose + Manuel Girdi + PDF Rapor
>>>>>>> ddded5f36dedc372c567f0ac5bae861d227b2eb3
"""

import streamlit as st
import cv2
import numpy as np
import math
import os
<<<<<<< HEAD
import io
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, date
=======
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime
import json
import io

>>>>>>> ddded5f36dedc372c567f0ac5bae861d227b2eb3
import mediapipe as mp

LM = mp.solutions.pose.PoseLandmark
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

<<<<<<< HEAD
=======
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAYFA AYARI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

>>>>>>> ddded5f36dedc372c567f0ac5bae861d227b2eb3
st.set_page_config(
    page_title="REBA Ergonomi Analizi",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

<<<<<<< HEAD
# ════════════════════════════════════════════════════════
# VERİ SINIFLARI
# ════════════════════════════════════════════════════════

class AcilarObj:
    """Hesaplanan vücut segment açıları."""
    def __init__(self):
        self.boyun_flexion = 0.0
        self.boyun_yan_egim = 0.0
        self.boyun_donus = False
        self.govde_flexion = 0.0
        self.govde_yan_egim = 0.0
        self.govde_donus = False
        self.diz_flexion_sol = 0.0
        self.diz_flexion_sag = 0.0
        self.ust_kol_aci = 0.0
        self.omuz_kalkmis = False
        self.kol_abdukte = False
        self.kol_destekli = False
        self.alt_kol_aci = 0.0
        self.bilek_aci = 0.0
        self.bilek_donus = False
        self.bilateral_destek = True
        self.guven = 0.0

@dataclass
class REBASkoru:
    boyun_skoru: int = 0
    govde_skoru: int = 0
    bacak_skoru: int = 0
    ust_kol_skoru: int = 0
    alt_kol_skoru: int = 0
    bilek_skoru: int = 0
    tablo_a: int = 0
    tablo_b: int = 0
    yuk_skoru: int = 0
    tutma_skoru: int = 0
    aktivite_skoru: int = 0
    skor_a: int = 0
    skor_b: int = 0
    skor_c: int = 0
    final_skor: int = 0
    risk_seviyesi: str = ""
    aksiyon: str = ""
    renk: str = "#16a34a"
    acılar: Optional[AcilarObj] = None

@dataclass
class FotoSonuc:
    idx: int = 0
    dosya_adi: str = ""
    skor: Optional[REBASkoru] = None
    overlay_img: Optional[np.ndarray] = None
    hata: str = ""

# ════════════════════════════════════════════════════════
# REBA TABLOLARI
# ════════════════════════════════════════════════════════

TABLO_A = [
=======
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VERİ SINIFLARI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class BodyAngles:
    neck_flexion: float = 0.0
    neck_side_bend: float = 0.0
    trunk_flexion: float = 0.0
    trunk_side_bend: float = 0.0
    trunk_twist: bool = False
    knee_flexion_left: float = 0.0
    knee_flexion_right: float = 0.0
    upper_arm_angle: float = 0.0
    shoulder_raised: bool = False
    arm_abducted: bool = False
    leaningOnArm: bool = False
    lower_arm_angle: float = 0.0
    wrist_angle: float = 0.0
    wrist_twist: bool = False
    bilateral_support: bool = True
    confidence: float = 0.0

@dataclass
class REBAResult:
    neck_score: int = 0
    trunk_score: int = 0
    leg_score: int = 0
    upper_arm_score: int = 0
    lower_arm_score: int = 0
    wrist_score: int = 0
    table_a_score: int = 0
    table_b_score: int = 0
    load_score: int = 0
    coupling_score: int = 0
    activity_score: int = 0
    score_a: int = 0
    score_b: int = 0
    score_c: int = 0
    final_score: int = 0
    risk_level: str = ""
    action: str = ""
    color: str = "#22c55e"
    angles: Optional[BodyAngles] = None
    frame_time: float = 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REBA TABLOLARI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TABLE_A = [
>>>>>>> ddded5f36dedc372c567f0ac5bae861d227b2eb3
    [[1,2,3,4],[1,2,3,4],[3,3,5,6]],
    [[2,3,4,5],[3,4,5,6],[4,5,6,7]],
    [[2,4,5,6],[4,5,6,7],[5,6,7,8]],
    [[3,5,6,7],[5,6,7,8],[6,7,8,9]],
    [[4,6,7,8],[6,7,8,9],[7,8,9,9]],
]
<<<<<<< HEAD
TABLO_B = [
=======

TABLE_B = [
>>>>>>> ddded5f36dedc372c567f0ac5bae861d227b2eb3
    [[1,2,2],[1,2,3]],
    [[1,2,3],[2,3,4]],
    [[3,4,5],[4,5,5]],
    [[4,5,5],[5,6,7]],
    [[6,7,8],[7,8,8]],
    [[7,8,8],[8,9,9]],
]
<<<<<<< HEAD
TABLO_C = [
=======

TABLE_C = [
>>>>>>> ddded5f36dedc372c567f0ac5bae861d227b2eb3
    [1,1,1,2,3,3,4,5,6,7,7,7],
    [1,2,2,3,4,4,5,6,6,7,7,8],
    [2,3,3,3,4,5,6,7,7,8,8,8],
    [3,4,4,4,5,6,7,8,8,9,9,9],
    [4,4,4,5,6,7,8,8,9,9,9,9],
    [6,6,6,7,8,8,9,9,10,10,10,10],
    [7,7,7,8,9,9,9,10,10,11,11,11],
    [8,8,8,9,10,10,10,10,10,11,11,11],
    [9,9,9,10,10,10,11,11,11,12,12,12],
    [10,10,10,11,11,11,11,12,12,12,12,12],
    [11,11,11,11,12,12,12,12,12,12,12,12],
    [12,12,12,12,12,12,12,12,12,12,12,12],
]
<<<<<<< HEAD

# ════════════════════════════════════════════════════════
# AÇI HESAPLAMA
# ════════════════════════════════════════════════════════

def aci_3nokta(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    mag = np.linalg.norm(ba) * np.linalg.norm(bc)
    if mag == 0: return 0.0
    return math.degrees(math.acos(np.clip(np.dot(ba,bc)/mag, -1.0, 1.0)))

def aci_dikey(a, b):
    return abs(math.degrees(math.atan2(abs(b[0]-a[0]), abs(b[1]-a[1]))))

def vucut_acilari_hesapla(landmarks, w, h) -> AcilarObj:
    a = AcilarObj()

    def p(idx):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)
    def v(idx):
        return landmarks[idx].visibility

    mid_omuz = (
        (p(LM.LEFT_SHOULDER)[0] + p(LM.RIGHT_SHOULDER)[0]) / 2,
        (p(LM.LEFT_SHOULDER)[1] + p(LM.RIGHT_SHOULDER)[1]) / 2
=======

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AÇI HESAPLAMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_angle_3p(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    dot = np.dot(ba, bc)
    mag = np.linalg.norm(ba) * np.linalg.norm(bc)
    if mag == 0: return 0.0
    return math.degrees(math.acos(np.clip(dot/mag, -1.0, 1.0)))

def calc_angle_vertical(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return abs(math.degrees(math.atan2(abs(dx), abs(dy))))

def calculate_body_angles(landmarks, w, h):
    angles = BodyAngles()

    def pt(idx):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    def vis(idx):
        return landmarks[idx].visibility

    mid_shoulder = (
        (pt(LM.LEFT_SHOULDER)[0] + pt(LM.RIGHT_SHOULDER)[0]) / 2,
        (pt(LM.LEFT_SHOULDER)[1] + pt(LM.RIGHT_SHOULDER)[1]) / 2
>>>>>>> ddded5f36dedc372c567f0ac5bae861d227b2eb3
    )
    mid_kalca = (
        (p(LM.LEFT_HIP)[0] + p(LM.RIGHT_HIP)[0]) / 2,
        (p(LM.LEFT_HIP)[1] + p(LM.RIGHT_HIP)[1]) / 2
    )

<<<<<<< HEAD
    a.guven = float(np.mean([v(l) for l in [
        LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
        LM.LEFT_HIP, LM.RIGHT_HIP
    ]]))

    # BOYUN
    a.boyun_flexion = aci_dikey(p(LM.NOSE), mid_omuz)
    sol_kulak = p(LM.LEFT_EAR)
    sag_kulak = p(LM.RIGHT_EAR)
    ear_dx = abs(sol_kulak[0] - sag_kulak[0])
    ear_dy = abs(sol_kulak[1] - sag_kulak[1])
    if ear_dx > 0:
        a.boyun_yan_egim = math.degrees(math.atan2(ear_dy, ear_dx))
    a.boyun_donus = a.boyun_yan_egim > 20

    # GÖVDE
    a.govde_flexion = aci_dikey(mid_omuz, mid_kalca)
    lateral = abs(mid_omuz[0] - mid_kalca[0])
    yukseklik = abs(mid_omuz[1] - mid_kalca[1])
    if yukseklik > 0:
        a.govde_yan_egim = math.degrees(math.atan2(lateral, yukseklik))
    omuz_gen = abs(p(LM.LEFT_SHOULDER)[0] - p(LM.RIGHT_SHOULDER)[0])
    kalca_gen = abs(p(LM.LEFT_HIP)[0] - p(LM.RIGHT_HIP)[0])
    if kalca_gen > 0:
        ratio = omuz_gen / kalca_gen
        a.govde_donus = ratio < 0.7 or ratio > 1.4

    # BACAKLAR
    a.diz_flexion_sol = 180 - aci_3nokta(
        p(LM.LEFT_HIP), p(LM.LEFT_KNEE), p(LM.LEFT_ANKLE))
    a.diz_flexion_sag = 180 - aci_3nokta(
        p(LM.RIGHT_HIP), p(LM.RIGHT_KNEE), p(LM.RIGHT_ANKLE))
    a.bilateral_destek = v(LM.LEFT_ANKLE) > 0.3 and v(LM.RIGHT_ANKLE) > 0.3

    # ÜST KOL
    ua_sol = aci_3nokta(p(LM.LEFT_HIP), p(LM.LEFT_SHOULDER), p(LM.LEFT_ELBOW))
    ua_sag = aci_3nokta(p(LM.RIGHT_HIP), p(LM.RIGHT_SHOULDER), p(LM.RIGHT_ELBOW))
    a.ust_kol_aci = max(ua_sol, ua_sag)
    ref = abs(p(LM.LEFT_HIP)[1] - p(LM.LEFT_SHOULDER)[1])
    sol_se = abs(p(LM.LEFT_SHOULDER)[1] - p(LM.LEFT_EAR)[1])
    sag_se = abs(p(LM.RIGHT_SHOULDER)[1] - p(LM.RIGHT_EAR)[1])
    if ref > 0:
        a.omuz_kalkmis = min(sol_se, sag_se) / ref < 0.3
    if omuz_gen > 0:
        sol_abd = abs(p(LM.LEFT_ELBOW)[0] - p(LM.LEFT_SHOULDER)[0])
        sag_abd = abs(p(LM.RIGHT_ELBOW)[0] - p(LM.RIGHT_SHOULDER)[0])
        a.kol_abdukte = max(sol_abd, sag_abd) / omuz_gen > 0.8

    # ALT KOL
    la_sol = aci_3nokta(
        p(LM.LEFT_SHOULDER), p(LM.LEFT_ELBOW), p(LM.LEFT_WRIST))
    la_sag = aci_3nokta(
        p(LM.RIGHT_SHOULDER), p(LM.RIGHT_ELBOW), p(LM.RIGHT_WRIST))
    a.alt_kol_aci = min(la_sol, la_sag)

    # BİLEK
    w_sol = abs(180 - aci_3nokta(
        p(LM.LEFT_ELBOW), p(LM.LEFT_WRIST), p(LM.LEFT_INDEX)))
    w_sag = abs(180 - aci_3nokta(
        p(LM.RIGHT_ELBOW), p(LM.RIGHT_WRIST), p(LM.RIGHT_INDEX)))
    a.bilek_aci = max(w_sol, w_sag)
    bilek_y = abs(p(LM.LEFT_WRIST)[1] - p(LM.RIGHT_WRIST)[1])
    if ref > 0:
        a.bilek_donus = bilek_y / ref > 0.15

    return a

# ════════════════════════════════════════════════════════
# REBA SKORLAMA
# ════════════════════════════════════════════════════════

def reba_skorla(a: AcilarObj, yuk_skoru: int, tutma: int, aktivite: int) -> REBASkoru:
    r = REBASkoru()
    r.yuk_skoru = yuk_skoru
    r.tutma_skoru = tutma
    r.aktivite_skoru = aktivite
    r.acılar = a

    # BOYUN
    if a.boyun_flexion <= 20: r.boyun_skoru = 1
    elif a.boyun_flexion <= 40: r.boyun_skoru = 2
    else: r.boyun_skoru = 3
    if a.boyun_yan_egim > 15: r.boyun_skoru += 1
    if a.boyun_donus: r.boyun_skoru += 1
    r.boyun_skoru = min(r.boyun_skoru, 6)

    # GÖVDE
    if a.govde_flexion <= 5: r.govde_skoru = 1
    elif a.govde_flexion <= 20: r.govde_skoru = 2
    elif a.govde_flexion <= 60: r.govde_skoru = 3
    else: r.govde_skoru = 4
    if a.govde_yan_egim > 10: r.govde_skoru += 1
    if a.govde_donus: r.govde_skoru += 1
    r.govde_skoru = min(r.govde_skoru, 5)

    # BACAK
    r.bacak_skoru = 1 if a.bilateral_destek else 2
    diz = max(a.diz_flexion_sol, a.diz_flexion_sag)
    if 30 <= diz < 60: r.bacak_skoru += 1
    elif diz >= 60: r.bacak_skoru += 2
    r.bacak_skoru = min(r.bacak_skoru, 4)

    # TABLO A
    t = min(r.govde_skoru - 1, 4)
    n = min(r.boyun_skoru - 1, 2)
    l = min(r.bacak_skoru - 1, 3)
    r.tablo_a = TABLO_A[t][n][l]
    r.skor_a = r.tablo_a + r.yuk_skoru

    # ÜST KOL
    if a.ust_kol_aci <= 20: r.ust_kol_skoru = 1
    elif a.ust_kol_aci <= 45: r.ust_kol_skoru = 2
    elif a.ust_kol_aci <= 90: r.ust_kol_skoru = 3
    else: r.ust_kol_skoru = 4
    if a.omuz_kalkmis: r.ust_kol_skoru += 1
    if a.kol_abdukte: r.ust_kol_skoru += 1
    if a.kol_destekli: r.ust_kol_skoru -= 1
    r.ust_kol_skoru = max(1, min(r.ust_kol_skoru, 6))

    # ALT KOL
    r.alt_kol_skoru = 1 if 60 <= a.alt_kol_aci <= 100 else 2

    # BİLEK
    r.bilek_skoru = 1 if a.bilek_aci <= 15 else 2
    if a.bilek_donus: r.bilek_skoru += 1
    r.bilek_skoru = min(r.bilek_skoru, 3)

    # TABLO B
    u = min(r.ust_kol_skoru - 1, 5)
    la = min(r.alt_kol_skoru - 1, 1)
    w = min(r.bilek_skoru - 1, 2)
    r.tablo_b = TABLO_B[u][la][w]
    r.skor_b = r.tablo_b + r.tutma_skoru

    # TABLO C + AKTİVİTE
    ca = min(r.skor_a - 1, 11)
    cb = min(r.skor_b - 1, 11)
    r.skor_c = TABLO_C[ca][cb]
    r.final_skor = min(r.skor_c + r.aktivite_skoru, 15)

    # RİSK SEVİYESİ
    s = r.final_skor
    if s == 1:
        r.risk_seviyesi = "Önemsiz Risk"; r.renk = "#16a34a"
        r.aksiyon = "Herhangi bir önlem gerekmez"
    elif s <= 3:
        r.risk_seviyesi = "Düşük Risk"; r.renk = "#65a30d"
        r.aksiyon = "Gerekirse iyileştirme yapılabilir"
    elif s <= 7:
        r.risk_seviyesi = "Orta Seviyeli Risk"; r.renk = "#d97706"
        r.aksiyon = "Daha ayrıntılı incele, değişiklik planla"
    elif s <= 10:
        r.risk_seviyesi = "Yüksek Risk"; r.renk = "#dc2626"
        r.aksiyon = "Araştırma yap ve aksiyon al"
    else:
        r.risk_seviyesi = "Çok Yüksek Risk"; r.renk = "#7c3aed"
        r.aksiyon = "Süreç çalışmaya uygun değil, derhal revize et"

    return r

# ════════════════════════════════════════════════════════
# İSKELET + AÇI OVERLAY
# ════════════════════════════════════════════════════════

def overlay_ciz(img: np.ndarray, landmarks, skor: REBASkoru) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()

    def p(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))
    def v(idx):
        return landmarks[idx].visibility

    # Renk seçimi
    s = skor.final_skor
    if s <= 3: col = (22, 163, 74)
    elif s <= 7: col = (217, 119, 6)
    elif s <= 10: col = (220, 38, 38)
    else: col = (124, 58, 237)

    # İskelet çizgileri
    for conn in POSE_CONNECTIONS:
        sl = landmarks[conn[0]]
        el = landmarks[conn[1]]
        if sl.visibility > 0.4 and el.visibility > 0.4:
            cv2.line(out, p(conn[0]), p(conn[1]), col, 2, cv2.LINE_AA)

    # Eklem noktaları
    for lm in landmarks:
        if lm.visibility > 0.4:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(out, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, pt, 5, col, 1, cv2.LINE_AA)

    # Açı etiketi yaz
    def etiket(img, nokta, metin, renk=(20, 20, 80)):
        x, y = nokta
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, thick = 0.4, 1
        (tw, th), _ = cv2.getTextSize(metin, font, fs, thick)
        x = max(2, min(x, w - tw - 6))
        y = max(th + 4, min(y, h - 4))
        cv2.rectangle(img, (x-2, y-th-4), (x+tw+4, y+4), (255, 255, 240), -1)
        cv2.rectangle(img, (x-2, y-th-4), (x+tw+4, y+4), renk, 1)
        cv2.putText(img, metin, (x, y), font, fs, renk, thick, cv2.LINE_AA)

    a = skor.acılar
    if a is None:
        return out

    mid_omuz_x = int((landmarks[LM.LEFT_SHOULDER].x + landmarks[LM.RIGHT_SHOULDER].x) / 2 * w)
    mid_omuz_y = int((landmarks[LM.LEFT_SHOULDER].y + landmarks[LM.RIGHT_SHOULDER].y) / 2 * h)
    mid_kalca_y = int((landmarks[LM.LEFT_HIP].y + landmarks[LM.RIGHT_HIP].y) / 2 * h)

    # Boyun
    if v(LM.NOSE) > 0.4:
        nx, ny = p(LM.NOSE)
        mod = ("YE" if a.boyun_yan_egim > 15 else "") + ("D" if a.boyun_donus else "")
        mod_str = f"+{mod}" if mod else ""
        etiket(out, (nx + 8, ny - 6),
               f"Boyun:{a.boyun_flexion:.0f}{mod_str} [{skor.boyun_skoru}]",
               renk=(15, 80, 160))

    # Gövde
    gy = (mid_omuz_y + mid_kalca_y) // 2
    mod = ("YE" if a.govde_yan_egim > 10 else "") + ("D" if a.govde_donus else "")
    mod_str = f"+{mod}" if mod else ""
    etiket(out, (mid_omuz_x + 10, gy),
           f"Govde:{a.govde_flexion:.0f}{mod_str} [{skor.govde_skoru}]",
           renk=(140, 60, 0))

    # Bacak
    if v(LM.LEFT_KNEE) > 0.4:
        kx, ky = p(LM.LEFT_KNEE)
        etiket(out, (kx + 6, ky),
               f"Diz:{max(a.diz_flexion_sol, a.diz_flexion_sag):.0f} [{skor.bacak_skoru}]",
               renk=(70, 100, 20))

    # Üst Kol
    if v(LM.RIGHT_ELBOW) > 0.4:
        ex, ey = p(LM.RIGHT_ELBOW)
        mod = ("OK" if a.omuz_kalkmis else "") + ("AB" if a.kol_abdukte else "")
        mod_str = f"+{mod}" if mod else ""
        etiket(out, (ex + 6, ey - 8),
               f"UKol:{a.ust_kol_aci:.0f}{mod_str} [{skor.ust_kol_skoru}]",
               renk=(120, 20, 120))

    # Alt Kol
    if v(LM.RIGHT_WRIST) > 0.4:
        wx, wy = p(LM.RIGHT_WRIST)
        etiket(out, (wx + 6, wy - 16),
               f"AKol:{a.alt_kol_aci:.0f} [{skor.alt_kol_skoru}]",
               renk=(0, 100, 140))
        # Bilek
        mod_str = "+D" if a.bilek_donus else ""
        etiket(out, (wx + 6, wy + 4),
               f"Bilek:{a.bilek_aci:.0f}{mod_str} [{skor.bilek_skoru}]",
               renk=(0, 80, 160))

    # REBA skor kutusu
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

# ════════════════════════════════════════════════════════
# PDF RAPORU
# ════════════════════════════════════════════════════════

def pdf_olustur(form_bilgi: dict, foto_sonuclari: List[FotoSonuc]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable,
                                     PageBreak, Image as RLImage)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import glob, tempfile

    # Türkçe font
    FONT, FONT_BOLD = 'Helvetica', 'Helvetica-Bold'
    try:
        candidates = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
        dejavu = [f for f in candidates if 'DejaVuSans' in f and 'Bold' not in f]
        dejavu_bold = [f for f in candidates if 'DejaVuSans-Bold' in f]
        if dejavu:
            pdfmetrics.registerFont(TTFont('TRF', dejavu[0]))
            FONT = 'TRF'
        if dejavu_bold:
            pdfmetrics.registerFont(TTFont('TRFB', dejavu_bold[0]))
            FONT_BOLD = 'TRFB'
        elif dejavu:
            pdfmetrics.registerFont(TTFont('TRFB', dejavu[0]))
            FONT_BOLD = 'TRFB'
    except Exception:
        pass

    gecerli = [s for s in foto_sonuclari if s.skor is not None]
    if not gecerli:
        return b""

    skorlar = [s.skor.final_skor for s in gecerli]
    ort = sum(skorlar) / len(skorlar)
    en_yuk_obj = max(gecerli, key=lambda x: x.skor.final_skor).skor

    KOYU = colors.HexColor('#1e3a5f')
    ACIK = colors.HexColor('#f0f4f8')
    CIZGI = colors.HexColor('#cbd5e1')

    def rk(skor):
        if skor <= 1: return colors.HexColor('#dcfce7')
        elif skor <= 3: return colors.HexColor('#ecfccb')
        elif skor <= 7: return colors.HexColor('#fef9c3')
        elif skor <= 10: return colors.HexColor('#fee2e2')
        else: return colors.HexColor('#ede9fe')

    def rt(skor):
        if skor <= 1: return colors.HexColor('#16a34a')
        elif skor <= 3: return colors.HexColor('#65a30d')
        elif skor <= 7: return colors.HexColor('#d97706')
        elif skor <= 10: return colors.HexColor('#dc2626')
        else: return colors.HexColor('#7c3aed')

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    ss = getSampleStyleSheet()

    def P(txt, bold=False, size=9, color=colors.black, space_after=0):
        fn = FONT_BOLD if bold else FONT
        st = ParagraphStyle('x', parent=ss['Normal'], fontName=fn,
                            fontSize=size, textColor=color, spaceAfter=space_after)
        return Paragraph(txt, st)

    def ts_base():
        return TableStyle([
            ('FONTNAME', (0,0), (-1,-1), FONT),
            ('FONTNAME', (0,0), (-1,0), FONT_BOLD),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('BACKGROUND', (0,0), (-1,0), KOYU),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.4, CIZGI),
            ('PADDING', (0,0), (-1,-1), 4),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, ACIK]),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ])

    story = []

    # ── ÖZET SAYFA ──
    story.append(P("REBA Ergonomi Risk Analiz Raporu", bold=True, size=18,
                   color=KOYU, space_after=2))
    story.append(P("Rapid Entire Body Assessment  |  AI Destekli Postur Analizi  |  v5.0",
                   size=9, color=colors.HexColor('#64748b'), space_after=8))
    story.append(HRFlowable(width="100%", thickness=2, color=KOYU))
    story.append(Spacer(1, 0.4*cm))

    # Form
    story.append(P("Form Bilgileri", bold=True, size=11, color=KOYU, space_after=4))
    fr = [
        [P("Bolum", bold=True), form_bilgi.get('bolum','—'),
         P("Tarih", bold=True), form_bilgi.get('tarih','—')],
        [P("Is Istasyonu", bold=True), form_bilgi.get('is_istasyonu','—'),
         P("Analist", bold=True), form_bilgi.get('analist','—')],
        [P("Is Adimi", bold=True), form_bilgi.get('is_adimi','—'),
         P("Olusturma", bold=True), datetime.now().strftime('%d.%m.%Y %H:%M')],
    ]
    tf = Table(fr, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm])
    tf.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,-1), FONT),
        ('FONTSIZE',(0,0),(-1,-1), 8.5),
        ('BACKGROUND',(0,0),(-1,-1), ACIK),
        ('GRID',(0,0),(-1,-1), 0.4, CIZGI),
        ('PADDING',(0,0),(-1,-1), 5),
    ]))
    story.append(tf)
    story.append(Spacer(1, 0.4*cm))

    # Manuel parametreler
    story.append(P("Manuel Girdi Parametreleri", bold=True, size=11, color=KOYU, space_after=4))
    mr = [
        ["Parametre", "Deger", "Aciklama"],
        [f"Yuk: {form_bilgi.get('yuk_kg',0)} kg",
         f"+{form_bilgi.get('yuk_skoru',0)}",
         form_bilgi.get('yuk_aciklama','')],
        ["Ani/Hizli Kuvvet",
         "Evet" if form_bilgi.get('shock') else "Hayir",
         "+1 eklendi" if form_bilgi.get('shock') else "—"],
        ["Tutma Kalitesi",
         f"+{form_bilgi.get('tutma',0)}",
         form_bilgi.get('tutma_label','')],
        ["Aktivite Skoru",
         f"+{form_bilgi.get('aktivite',0)}",
         form_bilgi.get('aktivite_aciklama','')],
    ]
    tm = Table(mr, colWidths=[4.5*cm, 2.5*cm, 11*cm])
    tm.setStyle(ts_base())
    story.append(tm)
    story.append(Spacer(1, 0.4*cm))

    # Genel özet
    story.append(P("Genel Degerlendirme Ozeti", bold=True, size=11, color=KOYU, space_after=4))
    oz = [
        ["Fotograf Sayisi", "Ortalama REBA", "En Yuksek REBA", "Risk Seviyesi"],
        [str(len(gecerli)), f"{ort:.1f}", str(max(skorlar)), en_yuk_obj.risk_seviyesi],
    ]
    to = Table(oz, colWidths=[4*cm, 4*cm, 4*cm, 6*cm])
    sto = ts_base()
    sto.add('FONTSIZE', (0,1), (-1,1), 12)
    sto.add('FONTNAME', (0,1), (-1,1), FONT_BOLD)
    sto.add('ALIGN', (0,0), (-1,-1), 'CENTER')
    sto.add('TEXTCOLOR', (1,1), (1,1), rt(round(ort)))
    sto.add('TEXTCOLOR', (2,1), (2,1), rt(max(skorlar)))
    sto.add('PADDING', (0,0), (-1,-1), 8)
    story.append(to)
    story.append(Spacer(1, 0.4*cm))

    # Foto özet tablosu
    story.append(P("Fotograf Bazli REBA Skorlari", bold=True, size=11, color=KOYU, space_after=4))
    fh = ["No","REBA","Risk","Boyun","Govde","Bacak","UstKol","AltKol","Bilek","SkorA","SkorB"]
    frows = [fh]
    for i, fs in enumerate(gecerli, 1):
        s = fs.skor
        frows.append([str(i), str(s.final_skor), s.risk_seviyesi,
                      str(s.boyun_skoru), str(s.govde_skoru), str(s.bacak_skoru),
                      str(s.ust_kol_skoru), str(s.alt_kol_skoru), str(s.bilek_skoru),
                      str(s.skor_a), str(s.skor_b)])
    cw = [1.2*cm,1.5*cm,3.8*cm,1.4*cm,1.4*cm,1.4*cm,1.8*cm,1.8*cm,1.4*cm,1.5*cm,1.5*cm]
    tf4 = Table(frows, colWidths=cw)
    st4 = ts_base()
    st4.add('ALIGN', (0,0), (-1,-1), 'CENTER')
    for i, fs in enumerate(gecerli, 1):
        st4.add('TEXTCOLOR', (1,i), (1,i), rt(fs.skor.final_skor))
        st4.add('FONTNAME', (1,i), (1,i), FONT_BOLD)
        st4.add('BACKGROUND', (0,i), (-1,i), rk(fs.skor.final_skor))
    tf4.setStyle(st4)
    story.append(tf4)
    story.append(Spacer(1, 0.4*cm))

    # Risk skalası
    story.append(P("REBA Risk Skalasi", bold=True, size=11, color=KOYU, space_after=4))
    rsr = [
        ["Skor","Risk Seviyesi","Onlem"],
        ["1","Onemsiz Risk","Herhangi bir onlem gerekmez"],
        ["2-3","Dusuk Risk","Gerekirse iyilestirme yapilabilir"],
        ["4-7","Orta Seviyeli Risk","Daha ayrintili incele, degisiklik planla"],
        ["8-10","Yuksek Risk","Arastirma yap ve aksiyon al"],
        ["11+","Cok Yuksek Risk","Surec calisismaya uygun degil, derhal revize et"],
    ]
    trs = Table(rsr, colWidths=[2*cm,4.5*cm,11.5*cm])
    strs = ts_base()
    for i, c in enumerate([
        colors.HexColor('#dcfce7'), colors.HexColor('#ecfccb'),
        colors.HexColor('#fef9c3'), colors.HexColor('#fee2e2'),
        colors.HexColor('#ede9fe')
    ], 1):
        strs.add('BACKGROUND', (0,i), (-1,i), c)
    trs.setStyle(strs)
    story.append(trs)

    # ── FOTO SAYFALARI ──
    for idx, fs in enumerate(gecerli, 1):
        story.append(PageBreak())
        s = fs.skor
        a = s.acılar

        story.append(P(f"Fotograf {idx}  —  Detayli REBA Analizi",
                       bold=True, size=16, color=KOYU, space_after=2))
        story.append(P(
            f"REBA Skoru: {s.final_skor}/15  |  {s.risk_seviyesi}  |  {s.aksiyon}",
            size=9, color=colors.HexColor('#64748b'), space_after=6))
        story.append(HRFlowable(width="100%", thickness=1.5, color=KOYU))
        story.append(Spacer(1, 0.3*cm))

        # Overlay görsel
        if fs.overlay_img is not None:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, fs.overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            try:
                img_w = 16*cm
                rl_img = RLImage(tmp_path, width=img_w)
                oh, ow = fs.overlay_img.shape[:2]
                rl_img.height = img_w * (oh / ow)
                story.append(rl_img)
                story.append(Spacer(1, 0.3*cm))
            finally:
                try: os.unlink(tmp_path)
                except: pass

        # Segment tablosu
        story.append(P("Segment Analizi", bold=True, size=11, color=KOYU, space_after=4))

        def temel_skor_boyun(ac):
            if ac <= 20: return 1
            elif ac <= 40: return 2
            return 3

        def temel_skor_govde(ac):
            if ac <= 5: return 1
            elif ac <= 20: return 2
            elif ac <= 60: return 3
            return 4

        def mod_boyun(a):
            m = []
            if a.boyun_yan_egim > 15: m.append("Yana egme +1")
            if a.boyun_donus: m.append("Donus +1")
            return ", ".join(m) or "—"

        def mod_govde(a):
            m = []
            if a.govde_yan_egim > 10: m.append("Yana egme +1")
            if a.govde_donus: m.append("Donus +1")
            return ", ".join(m) or "—"

        def mod_ustkol(a):
            m = []
            if a.omuz_kalkmis: m.append("Omuz kalkis +1")
            if a.kol_abdukte: m.append("Abduksiyon +1")
            if a.kol_destekli: m.append("Destekli -1")
            return ", ".join(m) or "—"

        if a:
            sr = [["Segment","Aci","Temel","Modifikator","Final Skor"]]
            sr.append(["Boyun", f"{a.boyun_flexion:.1f}", str(temel_skor_boyun(a.boyun_flexion)),
                       mod_boyun(a), str(s.boyun_skoru)])
            sr.append(["Govde", f"{a.govde_flexion:.1f}", str(temel_skor_govde(a.govde_flexion)),
                       mod_govde(a), str(s.govde_skoru)])
            sr.append(["Bacak/Diz", f"{max(a.diz_flexion_sol,a.diz_flexion_sag):.1f}",
                       "1" if a.bilateral_destek else "2", "—", str(s.bacak_skoru)])
            sr.append(["Ust Kol", f"{a.ust_kol_aci:.1f}",
                       str(1 if a.ust_kol_aci<=20 else 2 if a.ust_kol_aci<=45 else 3 if a.ust_kol_aci<=90 else 4),
                       mod_ustkol(a), str(s.ust_kol_skoru)])
            sr.append(["Alt Kol", f"{a.alt_kol_aci:.1f}",
                       "1" if 60 <= a.alt_kol_aci <= 100 else "2", "—", str(s.alt_kol_skoru)])
            sr.append(["Bilek", f"{a.bilek_aci:.1f}",
                       "1" if a.bilek_aci <= 15 else "2",
                       "Donus +1" if a.bilek_donus else "—", str(s.bilek_skoru)])

            tseg = Table(sr, colWidths=[3.5*cm, 3*cm, 2.5*cm, 4.5*cm, 2.5*cm])
            tseg.setStyle(ts_base())
            story.append(tseg)
            story.append(Spacer(1, 0.3*cm))

        # Skor hesaplama
        story.append(P("Skor Hesaplama", bold=True, size=11, color=KOYU, space_after=4))
        hr = [
            ["Adim","Hesaplama","Sonuc"],
            ["Tablo A", f"Govde({s.govde_skoru}) x Boyun({s.boyun_skoru}) x Bacak({s.bacak_skoru})", str(s.tablo_a)],
            ["Skor A", f"Tablo A({s.tablo_a}) + Yuk Skoru({s.yuk_skoru})", str(s.skor_a)],
            ["Tablo B", f"UstKol({s.ust_kol_skoru}) x AltKol({s.alt_kol_skoru}) x Bilek({s.bilek_skoru})", str(s.tablo_b)],
            ["Skor B", f"Tablo B({s.tablo_b}) + Tutma Skoru({s.tutma_skoru})", str(s.skor_b)],
            ["Tablo C", f"Skor A({s.skor_a}) x Skor B({s.skor_b})", str(s.skor_c)],
            ["REBA Skoru", f"Tablo C({s.skor_c}) + Aktivite({s.aktivite_skoru})", str(s.final_skor)],
        ]
        th = Table(hr, colWidths=[3.5*cm, 9*cm, 5.5*cm])
        sth = ts_base()
        sth.add('FONTNAME', (2,-1), (2,-1), FONT_BOLD)
        sth.add('FONTSIZE', (2,-1), (2,-1), 12)
        sth.add('TEXTCOLOR', (2,-1), (2,-1), rt(s.final_skor))
        sth.add('BACKGROUND', (0,-1), (-1,-1), rk(s.final_skor))
        th.setStyle(sth)
        story.append(th)

        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=CIZGI))
        story.append(P(
            f"REBA Analiz Ajani v5.0  |  Hignett & McAtamney (2000)  |  "
            f"AI tabanli aci tahmini +/-3-5 derece dogruluk payi icerir",
            size=7, color=colors.HexColor('#94a3b8')))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ════════════════════════════════════════════════════════
# CSS — AÇIK TEMA / KURUMSAL
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
.main .block-container { padding-top: 1.2rem !important; max-width: 1200px !important; }
.app-header {
    background:#fff; border:1px solid #e2e8f0; border-left:4px solid #1e3a5f;
    border-radius:8px; padding:16px 22px; margin-bottom:18px;
    display:flex; align-items:center; gap:14px;
}
.app-header-icon {
    width:42px; height:42px; background:#1e3a5f; border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    font-size:20px; flex-shrink:0;
}
.app-header h1 { margin:0; font-size:17px; font-weight:700; color:#0f172a; }
.app-header p { margin:3px 0 0; font-size:11px; color:#64748b; }
.metric-row {
    display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:18px;
}
.metric-box {
    background:#fff; border:1px solid #e2e8f0; border-radius:8px;
    padding:14px; text-align:center;
}
.metric-box .val {
    font-size:30px; font-weight:700; line-height:1;
    font-family:'IBM Plex Mono',monospace;
}
.metric-box .lbl { font-size:10px; color:#64748b; margin-top:4px; font-weight:500; }
.risk-badge {
    display:inline-block; padding:2px 9px; border-radius:4px;
    font-size:11px; font-weight:600; margin-top:5px;
}
.foto-header {
    padding:10px 14px; border-radius:8px 8px 0 0;
    display:flex; justify-content:space-between; align-items:center;
    font-size:13px; font-weight:600; color:white; margin-bottom:0;
}
.seg-bar-wrap { margin-bottom:9px; }
.seg-bar-label {
    display:flex; justify-content:space-between;
    font-size:12px; color:#374151; margin-bottom:2px;
}
.seg-bar-track {
    background:#f1f5f9; border-radius:3px; height:7px; overflow:hidden;
}
.seg-bar-fill { height:100%; border-radius:3px; }
.skor-ozet {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px;
    padding:10px 12px; font-size:12px; margin-top:8px;
}
.info-box {
    background:#eff6ff; border:1px solid #bfdbfe; border-left:3px solid #2563eb;
    border-radius:6px; padding:9px 13px; font-size:12px; color:#1e40af; margin:6px 0;
}
.warn-box {
    background:#fffbeb; border:1px solid #fde68a; border-left:3px solid #d97706;
    border-radius:6px; padding:9px 13px; font-size:12px; color:#92400e; margin:6px 0;
}
.error-box {
    background:#fef2f2; border:1px solid #fecaca; border-left:3px solid #dc2626;
    border-radius:6px; padding:9px 13px; font-size:12px; color:#991b1b; margin:6px 0;
}
.app-footer {
    text-align:center; font-size:10px; color:#94a3b8;
    padding:18px 0 8px; border-top:1px solid #e2e8f0; margin-top:28px;
=======
    key_lms = [LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP]
    angles.confidence = np.mean([vis(l) for l in key_lms])

    nose = pt(LM.NOSE)
    angles.neck_flexion = calc_angle_vertical(nose, mid_shoulder)

    left_ear = pt(LM.LEFT_EAR)
    right_ear = pt(LM.RIGHT_EAR)
    ear_dy = abs(left_ear[1] - right_ear[1])
    ear_dx = abs(left_ear[0] - right_ear[0])
    if ear_dx > 0:
        angles.neck_side_bend = math.degrees(math.atan2(ear_dy, ear_dx))

    angles.trunk_flexion = calc_angle_vertical(mid_shoulder, mid_hip)
    lateral_offset = abs(mid_shoulder[0] - mid_hip[0])
    trunk_height = abs(mid_shoulder[1] - mid_hip[1])
    if trunk_height > 0:
        angles.trunk_side_bend = math.degrees(math.atan2(lateral_offset, trunk_height))

    shoulder_width = abs(pt(LM.LEFT_SHOULDER)[0] - pt(LM.RIGHT_SHOULDER)[0])
    hip_width = abs(pt(LM.LEFT_HIP)[0] - pt(LM.RIGHT_HIP)[0])
    if hip_width > 0:
        twist_ratio = shoulder_width / hip_width
        angles.trunk_twist = twist_ratio < 0.7 or twist_ratio > 1.4

    angles.knee_flexion_left = 180 - calc_angle_3p(pt(LM.LEFT_HIP), pt(LM.LEFT_KNEE), pt(LM.LEFT_ANKLE))
    angles.knee_flexion_right = 180 - calc_angle_3p(pt(LM.RIGHT_HIP), pt(LM.RIGHT_KNEE), pt(LM.RIGHT_ANKLE))
    angles.bilateral_support = (vis(LM.LEFT_ANKLE) > 0.3 and vis(LM.RIGHT_ANKLE) > 0.3)

    ua_left = calc_angle_3p(pt(LM.LEFT_HIP), pt(LM.LEFT_SHOULDER), pt(LM.LEFT_ELBOW))
    ua_right = calc_angle_3p(pt(LM.RIGHT_HIP), pt(LM.RIGHT_SHOULDER), pt(LM.RIGHT_ELBOW))
    angles.upper_arm_angle = max(ua_left, ua_right)

    ref_dist = abs(pt(LM.LEFT_HIP)[1] - pt(LM.LEFT_SHOULDER)[1])
    left_se = abs(pt(LM.LEFT_SHOULDER)[1] - pt(LM.LEFT_EAR)[1])
    right_se = abs(pt(LM.RIGHT_SHOULDER)[1] - pt(LM.RIGHT_EAR)[1])
    if ref_dist > 0:
        angles.shoulder_raised = min(left_se, right_se) / ref_dist < 0.3

    left_abd = abs(pt(LM.LEFT_ELBOW)[0] - pt(LM.LEFT_SHOULDER)[0])
    right_abd = abs(pt(LM.RIGHT_ELBOW)[0] - pt(LM.RIGHT_SHOULDER)[0])
    if shoulder_width > 0:
        angles.arm_abducted = max(left_abd, right_abd) / shoulder_width > 0.8

    la_left = calc_angle_3p(pt(LM.LEFT_SHOULDER), pt(LM.LEFT_ELBOW), pt(LM.LEFT_WRIST))
    la_right = calc_angle_3p(pt(LM.RIGHT_SHOULDER), pt(LM.RIGHT_ELBOW), pt(LM.RIGHT_WRIST))
    angles.lower_arm_angle = min(la_left, la_right)

    w_left = abs(180 - calc_angle_3p(pt(LM.LEFT_ELBOW), pt(LM.LEFT_WRIST), pt(LM.LEFT_INDEX)))
    w_right = abs(180 - calc_angle_3p(pt(LM.RIGHT_ELBOW), pt(LM.RIGHT_WRIST), pt(LM.RIGHT_INDEX)))
    angles.wrist_angle = max(w_left, w_right)

    wrist_y_diff = abs(pt(LM.LEFT_WRIST)[1] - pt(LM.RIGHT_WRIST)[1])
    if ref_dist > 0:
        angles.wrist_twist = wrist_y_diff / ref_dist > 0.15

    return angles

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REBA SKORLAMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_load_score(kg: float, shock: bool) -> int:
    if kg < 5:
        s = 0
    elif kg <= 10:
        s = 1
    else:
        s = 2
    if shock:
        s += 1
    return s

def score_reba(angles: BodyAngles, load_score: int, coupling: int, activity: int) -> REBAResult:
    r = REBAResult()
    r.angles = angles
    r.load_score = load_score
    r.coupling_score = coupling
    r.activity_score = activity

    # BOYUN
    if angles.neck_flexion <= 20:
        r.neck_score = 1
    elif angles.neck_flexion <= 40:
        r.neck_score = 2
    else:
        r.neck_score = 3
    if angles.neck_side_bend > 15:
        r.neck_score += 1
    r.neck_score = min(r.neck_score, 6)

    # GÖVDE
    if angles.trunk_flexion <= 5:
        r.trunk_score = 1
    elif angles.trunk_flexion <= 20:
        r.trunk_score = 2
    elif angles.trunk_flexion <= 60:
        r.trunk_score = 3
    else:
        r.trunk_score = 4
    if angles.trunk_side_bend > 10:
        r.trunk_score += 1
    if angles.trunk_twist:
        r.trunk_score += 1
    r.trunk_score = min(r.trunk_score, 5)

    # BACAK
    r.leg_score = 1 if angles.bilateral_support else 2
    knee = max(angles.knee_flexion_left, angles.knee_flexion_right)
    if 30 <= knee < 60:
        r.leg_score += 1
    elif knee >= 60:
        r.leg_score += 2
    r.leg_score = min(r.leg_score, 4)

    # TABLO A
    t = min(r.trunk_score - 1, 4)
    n = min(r.neck_score - 1, 2)
    l = min(r.leg_score - 1, 3)
    r.table_a_score = TABLE_A[t][n][l]
    r.score_a = r.table_a_score + r.load_score

    # ÜST KOL
    if angles.upper_arm_angle <= 20:
        r.upper_arm_score = 1
    elif angles.upper_arm_angle <= 45:
        r.upper_arm_score = 2
    elif angles.upper_arm_angle <= 90:
        r.upper_arm_score = 3
    else:
        r.upper_arm_score = 4
    if angles.shoulder_raised:
        r.upper_arm_score += 1
    if angles.arm_abducted:
        r.upper_arm_score += 1
    r.upper_arm_score = max(1, min(r.upper_arm_score, 6))

    # ALT KOL
    r.lower_arm_score = 1 if 60 <= angles.lower_arm_angle <= 100 else 2

    # BİLEK
    r.wrist_score = 1 if angles.wrist_angle <= 15 else 2
    if angles.wrist_twist:
        r.wrist_score += 1
    r.wrist_score = min(r.wrist_score, 3)

    # TABLO B
    u = min(r.upper_arm_score - 1, 5)
    la = min(r.lower_arm_score - 1, 1)
    w = min(r.wrist_score - 1, 2)
    r.table_b_score = TABLE_B[u][la][w]
    r.score_b = r.table_b_score + r.coupling_score

    # TABLO C
    ca = min(r.score_a - 1, 11)
    cb = min(r.score_b - 1, 11)
    r.score_c = TABLE_C[ca][cb]
    r.final_score = min(r.score_c + r.activity_score, 15)

    # RİSK SEVİYESİ
    s = r.final_score
    if s == 1:
        r.risk_level = "Önemsiz Risk"
        r.color = "#22c55e"
        r.action = "Herhangi bir önlem gerekmez"
    elif s <= 3:
        r.risk_level = "Düşük Risk"
        r.color = "#86efac"
        r.action = "Eğer gerekli ise iyileştirme yap"
    elif s <= 7:
        r.risk_level = "Orta Seviyeli Risk"
        r.color = "#fbbf24"
        r.action = "Daha ayrıntılı incele ve ileride değişiklik yap"
    elif s <= 10:
        r.risk_level = "Yüksek Risk"
        r.color = "#f97316"
        r.action = "Çözüm için araştırma yap ve aksiyon al"
    else:
        r.risk_level = "Çok Yüksek Risk"
        r.color = "#ef4444"
        r.action = "Süreç çalışmaya uygun değil, derhal revize et"

    return r

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VİDEO İŞLEME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_video_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps if fps > 0 else 0

def extract_frames(path, frames_per_second=3, max_duration=15.0):
    """Her saniyeden en fazla 3 frame çıkar, max 15 sn."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = min(total_frames / fps, max_duration)

    # Her saniyeden kaç frame alacağız
    frame_interval = max(1, int(fps / frames_per_second))

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = frame_idx / fps
        if current_time > max_duration:
            break
        if frame_idx % frame_interval == 0:
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))
            frames.append((frame, round(current_time, 2)))
        frame_idx += 1

    cap.release()
    return frames

def extract_image_frame(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    h, w = img.shape[:2]
    if w > 1280:
        scale = 1280 / w
        img = cv2.resize(img, (1280, int(h * scale)))
    return [(img, 0.0)]

def draw_skeleton(image, landmarks, score, frame_time):
    h, w = image.shape[:2]
    overlay = image.copy()

    if score <= 3:
        color_bgr = (34, 197, 94)
    elif score <= 7:
        color_bgr = (36, 191, 251)
    elif score <= 10:
        color_bgr = (22, 115, 249)
    else:
        color_bgr = (68, 68, 239)

    for conn in POSE_CONNECTIONS:
        s_lm = landmarks[conn[0]]
        e_lm = landmarks[conn[1]]
        if s_lm.visibility > 0.3 and e_lm.visibility > 0.3:
            sp = (int(s_lm.x * w), int(s_lm.y * h))
            ep = (int(e_lm.x * w), int(e_lm.y * h))
            cv2.line(overlay, sp, ep, color_bgr, 2, cv2.LINE_AA)

    for lm in landmarks:
        if lm.visibility > 0.3:
            p = (int(lm.x * w), int(lm.y * h))
            cv2.circle(overlay, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, p, 4, color_bgr, 1, cv2.LINE_AA)

    cv2.rectangle(overlay, (8, 8), (180, 55), (0, 0, 0), -1)
    cv2.rectangle(overlay, (8, 8), (180, 55), color_bgr, 2)
    cv2.putText(overlay, f"REBA: {score}", (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2, cv2.LINE_AA)
    cv2.putText(overlay, f"t={frame_time:.2f}s", (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.85, image, 0.15, 0)

def process_frames(frames, load_score, coupling, activity, progress_bar, status_text):
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    results = []
    annotated = []

    for i, (frame, t) in enumerate(frames):
        progress_bar.progress((i + 1) / len(frames))
        status_text.text(f"Kare {i+1}/{len(frames)} analiz ediliyor... (t={t:.2f}s)")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)

        if pose_result.pose_landmarks:
            lms = pose_result.pose_landmarks.landmark
            h, w = frame.shape[:2]
            angles = calculate_body_angles(lms, w, h)
            reba = score_reba(angles, load_score, coupling, activity)
            reba.frame_time = t
            results.append(reba)
            ann = draw_skeleton(frame.copy(), lms, reba.final_score, t)
            annotated.append((ann, reba))
        else:
            annotated.append((frame.copy(), None))

    pose.close()
    return results, annotated

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PDF RAPOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_pdf_report(form_info, results, worst, avg_score, annotated_img=None):
    """Basit PDF raporu - reportlab kullanır."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import base64

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        story = []

        # Başlık stili
        title_style = ParagraphStyle('Title', parent=styles['Title'],
                                     fontSize=16, spaceAfter=6,
                                     textColor=colors.HexColor('#1e3a5f'))
        sub_style = ParagraphStyle('Sub', parent=styles['Normal'],
                                   fontSize=10, textColor=colors.grey)
        h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                                  fontSize=12, spaceBefore=12, spaceAfter=4,
                                  textColor=colors.HexColor('#1e3a5f'))
        normal = styles['Normal']

        # Başlık
        story.append(Paragraph("REBA Ergonomi Risk Analiz Raporu", title_style))
        story.append(Paragraph("Rapid Entire Body Assessment | AI Destekli Analiz", sub_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1e3a5f')))
        story.append(Spacer(1, 0.3*cm))

        # Form Bilgileri
        story.append(Paragraph("Form Bilgileri", h2_style))
        form_data = [
            ["Bölüm", form_info.get('bolum', '-'),
             "Tarih", form_info.get('tarih', '-')],
            ["İş İstasyonu", form_info.get('is_istasyonu', '-'),
             "Analist", form_info.get('analist', '-')],
            ["İş Adımı / Kodu", form_info.get('is_adimi', '-'),
             "Dosya", form_info.get('dosya_adi', '-')],
        ]
        t = Table(form_data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#475569')),
            ('TEXTCOLOR', (2,0), (2,-1), colors.HexColor('#475569')),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))

        # Sonuç Özeti
        story.append(Paragraph("Analiz Sonucu", h2_style))

        risk_color_hex = worst.color.replace('#', '')
        r_int = int(risk_color_hex[0:2], 16) / 255
        g_int = int(risk_color_hex[2:4], 16) / 255
        b_int = int(risk_color_hex[4:6], 16) / 255
        risk_col = colors.Color(r_int, g_int, b_int)

        summary_data = [
            ["En Yüksek REBA Skoru", f"{worst.final_score} / 15",
             "Risk Seviyesi", worst.risk_level],
            ["Ortalama REBA Skoru", f"{avg_score:.1f}",
             "Önerilen Aksiyon", worst.action],
            ["Analiz Edilen Kare", f"{len(results)}",
             "En Riskli Zaman", f"{worst.frame_time:.2f}s"],
        ]
        t2 = Table(summary_data, colWidths=[4*cm, 3.5*cm, 4*cm, 5.5*cm])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#475569')),
            ('TEXTCOLOR', (2,0), (2,-1), colors.HexColor('#475569')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0,0), (-1,-1), 4),
            ('BACKGROUND', (1,0), (1,0), risk_col),
            ('TEXTCOLOR', (1,0), (1,0), colors.white),
            ('FONTNAME', (1,0), (1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (1,0), (1,0), 14),
            ('ALIGN', (1,0), (1,0), 'CENTER'),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.3*cm))

        # Manuel Girdi Özeti
        story.append(Paragraph("Manuel Girdi Parametreleri", h2_style))
        manual_data = [
            ["Yük (kg)", f"{form_info.get('yuk_kg', 0)} kg",
             "Yük Skoru", f"+{form_info.get('load_score', 0)}"],
            ["Ani/Hızlı Kuvvet", "Evet" if form_info.get('shock') else "Hayır",
             "Tutma Skoru", f"+{form_info.get('coupling', 0)}"],
            ["Aktivite Skoru", f"+{form_info.get('activity', 0)}",
             "Tutma Tipi", form_info.get('coupling_label', '-')],
        ]
        t3 = Table(manual_data, colWidths=[3.5*cm, 3.5*cm, 3.5*cm, 6.5*cm])
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#475569')),
            ('TEXTCOLOR', (2,0), (2,-1), colors.HexColor('#475569')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t3)
        story.append(Spacer(1, 0.3*cm))

        # Kare Bazlı Sonuçlar
        story.append(Paragraph("Kare Bazlı REBA Skorları", h2_style))
        frame_header = ["Süre (s)", "REBA Skoru", "Risk Seviyesi",
                        "Boyun", "Gövde", "Bacak", "Üst Kol", "Alt Kol", "Bilek"]
        frame_rows = [frame_header]
        for r in sorted(results, key=lambda x: x.frame_time):
            frame_rows.append([
                f"{r.frame_time:.2f}",
                str(r.final_score),
                r.risk_level,
                str(r.neck_score),
                str(r.trunk_score),
                str(r.leg_score),
                str(r.upper_arm_score),
                str(r.lower_arm_score),
                str(r.wrist_score),
            ])

        col_widths = [1.5*cm, 2*cm, 4*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.8*cm, 1.8*cm, 1.5*cm]
        t4 = Table(frame_rows, colWidths=col_widths)
        table_style = [
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 3),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ]
        # En yüksek skoru kırmızı yap
        for i, r in enumerate(sorted(results, key=lambda x: x.frame_time), 1):
            if r.final_score >= 11:
                table_style.append(('BACKGROUND', (1,i), (1,i), colors.HexColor('#ef4444')))
                table_style.append(('TEXTCOLOR', (1,i), (1,i), colors.white))
            elif r.final_score >= 8:
                table_style.append(('BACKGROUND', (1,i), (1,i), colors.HexColor('#f97316')))
                table_style.append(('TEXTCOLOR', (1,i), (1,i), colors.white))
            elif r.final_score >= 4:
                table_style.append(('BACKGROUND', (1,i), (1,i), colors.HexColor('#fbbf24')))
        t4.setStyle(TableStyle(table_style))
        story.append(t4)
        story.append(Spacer(1, 0.5*cm))

        # Risk Açıklama Tablosu
        story.append(Paragraph("Risk Skalası", h2_style))
        risk_scale = [
            ["Skor", "Risk Seviyesi", "Önlem"],
            ["1", "Önemsiz Risk", "Herhangi bir önlem gerekmez"],
            ["2-3", "Düşük Risk", "Eğer gerekli ise iyileştirme yap"],
            ["4-7", "Orta Seviyeli Risk", "Daha ayrıntılı incele ve ileride değişiklik yap"],
            ["8-10", "Yüksek Risk", "Çözüm için araştırma yap ve aksiyon al"],
            ["11+", "Çok Yüksek Risk", "Süreç çalışmaya uygun değil, derhal revize et"],
        ]
        t5 = Table(risk_scale, colWidths=[2*cm, 4*cm, 11*cm])
        t5.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0,0), (-1,-1), 3),
            ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#dcfce7')),
            ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#f0fdf4')),
            ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#fef9c3')),
            ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#ffedd5')),
            ('BACKGROUND', (0,5), (-1,5), colors.HexColor('#fee2e2')),
        ]))
        story.append(t5)

        # Footer
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(Paragraph(
            f"Bu rapor REBA Ergonomi Analiz Ajanı v4.0 tarafından oluşturulmuştur. "
            f"Referans: Hignett & McAtamney (2000). Applied Ergonomics, 31, 201-205. "
            f"Oluşturma tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            ParagraphStyle('footer', parent=styles['Normal'], fontSize=7, textColor=colors.grey)
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except ImportError:
        return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
.stApp { background-color: #f8fafc; }

.reba-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    padding: 20px 28px; border-radius: 12px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 16px;
}
.reba-header h1 {
    color: white; font-size: 24px; font-weight: 800;
    margin: 0; letter-spacing: -0.5px;
}
.reba-header p {
    color: rgba(255,255,255,0.75); font-size: 12px; margin: 4px 0 0 0;
}

.score-big {
    font-size: 72px; font-weight: 900; line-height: 1;
    text-align: center; padding: 16px 0;
}
.metric-card {
    background: white; border-radius: 10px; padding: 14px 16px;
    border: 1px solid #e2e8f0; margin-bottom: 8px;
}
.metric-label {
    font-size: 11px; color: #64748b; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.metric-value {
    font-size: 22px; font-weight: 800; color: #1e293b; margin-top: 2px;
}
.section-title {
    font-size: 13px; font-weight: 700; color: #1e3a5f;
    text-transform: uppercase; letter-spacing: 0.08em;
    border-bottom: 2px solid #1e3a5f; padding-bottom: 4px;
    margin: 16px 0 10px 0;
}
.warning-box {
    background: #fef3c7; border: 1px solid #fbbf24; border-radius: 8px;
    padding: 12px 16px; font-size: 13px; color: #92400e;
}
.info-box {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
    padding: 12px 16px; font-size: 13px; color: #1e40af;
}
.frame-badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 700; margin: 2px;
>>>>>>> ddded5f36dedc372c567f0ac5bae861d227b2eb3
}
</style>
""", unsafe_allow_html=True)

<<<<<<< HEAD
# ════════════════════════════════════════════════════════
# BAŞLIK
# ════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
    <div class="app-header-icon">🦺</div>
    <div>
        <h1>REBA Ergonomi Risk Analiz Sistemi</h1>
        <p>Rapid Entire Body Assessment &nbsp;·&nbsp; Çoklu Fotoğraf Analizi &nbsp;·&nbsp;
           AI Destekli Postür Değerlendirmesi &nbsp;·&nbsp; v5.0</p>
=======
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BAŞLIK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<div class="reba-header">
    <div style="font-size:40px">🦺</div>
    <div>
        <h1>REBA Ergonomi Risk Analiz Ajanı</h1>
        <p>Rapid Entire Body Assessment | MediaPipe Pose AI | ISG Uzmanı Aracı | v4.0</p>
>>>>>>> ddded5f36dedc372c567f0ac5bae861d227b2eb3
    </div>
</div>
""", unsafe_allow_html=True)

<<<<<<< HEAD
# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📋 Form Bilgileri")
    bolum = st.text_input("Bölüm", placeholder="örn: Formasyon")
    is_istasyonu = st.text_input("İş İstasyonu", placeholder="örn: Final 1 Sonu")
    is_adimi = st.text_input("İş Adımı / Kodu", placeholder="örn: Paletten akü alma")
    analist = st.text_input("Analist", placeholder="Ad Soyad")
    tarih = st.date_input("Tarih", value=date.today())

    st.markdown("### ⚖️ 4. Yük Analizi")
    yuk_kg = st.number_input("Yük (kg)", min_value=0.0, max_value=200.0,
                              value=0.0, step=0.5, format="%.1f")
    shock = st.checkbox("Ani / hızlı kuvvet uygulaması (+1)")

    if yuk_kg < 5: yuk_base, yuk_aci = 0, f"{yuk_kg:.1f} kg → +0 (5 kg altı)"
    elif yuk_kg <= 10: yuk_base, yuk_aci = 1, f"{yuk_kg:.1f} kg → +1 (5–10 kg)"
    else: yuk_base, yuk_aci = 2, f"{yuk_kg:.1f} kg → +2 (10 kg üstü)"
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
        "Uygun tutma yeri mevcut (+0)": (0, "İdeal kavrama"),
        "Kabul edilebilir ama ideal değil (+1)": (1, "Yeterli kavrama"),
        "Tutulabilir ama uygun değil (+2)": (2, "Zor kavrama"),
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
            Aktivite: <strong>+{aktivite_val}</strong> &nbsp;
            <span style="color:#64748b">({', '.join(akt_aci)})</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px;color:#94a3b8;line-height:1.9">
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
    <div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:8px">📁 Fotoğraf Yükle</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Bir veya birden fazla fotoğraf yükleyin. Her fotoğraf için ayrı REBA analizi yapılır.<br>
    Desteklenen formatlar: <strong>JPG, PNG, WEBP</strong>
    </div>
    """, unsafe_allow_html=True)
    yuklenen = st.file_uploader(
        "Fotoğraf",
        type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
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
    <div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:8px">ℹ️ Parametreler</div>
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
        <td style="padding:6px 4px;font-weight:600">+{aktivite_val}{(' — '+', '.join(akt_aci)) if akt_aci else ''}</td>
    </tr>
    <tr>
        <td style="padding:6px 4px;color:#64748b">Fotoğraf</td>
        <td style="padding:6px 4px;font-weight:600">{len(yuklenen) if yuklenen else 0} adet</td>
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
                          use_container_width=True,
                          type="primary")
with col_uyari:
    if not form_tamam:
        eksikler = []
        if not (bolum or is_istasyonu or is_adimi):
            eksikler.append("form bilgisi")
        if not yuklenen:
            eksikler.append("fotoğraf")
        st.markdown(f"""
        <div class="warn-box" style="margin-top:4px">
        ⚠️ Analiz için gerekli: <strong>{' ve '.join(eksikler)}</strong>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# ANALİZ
# ════════════════════════════════════════════════════════

if calistir and form_tamam:
    st.markdown("---")
    pb = st.progress(0)
    durum = st.empty()
    foto_sonuclari: List[FotoSonuc] = []

    for i, dosya in enumerate(yuklenen):
        durum.text(f"Analiz ediliyor: {dosya.name} ({i+1}/{len(yuklenen)})")
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
                    skor = reba_skorla(a_obj, yuk_skoru_val, tutma_val, aktivite_val)
                    fs.skor = skor
                    fs.overlay_img = overlay_ciz(img.copy(), lms, skor)
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
        ❌ Hiçbir fotoğrafta kişi tespit edilemedi. Fotoğrafların tam vücut gösterir,
        net ve iyi aydınlatılmış olduğundan emin olun.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Metrikler
    skorlar = [s.skor.final_skor for s in gecerli]
    ort = sum(skorlar) / len(skorlar)
    en_yuk = max(skorlar)
    en_dus = min(skorlar)

    def risk_info(skor):
        if skor <= 1: return "Önemsiz", "#16a34a"
        elif skor <= 3: return "Düşük", "#65a30d"
        elif skor <= 7: return "Orta", "#d97706"
        elif skor <= 10: return "Yüksek", "#dc2626"
        else: return "Çok Yüksek", "#7c3aed"

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
            <span class="risk-badge" style="background:{ort_c}18;color:{ort_c};border:1px solid {ort_c}40">{ort_r}</span>
        </div>
        <div class="metric-box">
            <div class="val" style="color:{yuk_c}">{en_yuk}</div>
            <div class="lbl">En Yüksek REBA</div>
            <span class="risk-badge" style="background:{yuk_c}18;color:{yuk_c};border:1px solid {yuk_c}40">{yuk_r}</span>
        </div>
        <div class="metric-box">
            <div class="val" style="color:#475569">{en_dus}</div>
            <div class="lbl">En Düşük REBA</div>
        </div>
        <div class="metric-box">
            <div class="val" style="color:#0f172a">{len(gecerli)}</div>
            <div class="lbl">Analiz Edilen</div>
            {f'<div style="font-size:10px;color:#dc2626;margin-top:2px">{len(foto_sonuclari)-len(gecerli)} başarısız</div>' if len(foto_sonuclari) > len(gecerli) else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Başarısız uyarıları
    for fs in foto_sonuclari:
        if fs.hata:
            st.markdown(f"""
            <div class="warn-box">
            ⚠️ <strong>{fs.dosya_adi}</strong>: {fs.hata}
            </div>
            """, unsafe_allow_html=True)

    # ── FOTO DETAYLARI ──
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;
                letter-spacing:0.1em;margin:16px 0 12px">
        🔍 Fotoğraf Bazlı Analiz
    </div>
    """, unsafe_allow_html=True)

    for fs in gecerli:
        s = fs.skor
        a = s.acılar
        _, renk = risk_info(s.final_skor)

        st.markdown(f"""
        <div class="foto-header" style="background:{renk}">
            <span>📷 Fotoğraf {fs.idx} &nbsp;·&nbsp; {fs.dosya_adi}</span>
            <span>REBA {s.final_skor}/15 &nbsp;·&nbsp; {s.risk_seviyesi}</span>
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
                 f"{a.boyun_flexion:.0f}°" +
                 ("+YanEğ" if a.boyun_yan_egim > 15 else "") +
                 ("+Dön" if a.boyun_donus else "")),
                ("Gövde", s.govde_skoru, 5,
                 f"{a.govde_flexion:.0f}°" +
                 ("+YanEğ" if a.govde_yan_egim > 10 else "") +
                 ("+Dön" if a.govde_donus else "")),
                ("Bacak/Diz", s.bacak_skoru, 4,
                 f"Diz {max(a.diz_flexion_sol, a.diz_flexion_sag):.0f}°"),
                ("Üst Kol", s.ust_kol_skoru, 6,
                 f"{a.ust_kol_aci:.0f}°" +
                 ("+OmKalK" if a.omuz_kalkmis else "") +
                 ("+Abd" if a.kol_abdukte else "")),
                ("Alt Kol", s.alt_kol_skoru, 2, f"{a.alt_kol_aci:.0f}°"),
                ("Bilek", s.bilek_skoru, 3,
                 f"{a.bilek_aci:.0f}°" + ("+Dön" if a.bilek_donus else "")),
            ]

            for seg_ad, seg_val, seg_max, seg_aci in segs:
                pct = seg_val / seg_max
                bc = "#16a34a" if pct <= 0.4 else "#d97706" if pct <= 0.65 else "#dc2626"
                st.markdown(f"""
                <div class="seg-bar-wrap">
                    <div class="seg-bar-label">
                        <span>{seg_ad}
                            <span style="color:#94a3b8;font-size:11px">&nbsp;{seg_aci}</span>
                        </span>
                        <span style="font-weight:700;color:{bc};font-family:'IBM Plex Mono',monospace">
                            {seg_val}/{seg_max}
                        </span>
                    </div>
                    <div class="seg-bar-track">
                        <div class="seg-bar-fill" style="width:{pct*100:.0f}%;background:{bc}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="skor-ozet">
                <div style="display:flex;gap:14px;flex-wrap:wrap;color:#374151">
                    <span>Tablo A: <strong>{s.tablo_a}</strong> + Yük <strong>+{s.yuk_skoru}</strong>
                          = Skor A: <strong style="color:#1d4ed8">{s.skor_a}</strong></span>
                    <span>Tablo B: <strong>{s.tablo_b}</strong> + Tutma <strong>+{s.tutma_skoru}</strong>
                          = Skor B: <strong style="color:#1d4ed8">{s.skor_b}</strong></span>
                </div>
                <div style="margin-top:7px;border-top:1px solid #e2e8f0;padding-top:7px">
                    Tablo C: <strong>{s.skor_c}</strong> + Aktivite: <strong>+{s.aktivite_skoru}</strong>
                    &nbsp;=&nbsp;
                    <strong style="font-size:15px;color:{renk}">REBA {s.final_skor}</strong>
                    &nbsp;&nbsp;
                    <span style="color:{renk};font-weight:600">{s.risk_seviyesi}</span>
                    <br>
                    <span style="color:#64748b;font-size:11px">
                        → {s.aksiyon} &nbsp;·&nbsp; AI Güven: {a.guven:.0%}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="border-bottom:1px solid #e2e8f0;margin:8px 0 18px"></div>',
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
        'yuk_kg': yuk_kg, 'shock': shock,
        'yuk_skoru': yuk_skoru_val,
        'yuk_aciklama': yuk_aci + (" + Ani kuvvet +1" if shock else ""),
        'tutma': tutma_val, 'tutma_label': tutma_label,
        'aktivite': aktivite_val, 'aktivite_aciklama': akt_aciklama_str,
    }

    col_p, col_b = st.columns([1, 3])
    with col_p:
        with st.spinner("PDF hazırlanıyor..."):
            pdf_bytes = pdf_olustur(form_bilgi, foto_sonuclari)
        if pdf_bytes:
            ad = f"REBA_{(bolum or is_istasyonu or 'analiz').replace(' ','_')}_{tarih}.pdf"
            st.download_button("⬇️  PDF Raporu İndir",
                               data=pdf_bytes, file_name=ad,
                               mime="application/pdf",
                               use_container_width=True)
        else:
            st.error("PDF oluşturulamadı.")

st.markdown("""
<div class="app-footer">
    REBA Ergonomi Risk Analiz Sistemi v5.0 &nbsp;·&nbsp;
    Hignett & McAtamney (2000), Applied Ergonomics 31(2), 201–205 &nbsp;·&nbsp;
    MediaPipe Pose (Google LLC) &nbsp;·&nbsp;
    AI tabanlı açı tahmini ±3–5° doğruluk payı içerir — profesyonel değerlendirmenin yerini tutmaz
=======
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR — FORM BİLGİLERİ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("### 📋 Form Bilgileri")

    bolum = st.text_input("Bölüm", placeholder="örn: Formasyon")
    is_istasyonu = st.text_input("İş İstasyonu", placeholder="örn: Final 1 Sonu")
    is_adimi = st.text_input("İş Adımı ve Kodu", placeholder="örn: Paletten akü alma")
    analist = st.text_input("Analist Adı", placeholder="Ad Soyad")
    tarih = st.date_input("Tarih", value=datetime.today())

    st.markdown("---")
    st.markdown("### ⚖️ 4. Yük Analizi")
    yuk_kg = st.number_input("Taşınan/Kaldırılan Yük (kg)", min_value=0.0, max_value=100.0,
                              value=0.0, step=0.5, format="%.1f")
    shock = st.checkbox("Ani veya hızlı kuvvet uygulanıyor (+1)")

    # Otomatik yük skoru hesapla
    if yuk_kg < 5:
        load_base = 0
        load_label = f"{yuk_kg} kg → +0 (5 kg altı)"
    elif yuk_kg <= 10:
        load_base = 1
        load_label = f"{yuk_kg} kg → +1 (5-10 kg arası)"
    else:
        load_base = 2
        load_label = f"{yuk_kg} kg → +2 (10 kg üstü)"

    load_score_val = load_base + (1 if shock else 0)
    st.info(f"Yük Skoru: **+{load_score_val}**\n\n{load_label}" +
            (" + Ani kuvvet +1" if shock else ""))

    st.markdown("---")
    st.markdown("### 🤜 8. Materyali Tutma Analizi")
    coupling_options = {
        "Uygun tutma yeri mevcut (+0)": 0,
        "Kabul edilebilir fakat ideal olmayan tutma yeri (+1)": 1,
        "Tutulabilecek fakat kabul edilebilir olmayan (+2)": 2,
        "Tutma yeri yok / kaldırmaya müsait değil (+3)": 3,
    }
    coupling_label = st.selectbox("Tutma Kalitesi", list(coupling_options.keys()))
    coupling_val = coupling_options[coupling_label]

    st.markdown("---")
    st.markdown("### 🏃 9. Aktivite Analizi")
    act1 = st.checkbox("Uzuv statik pozisyonda 1 dk+ tutulyor (+1)")
    act2 = st.checkbox("Dakikada 4'ten fazla tekrarlı hareket (+1)")
    act3 = st.checkbox("Stabil olmayan zemin / hızlı vücut değişimi (+1)")
    activity_val = int(act1) + int(act2) + int(act3)
    if activity_val > 0:
        st.info(f"Aktivite Skoru: **+{activity_val}**")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px; color:#94a3b8; line-height:1.7">
    <b>REBA Risk Skalası</b><br>
    🟢 1 → Önemsiz<br>
    🟡 2-3 → Düşük<br>
    🟠 4-7 → Orta<br>
    🔴 8-10 → Yüksek<br>
    ⛔ 11+ → Çok Yüksek
    </div>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANA ALAN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

col_upload, col_info = st.columns([3, 2])

with col_upload:
    st.markdown('<div class="section-title">📁 Medya Yükle</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    📹 <b>Video:</b> MP4, MOV, WEBM, AVI — <b>Maksimum 15 saniye</b><br>
    📸 <b>Görüntü:</b> JPG, PNG, WEBP<br>
    Her saniyeden <b>3 kare</b> analiz edilir. Video 15 saniyeyi geçerse reddedilir.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Dosya seçin",
        type=["mp4", "mov", "m4v", "webm", "avi", "jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

with col_info:
    st.markdown('<div class="section-title">ℹ️ Analiz Parametreleri</div>', unsafe_allow_html=True)
    params_ok = True
    issues = []

    if not bolum and not is_istasyonu and not is_adimi:
        issues.append("Form bilgileri eksik (Bölüm, İş İstasyonu, İş Adımı)")
        params_ok = False

    if uploaded is None:
        issues.append("Video veya görüntü yüklenmedi")
        params_ok = False

    # Parametreleri göster
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Yük Skoru</div>
        <div class="metric-value" style="color:{'#ef4444' if load_score_val >= 2 else '#f97316' if load_score_val == 1 else '#22c55e'}">
            +{load_score_val} &nbsp;<span style="font-size:14px;color:#64748b">({yuk_kg} kg)</span>
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Tutma Skoru</div>
        <div class="metric-value" style="color:{'#ef4444' if coupling_val >= 2 else '#f97316' if coupling_val == 1 else '#22c55e'}">
            +{coupling_val}
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Aktivite Skoru</div>
        <div class="metric-value" style="color:{'#ef4444' if activity_val == 3 else '#f97316' if activity_val >= 1 else '#22c55e'}">
            +{activity_val}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if issues:
        for issue in issues:
            st.markdown(f'<div class="warning-box">⚠️ {issue}</div>', unsafe_allow_html=True)

# Analizi Başlat Butonu
st.markdown("---")

# Form eksiği varsa uyar
form_complete = bool(bolum or is_istasyonu or is_adimi) and uploaded is not None

col_btn, col_note = st.columns([1, 3])
with col_btn:
    run_btn = st.button(
        "🚀 ANALİZİ BAŞLAT",
        disabled=not form_complete,
        use_container_width=True,
        type="primary"
    )

with col_note:
    if not form_complete:
        st.markdown("""
        <div class="warning-box" style="margin-top:6px">
        ⚠️ Analizi başlatmak için: En az bir form bilgisi girin (Bölüm / İş İstasyonu / İş Adımı)
        ve bir video/görüntü yükleyin.
        </div>
        """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALİZ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if run_btn and form_complete:
    file_ext = uploaded.name.lower().split(".")[-1]
    is_video = file_ext in ["mp4", "mov", "m4v", "webm", "avi"]
    is_image = file_ext in ["jpg", "jpeg", "png", "webp", "bmp"]

    st.markdown("---")
    st.markdown('<div class="section-title">🔍 Analiz Sonuçları</div>', unsafe_allow_html=True)

    # VIDEO İŞLEME
    if is_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            # Süre kontrolü
            duration = get_video_duration(tmp_path)

            if duration > 15.0:
                st.error(f"""
                ❌ **Video çok uzun!**

                Yüklediğiniz video **{duration:.1f} saniye** uzunluğunda.
                Maksimum izin verilen süre **15 saniye**dir.

                Lütfen videoyu kısaltarak tekrar yükleyin.
                """)
                os.unlink(tmp_path)
                st.stop()

            st.info(f"📹 Video: **{uploaded.name}** | Süre: **{duration:.1f}s** | Her saniyeden 3 kare alınacak")

            progress = st.progress(0)
            status = st.empty()
            status.text("Kareler çıkarılıyor...")

            frames = extract_frames(tmp_path, frames_per_second=3, max_duration=15.0)

            if not frames:
                st.error("❌ Video okunamadı.")
                os.unlink(tmp_path)
                st.stop()

            st.info(f"📊 Toplam **{len(frames)} kare** çıkarıldı")

            results, annotated = process_frames(frames, load_score_val, coupling_val, activity_val, progress, status)
            progress.empty()
            status.empty()

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    elif is_image:
        file_bytes = uploaded.read()
        frames = extract_image_frame(file_bytes)

        if not frames:
            st.error("❌ Görüntü okunamadı.")
            st.stop()

        progress = st.progress(0)
        status = st.empty()
        results, annotated = process_frames(frames, load_score_val, coupling_val, activity_val, progress, status)
        progress.empty()
        status.empty()
    else:
        st.error("❌ Desteklenmeyen dosya formatı.")
        st.stop()

    # Sonuç yoksa
    if not results:
        st.error("❌ Hiçbir karede kişi tespit edilemedi. Kameranın çalışanı tam gösterdiğinden emin olun.")
        st.stop()

    # ── SONUÇLARI HESAPLA ──
    worst = max(results, key=lambda r: r.final_score)
    avg_score = sum(r.final_score for r in results) / len(results)
    max_score = worst.final_score

    # En yüksek skora sahip tüm karelerin zamanları
    max_frames = [r for r in results if r.final_score == max_score]
    max_times = sorted(set(r.frame_time for r in max_frames))

    # ── ANA SKOR KARTI ──
    col_score, col_details, col_skeleton = st.columns([1, 2, 2])

    with col_score:
        st.markdown(f"""
        <div style="background:white; border-radius:14px; padding:20px; border:2px solid {worst.color};
                    text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.08)">
            <div style="font-size:11px; color:#64748b; font-weight:700;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px">
                EN YÜKSEK REBA SKORU
            </div>
            <div style="font-size:80px; font-weight:900; color:{worst.color}; line-height:1">
                {max_score}
            </div>
            <div style="font-size:11px; color:#94a3b8; margin-bottom:12px">/15</div>
            <div style="background:{worst.color}22; color:{worst.color}; font-weight:700;
                        padding:6px 14px; border-radius:20px; font-size:13px; display:inline-block">
                {worst.risk_level}
            </div>
            <div style="font-size:11px; color:#64748b; margin-top:10px; line-height:1.5">
                {worst.action}
            </div>
            <hr style="border-color:#e2e8f0; margin:12px 0">
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px">
                <span style="color:#64748b">Ortalama Skor</span>
                <span style="font-weight:700; color:#1e293b">{avg_score:.1f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px">
                <span style="color:#64748b">Analiz Edilen Kare</span>
                <span style="font-weight:700; color:#1e293b">{len(results)}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px">
                <span style="color:#64748b">Güven</span>
                <span style="font-weight:700; color:#1e293b">{worst.angles.confidence:.0%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_details:
        st.markdown("**Segment Skorları (En Kötü Kare)**")

        segments = [
            ("🔝 Boyun", worst.neck_score, 6),
            ("🧍 Gövde", worst.trunk_score, 5),
            ("🦵 Bacak", worst.leg_score, 4),
            ("💪 Üst Kol", worst.upper_arm_score, 6),
            ("🦾 Alt Kol", worst.lower_arm_score, 2),
            ("🤚 Bilek", worst.wrist_score, 3),
        ]

        for label, val, max_val in segments:
            pct = val / max_val
            bar_color = "#22c55e" if pct <= 0.4 else "#fbbf24" if pct <= 0.65 else "#ef4444"
            st.markdown(f"""
            <div style="margin-bottom:8px">
                <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px">
                    <span>{label}</span>
                    <span style="font-weight:700; color:{bar_color}">{val}/{max_val}</span>
                </div>
                <div style="background:#e2e8f0; border-radius:4px; height:8px; overflow:hidden">
                    <div style="width:{pct*100:.0f}%; background:{bar_color}; height:100%;
                                border-radius:4px; transition:width 0.5s"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**Skor Hesaplama**")
        st.markdown(f"""
        <div style="background:#f8fafc; border-radius:8px; padding:10px; font-size:12px; border:1px solid #e2e8f0">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px">
            <span style="color:#64748b">Tablo A (Boyun+Gövde+Bacak)</span>
            <span style="font-weight:700">{worst.table_a_score}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px">
            <span style="color:#64748b">+ Yük Skoru</span>
            <span style="font-weight:700; color:#f97316">+{worst.load_score}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-weight:700; border-top:1px solid #e2e8f0; padding-top:4px">
            <span>= Skor A</span>
            <span style="color:#2563eb">{worst.score_a}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px; margin-top:8px">
            <span style="color:#64748b">Tablo B (Kol+Bilek)</span>
            <span style="font-weight:700">{worst.table_b_score}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px">
            <span style="color:#64748b">+ Tutma Skoru</span>
            <span style="font-weight:700; color:#f97316">+{worst.coupling_score}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-weight:700; border-top:1px solid #e2e8f0; padding-top:4px">
            <span>= Skor B</span>
            <span style="color:#2563eb">{worst.score_b}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px; margin-top:8px">
            <span style="color:#64748b">Tablo C (A × B)</span>
            <span style="font-weight:700">{worst.score_c}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px">
            <span style="color:#64748b">+ Aktivite Skoru</span>
            <span style="font-weight:700; color:#f97316">+{worst.activity_score}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-weight:800; font-size:14px;
                    border-top:2px solid #1e3a5f; padding-top:6px; margin-top:4px">
            <span style="color:#1e3a5f">= NİHAİ REBA SKORU</span>
            <span style="color:{worst.color}">{worst.final_score}</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_skeleton:
        st.markdown("**En Riskli Kare — İskelet Overlay**")

        # En kötü kareyi bul
        best_ann = None
        for ann_frame, reba in annotated:
            if reba and reba.frame_time == worst.frame_time:
                best_ann = ann_frame
                break
        if best_ann is None:
            for ann_frame, reba in annotated:
                if reba and reba.final_score == worst.final_score:
                    best_ann = ann_frame
                    break

        if best_ann is not None:
            rgb_frame = cv2.cvtColor(best_ann, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, caption=f"t={worst.frame_time:.2f}s | REBA={worst.final_score}",
                     use_container_width=True)

    # ── EN YÜKSEK RİSK ZAMANLARI ──
    st.markdown("---")
    st.markdown(f'<div class="section-title">⏱️ En Yüksek Risk Zamanları (REBA={max_score})</div>',
                unsafe_allow_html=True)

    if len(max_times) == 1:
        st.markdown(f"""
        <div style="background:{worst.color}15; border:1px solid {worst.color}50;
                    border-radius:8px; padding:12px 16px; font-size:14px">
        ⚠️ En yüksek risk skoru <b>{worst.color and max_score}</b> değerinde,
        <b>t = {max_times[0]:.2f}s</b> anında tespit edildi.
        </div>
        """, unsafe_allow_html=True)
    else:
        times_str = " &nbsp;•&nbsp; ".join([
            f'<span class="frame-badge" style="background:{worst.color}22; color:{worst.color}; border:1px solid {worst.color}50">t={t:.2f}s</span>'
            for t in max_times
        ])
        st.markdown(f"""
        <div style="background:{worst.color}15; border:1px solid {worst.color}50;
                    border-radius:8px; padding:12px 16px; font-size:14px">
        ⚠️ En yüksek risk skoru <b>REBA={max_score}</b>, <b>{len(max_times)} farklı anda</b> tespit edildi:<br>
        <div style="margin-top:8px">{times_str}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── ZAMAN ÇİZELGESİ ──
    if len(results) > 1:
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Zaman Çizelgesi</div>', unsafe_allow_html=True)

        results_sorted = sorted(results, key=lambda r: r.frame_time)
        cols = st.columns(min(len(results_sorted), 12))
        for i, r in enumerate(results_sorted[:12]):
            with cols[i]:
                is_worst = r.final_score == max_score
                border = f"border:2px solid {r.color}" if is_worst else f"border:1px solid {r.color}50"
                st.markdown(f"""
                <div style="background:white; border-radius:8px; padding:8px 4px; text-align:center;
                            {border}; margin-bottom:4px">
                    <div style="font-size:20px; font-weight:800; color:{r.color}">{r.final_score}</div>
                    <div style="font-size:9px; color:#64748b">{r.frame_time:.2f}s</div>
                    {'<div style="font-size:8px; color:' + r.color + '; font-weight:700">▲ MAX</div>' if is_worst else ''}
                </div>
                """, unsafe_allow_html=True)

    # ── PDF RAPOR ──
    st.markdown("---")
    st.markdown('<div class="section-title">📄 Rapor</div>', unsafe_allow_html=True)

    col_pdf, col_json = st.columns([1, 1])

    form_info = {
        'bolum': bolum,
        'is_istasyonu': is_istasyonu,
        'is_adimi': is_adimi,
        'analist': analist,
        'tarih': str(tarih),
        'dosya_adi': uploaded.name,
        'yuk_kg': yuk_kg,
        'shock': shock,
        'load_score': load_score_val,
        'coupling': coupling_val,
        'coupling_label': coupling_label,
        'activity': activity_val,
    }

    with col_pdf:
        pdf_bytes = generate_pdf_report(form_info, results, worst, avg_score)
        if pdf_bytes:
            dosya_adi = f"REBA_{bolum or is_istasyonu or 'analiz'}_{tarih}.pdf".replace(" ", "_")
            st.download_button(
                label="⬇️ PDF Raporu İndir",
                data=pdf_bytes,
                file_name=dosya_adi,
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning("PDF oluşturulamadı. reportlab kurulu değil.")
            st.markdown("requirements.txt'e `reportlab` ekleyin.")

    with col_json:
        export = {
            "form": form_info,
            "en_yuksek_skor": max_score,
            "risk_seviyesi": worst.risk_level,
            "ortalama_skor": round(avg_score, 2),
            "en_riskli_zamanlar_sn": max_times,
            "analiz_edilen_kare": len(results),
            "kareler": [
                {
                    "zaman_sn": r.frame_time,
                    "reba_skoru": r.final_score,
                    "risk": r.risk_level,
                    "skor_a": r.score_a,
                    "skor_b": r.score_b,
                    "boyun": r.neck_score,
                    "govde": r.trunk_score,
                    "bacak": r.leg_score,
                    "ust_kol": r.upper_arm_score,
                    "alt_kol": r.lower_arm_score,
                    "bilek": r.wrist_score,
                }
                for r in sorted(results, key=lambda x: x.frame_time)
            ]
        }
        json_dosya = f"REBA_{bolum or 'analiz'}_{tarih}.json".replace(" ", "_")
        st.download_button(
            label="⬇️ JSON Veri İndir",
            data=json.dumps(export, ensure_ascii=False, indent=2),
            file_name=json_dosya,
            mime="application/json",
            use_container_width=True,
        )

# ── FOOTER ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:10px; color:#94a3b8; line-height:2">
REBA Ergonomi Analiz Ajanı v4.0 &nbsp;|&nbsp;
Hignett & McAtamney (2000). Rapid Entire Body Assessment. Applied Ergonomics, 31(2), 201-205. &nbsp;|&nbsp;
MediaPipe Pose (Google LLC) &nbsp;|&nbsp;
AI tabanlı açı tahmini ±3-5° doğruluk payı içerir — profesyonel değerlendirmenin yerini tutmaz
>>>>>>> ddded5f36dedc372c567f0ac5bae861d227b2eb3
</div>
""", unsafe_allow_html=True)
