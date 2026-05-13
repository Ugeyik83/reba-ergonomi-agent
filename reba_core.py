"""
reba_core.py — REBA Hesaplama Motoru
Veri sınıfları, açı hesaplama, REBA skorlama.
Hiçbir UI / görsel bağımlılığı yoktur.
Referans: Hignett & McAtamney (2000), Applied Ergonomics 31(2), 201-205
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ════════════════════════════════════════════════════════
# VERİ SINIFLARI
# ════════════════════════════════════════════════════════

class AcilarObj:
    """MediaPipe landmark'larından hesaplanan vücut segment açıları."""
    def __init__(self):
        # Boyun
        self.boyun_flexion: float = 0.0      # derece, öne eğilme
        self.boyun_yan_egim: float = 0.0     # derece, yana eğilme
        self.boyun_donus: bool = False        # dönme hareketi var mı

        # Gövde
        self.govde_flexion: float = 0.0      # derece, öne eğilme
        self.govde_yan_egim: float = 0.0     # derece, yana eğilme
        self.govde_donus: bool = False        # dönme hareketi var mı

        # Bacak
        self.diz_flexion_sol: float = 0.0    # derece
        self.diz_flexion_sag: float = 0.0    # derece
        self.bilateral_destek: bool = True   # iki ayak üzerinde mi

        # Üst Kol
        self.ust_kol_aci: float = 0.0        # derece, gövdeden uzaklık
        self.omuz_kalkmis: bool = False       # omuz yukarı kalkmış mı
        self.kol_abdukte: bool = False        # kol dışa açılmış mı
        self.kol_destekli: bool = False       # kol desteklenmiş mi

        # Alt Kol
        self.alt_kol_aci: float = 0.0        # derece, dirsek fleksiyonu

        # Bilek
        self.bilek_aci: float = 0.0          # derece, nötralden sapma
        self.bilek_donus: bool = False        # bilek dönme / yana bükülme

        # Güven skoru (MediaPipe visibility ortalaması)
        self.guven: float = 0.0


@dataclass
class REBASkoru:
    """Hesaplanmış REBA skoru ve tüm ara değerler."""
    # Segment skorları
    boyun_skoru: int = 0
    govde_skoru: int = 0
    bacak_skoru: int = 0
    ust_kol_skoru: int = 0
    alt_kol_skoru: int = 0
    bilek_skoru: int = 0

    # Tablo değerleri
    tablo_a: int = 0
    tablo_b: int = 0

    # Manuel girdi skorları
    yuk_skoru: int = 0
    tutma_skoru: int = 0
    aktivite_skoru: int = 0

    # Ara skorlar
    skor_a: int = 0    # Tablo A + Yük
    skor_b: int = 0    # Tablo B + Tutma
    skor_c: int = 0    # Tablo C

    # Final
    final_skor: int = 0
    risk_seviyesi: str = ""
    aksiyon: str = ""
    renk: str = "#16a34a"

    # Açı referansı (overlay ve PDF için)
    acılar: Optional[AcilarObj] = None


@dataclass
class FotoSonuc:
    """Tek bir fotoğraf analiz sonucu."""
    idx: int = 0
    dosya_adi: str = ""
    skor: Optional[REBASkoru] = None
    overlay_img: Optional[np.ndarray] = None   # cv2 BGR array
    hata: str = ""


# ════════════════════════════════════════════════════════
# REBA TABLOLARI (Hignett & McAtamney, 2000)
# ════════════════════════════════════════════════════════

# TABLO_A[govde-1][boyun-1][bacak-1]
TABLO_A = [
    [[1,2,3,4],[1,2,3,4],[3,3,5,6]],
    [[2,3,4,5],[3,4,5,6],[4,5,6,7]],
    [[2,4,5,6],[4,5,6,7],[5,6,7,8]],
    [[3,5,6,7],[5,6,7,8],[6,7,8,9]],
    [[4,6,7,8],[6,7,8,9],[7,8,9,9]],
]

# TABLO_B[ustkol-1][altkol-1][bilek-1]
TABLO_B = [
    [[1,2,2],[1,2,3]],
    [[1,2,3],[2,3,4]],
    [[3,4,5],[4,5,5]],
    [[4,5,5],[5,6,7]],
    [[6,7,8],[7,8,8]],
    [[7,8,8],[8,9,9]],
]

# TABLO_C[skor_a-1][skor_b-1]
TABLO_C = [
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


# ════════════════════════════════════════════════════════
# AÇI HESAPLAMA FONKSİYONLARI
# ════════════════════════════════════════════════════════

def aci_3nokta(a: tuple, b: tuple, c: tuple) -> float:
    """b noktasındaki açıyı hesapla (a-b-c üçgeni)."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    mag = np.linalg.norm(ba) * np.linalg.norm(bc)
    if mag == 0:
        return 0.0
    cos_val = np.clip(np.dot(ba, bc) / mag, -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


def aci_dikey(a: tuple, b: tuple) -> float:
    """İki nokta arasındaki çizginin dikey eksenden açısı."""
    return abs(math.degrees(math.atan2(abs(b[0] - a[0]), abs(b[1] - a[1]))))


def vucut_acilari_hesapla(landmarks, w: int, h: int) -> AcilarObj:
    """
    MediaPipe pose landmark'larından REBA için gerekli açıları hesapla.
    landmarks: mediapipe pose_landmarks.landmark listesi
    w, h: görüntü genişlik ve yüksekliği (piksel)
    """
    import mediapipe as mp
    LM = mp.solutions.pose.PoseLandmark

    a = AcilarObj()

    def p(idx):
        """Landmark → piksel koordinat."""
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    def v(idx):
        """Landmark görünürlük skoru."""
        return landmarks[idx].visibility

    # Omuz ve kalça orta noktaları
    mid_omuz = (
        (p(LM.LEFT_SHOULDER)[0] + p(LM.RIGHT_SHOULDER)[0]) / 2,
        (p(LM.LEFT_SHOULDER)[1] + p(LM.RIGHT_SHOULDER)[1]) / 2,
    )
    mid_kalca = (
        (p(LM.LEFT_HIP)[0] + p(LM.RIGHT_HIP)[0]) / 2,
        (p(LM.LEFT_HIP)[1] + p(LM.RIGHT_HIP)[1]) / 2,
    )

    # Genel güven skoru
    a.guven = float(np.mean([v(l) for l in [
        LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
        LM.LEFT_HIP, LM.RIGHT_HIP,
    ]]))

    # ── BOYUN ──
    a.boyun_flexion = aci_dikey(p(LM.NOSE), mid_omuz)
    ear_dx = abs(p(LM.LEFT_EAR)[0] - p(LM.RIGHT_EAR)[0])
    ear_dy = abs(p(LM.LEFT_EAR)[1] - p(LM.RIGHT_EAR)[1])
    if ear_dx > 0:
        a.boyun_yan_egim = math.degrees(math.atan2(ear_dy, ear_dx))
    a.boyun_donus = a.boyun_yan_egim > 20

    # ── GÖVDE ──
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

    # ── BACAKLAR ──
    a.diz_flexion_sol = 180 - aci_3nokta(
        p(LM.LEFT_HIP), p(LM.LEFT_KNEE), p(LM.LEFT_ANKLE))
    a.diz_flexion_sag = 180 - aci_3nokta(
        p(LM.RIGHT_HIP), p(LM.RIGHT_KNEE), p(LM.RIGHT_ANKLE))
    a.bilateral_destek = v(LM.LEFT_ANKLE) > 0.3 and v(LM.RIGHT_ANKLE) > 0.3

    # ── ÜST KOL ──
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

    # ── ALT KOL ──
    la_sol = aci_3nokta(
        p(LM.LEFT_SHOULDER), p(LM.LEFT_ELBOW), p(LM.LEFT_WRIST))
    la_sag = aci_3nokta(
        p(LM.RIGHT_SHOULDER), p(LM.RIGHT_ELBOW), p(LM.RIGHT_WRIST))
    a.alt_kol_aci = min(la_sol, la_sag)

    # ── BİLEK ──
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

def reba_skorla(
    a: AcilarObj,
    yuk_skoru: int,
    tutma: int,
    aktivite: int
) -> REBASkoru:
    """
    REBA formunu hesapla.
    a: vucut_acilari_hesapla() çıktısı
    yuk_skoru: 0-3 (yük + ani kuvvet)
    tutma: 0-3 (tutma kalitesi)
    aktivite: 0-3 (aktivite checkpoint'leri toplamı)
    """
    r = REBASkoru()
    r.yuk_skoru = yuk_skoru
    r.tutma_skoru = tutma
    r.aktivite_skoru = aktivite
    r.acılar = a

    # ── BOYUN (1-6) ──
    if a.boyun_flexion <= 20:
        r.boyun_skoru = 1
    elif a.boyun_flexion <= 40:
        r.boyun_skoru = 2
    else:
        r.boyun_skoru = 3
    if a.boyun_yan_egim > 15:
        r.boyun_skoru += 1   # yana eğilme modifier
    if a.boyun_donus:
        r.boyun_skoru += 1   # dönme modifier
    r.boyun_skoru = min(r.boyun_skoru, 6)

    # ── GÖVDE (1-5) ──
    if a.govde_flexion <= 5:
        r.govde_skoru = 1
    elif a.govde_flexion <= 20:
        r.govde_skoru = 2
    elif a.govde_flexion <= 60:
        r.govde_skoru = 3
    else:
        r.govde_skoru = 4
    if a.govde_yan_egim > 10:
        r.govde_skoru += 1   # yana eğilme modifier
    if a.govde_donus:
        r.govde_skoru += 1   # dönme modifier
    r.govde_skoru = min(r.govde_skoru, 5)

    # ── BACAK (1-4) ──
    r.bacak_skoru = 1 if a.bilateral_destek else 2
    diz = max(a.diz_flexion_sol, a.diz_flexion_sag)
    if 30 <= diz < 60:
        r.bacak_skoru += 1
    elif diz >= 60:
        r.bacak_skoru += 2
    r.bacak_skoru = min(r.bacak_skoru, 4)

    # ── TABLO A → SKOR A ──
    t = min(r.govde_skoru - 1, 4)
    n = min(r.boyun_skoru - 1, 2)
    l = min(r.bacak_skoru - 1, 3)
    r.tablo_a = TABLO_A[t][n][l]
    r.skor_a = r.tablo_a + r.yuk_skoru

    # ── ÜST KOL (1-6) ──
    if a.ust_kol_aci <= 20:
        r.ust_kol_skoru = 1
    elif a.ust_kol_aci <= 45:
        r.ust_kol_skoru = 2
    elif a.ust_kol_aci <= 90:
        r.ust_kol_skoru = 3
    else:
        r.ust_kol_skoru = 4
    if a.omuz_kalkmis:
        r.ust_kol_skoru += 1   # omuz kalkış modifier
    if a.kol_abdukte:
        r.ust_kol_skoru += 1   # abduksiyon modifier
    if a.kol_destekli:
        r.ust_kol_skoru -= 1   # destek modifier (azaltır)
    r.ust_kol_skoru = max(1, min(r.ust_kol_skoru, 6))

    # ── ALT KOL (1-2) ──
    r.alt_kol_skoru = 1 if 60 <= a.alt_kol_aci <= 100 else 2

    # ── BİLEK (1-3) ──
    r.bilek_skoru = 1 if a.bilek_aci <= 15 else 2
    if a.bilek_donus:
        r.bilek_skoru += 1   # dönme modifier
    r.bilek_skoru = min(r.bilek_skoru, 3)

    # ── TABLO B → SKOR B ──
    u = min(r.ust_kol_skoru - 1, 5)
    la = min(r.alt_kol_skoru - 1, 1)
    w = min(r.bilek_skoru - 1, 2)
    r.tablo_b = TABLO_B[u][la][w]
    r.skor_b = r.tablo_b + r.tutma_skoru

    # ── TABLO C → FİNAL ──
    ca = min(r.skor_a - 1, 11)
    cb = min(r.skor_b - 1, 11)
    r.skor_c = TABLO_C[ca][cb]
    r.final_skor = min(r.skor_c + r.aktivite_skoru, 15)

    # ── RİSK SEVİYESİ ──
    s = r.final_skor
    if s == 1:
        r.risk_seviyesi = "Önemsiz Risk"
        r.renk = "#16a34a"
        r.aksiyon = "Herhangi bir önlem gerekmez"
    elif s <= 3:
        r.risk_seviyesi = "Düşük Risk"
        r.renk = "#65a30d"
        r.aksiyon = "Gerekirse iyileştirme yapılabilir"
    elif s <= 7:
        r.risk_seviyesi = "Orta Seviyeli Risk"
        r.renk = "#d97706"
        r.aksiyon = "Daha ayrıntılı incele, değişiklik planla"
    elif s <= 10:
        r.risk_seviyesi = "Yüksek Risk"
        r.renk = "#dc2626"
        r.aksiyon = "Araştırma yap ve aksiyon al"
    else:
        r.risk_seviyesi = "Çok Yüksek Risk"
        r.renk = "#7c3aed"
        r.aksiyon = "Süreç çalışmaya uygun değil, derhal revize et"

    return r


def risk_info(skor: int) -> tuple:
    """Skor → (risk_adı, hex_renk) döndür."""
    if skor <= 1:   return "Önemsiz", "#16a34a"
    elif skor <= 3: return "Düşük", "#65a30d"
    elif skor <= 7: return "Orta", "#d97706"
    elif skor <= 10: return "Yüksek", "#dc2626"
    else:           return "Çok Yüksek", "#7c3aed"
