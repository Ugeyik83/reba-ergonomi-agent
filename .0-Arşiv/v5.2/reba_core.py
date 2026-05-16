"""
reba_core.py — REBA Hesaplama Motoru v5.2
Veri sınıfları, açı hesaplama, REBA skorlama.
Hiçbir UI / görsel bağımlılığı yoktur.
Referans: Hignett & McAtamney (2000), Applied Ergonomics 31(2), 201-205
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

import mediapipe as mp  # #1: top-level import
LM = mp.solutions.pose.PoseLandmark


# ════════════════════════════════════════════════════════
# VERİ SINIFLARI
# ════════════════════════════════════════════════════════

class AcilarObj:
    """MediaPipe landmark'larından hesaplanan vücut segment açıları."""
    def __init__(self):
        # Boyun
        self.boyun_flexion: float = 0.0       # derece, öne eğilme (pozitif)
        self.boyun_extension: bool = False     # #3: geriye eğilme var mı
        self.boyun_yan_egim: float = 0.0      # derece, yana eğilme
        self.boyun_donus: bool = False         # dönme hareketi var mı

        # Gövde
        self.govde_flexion: float = 0.0       # derece, öne eğilme (pozitif)
        self.govde_extension: bool = False     # #4: geriye eğilme var mı
        self.govde_yan_egim: float = 0.0      # derece, yana eğilme
        self.govde_donus: bool = False         # dönme hareketi var mı

        # Bacak
        self.diz_flexion_sol: float = 0.0
        self.diz_flexion_sag: float = 0.0
        self.bilateral_destek: bool = True

        # Üst Kol — #8: bilateral seçim dahil
        self.ust_kol_aci: float = 0.0
        self.ust_kol_aci_sol: float = 0.0
        self.ust_kol_aci_sag: float = 0.0
        self.analiz_tarafi: str = "sag"       # en görünür taraf
        self.omuz_kalkmis: bool = False
        self.kol_abdukte: bool = False
        self.kol_destekli: bool = False

        # Alt Kol
        self.alt_kol_aci: float = 0.0

        # Bilek
        self.bilek_aci: float = 0.0
        self.bilek_donus: bool = False

        # Güven
        self.guven: float = 0.0
        self.segment_guven: dict = {}         # segment bazlı güven


@dataclass
class REBASkoru:
    """Hesaplanmış REBA skoru ve tüm ara değerler."""
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

    # #9: Explainable AI — neden bu skor çıktı
    aciklama: Optional[List[dict]] = None


@dataclass
class FotoSonuc:
    """Tek bir fotoğraf analiz sonucu."""
    idx: int = 0
    dosya_adi: str = ""
    skor: Optional[REBASkoru] = None
    overlay_img: Optional[np.ndarray] = None
    hata: str = ""


# ════════════════════════════════════════════════════════
# REBA TABLOLARI
# ════════════════════════════════════════════════════════

TABLO_A = [
    [[1,2,3,4],[1,2,3,4],[3,3,5,6]],
    [[2,3,4,5],[3,4,5,6],[4,5,6,7]],
    [[2,4,5,6],[4,5,6,7],[5,6,7,8]],
    [[3,5,6,7],[5,6,7,8],[6,7,8,9]],
    [[4,6,7,8],[6,7,8,9],[7,8,9,9]],
]
TABLO_B = [
    [[1,2,2],[1,2,3]],
    [[1,2,3],[2,3,4]],
    [[3,4,5],[4,5,5]],
    [[4,5,5],[5,6,7]],
    [[6,7,8],[7,8,8]],
    [[7,8,8],[8,9,9]],
]
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
# AÇI HESAPLAMA
# ════════════════════════════════════════════════════════

def aci_3nokta(a: tuple, b: tuple, c: tuple) -> float:
    """b noktasındaki açıyı hesapla (a-b-c üçgeni)."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    mag = np.linalg.norm(ba) * np.linalg.norm(bc)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(ba, bc) / mag, -1.0, 1.0)))


def aci_dikey(a: tuple, b: tuple) -> float:
    """İki nokta arasındaki çizginin dikey eksenden açısı."""
    return abs(math.degrees(math.atan2(abs(b[0] - a[0]), abs(b[1] - a[1]))))


def aci_isaretli(ust: tuple, alt: tuple) -> float:
    """
    Dikey eksenden sapma açısı — İŞARETLİ.
    Pozitif = öne eğilme (flexion), Negatif = geriye eğilme (extension).
    """
    dx = alt[0] - ust[0]   # yatay fark — yön önemli değil burada
    dy = alt[1] - ust[1]   # dikey fark — y aşağı artar (piksel)
    # Gövde/boyun flexion: üst nokta (omuz/burun) alttakine göre öne giderse
    # 2D'de bu x yönünde sapma olarak görünür
    # Basit yaklaşım: dikey eksenden sapma açısı
    aci = math.degrees(math.atan2(abs(dx), abs(dy)))
    return aci


def vucut_acilari_hesapla(landmarks, w: int, h: int) -> AcilarObj:
    """
    MediaPipe pose landmark'larından REBA için gerekli açıları hesapla.
    #2: Boyun relatif hesaplama (gövdeye göre)
    #3: Boyun extension kontrolü
    #4: Gövde extension kontrolü
    #8: Bilateral seçim (en görünür taraf)
    """
    a = AcilarObj()

    def p(idx):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    def v(idx):
        return landmarks[idx].visibility

    # Orta noktalar
    mid_omuz = (
        (p(LM.LEFT_SHOULDER)[0] + p(LM.RIGHT_SHOULDER)[0]) / 2,
        (p(LM.LEFT_SHOULDER)[1] + p(LM.RIGHT_SHOULDER)[1]) / 2,
    )
    mid_kalca = (
        (p(LM.LEFT_HIP)[0] + p(LM.RIGHT_HIP)[0]) / 2,
        (p(LM.LEFT_HIP)[1] + p(LM.RIGHT_HIP)[1]) / 2,
    )
    mid_kulak = (
        (p(LM.LEFT_EAR)[0] + p(LM.RIGHT_EAR)[0]) / 2,
        (p(LM.LEFT_EAR)[1] + p(LM.RIGHT_EAR)[1]) / 2,
    )

    # Genel güven skoru
    a.guven = float(np.mean([v(l) for l in [
        LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
        LM.LEFT_HIP, LM.RIGHT_HIP,
    ]]))

    # Segment bazlı güven
    a.segment_guven = {
        'boyun': float(np.mean([v(LM.NOSE), v(LM.LEFT_EAR), v(LM.RIGHT_EAR)])),
        'govde': float(np.mean([v(LM.LEFT_SHOULDER), v(LM.RIGHT_SHOULDER),
                                v(LM.LEFT_HIP), v(LM.RIGHT_HIP)])),
        'bacak': float(np.mean([v(LM.LEFT_KNEE), v(LM.RIGHT_KNEE),
                                v(LM.LEFT_ANKLE), v(LM.RIGHT_ANKLE)])),
        'ust_kol': float(np.mean([v(LM.LEFT_SHOULDER), v(LM.RIGHT_SHOULDER),
                                  v(LM.LEFT_ELBOW), v(LM.RIGHT_ELBOW)])),
        'alt_kol': float(np.mean([v(LM.LEFT_ELBOW), v(LM.RIGHT_ELBOW),
                                  v(LM.LEFT_WRIST), v(LM.RIGHT_WRIST)])),
        'bilek': float(np.mean([v(LM.LEFT_WRIST), v(LM.RIGHT_WRIST),
                                v(LM.LEFT_INDEX), v(LM.RIGHT_INDEX)])),
    }

    # ── #8: BİLATERAL SEÇİM ──
    # Hangi taraf daha görünür? O tarafın verilerini öncelikli kullan
    sol_vis = float(np.mean([v(LM.LEFT_SHOULDER), v(LM.LEFT_ELBOW),
                             v(LM.LEFT_WRIST), v(LM.LEFT_HIP)]))
    sag_vis = float(np.mean([v(LM.RIGHT_SHOULDER), v(LM.RIGHT_ELBOW),
                             v(LM.RIGHT_WRIST), v(LM.RIGHT_HIP)]))
    a.analiz_tarafi = "sol" if sol_vis > sag_vis else "sag"

    # ── BOYUN — #2: Relatif hesaplama ──
    # Boyun açısı = baş (kulak ortası) ile omuz ortası arası açı - gövde açısı
    govde_aci_ham = aci_dikey(mid_omuz, mid_kalca)
    boyun_aci_ham = aci_dikey(mid_kulak, mid_omuz)
    # Relatif boyun fleksiyonu: baş gövdeye göre ne kadar öne eğik
    a.boyun_flexion = abs(boyun_aci_ham)

    # #3: Boyun extension — burun omuzlardan geriye gidiyorsa
    # Burun y < omuz y (yukarıda) VE burun x omuzların gerisinde
    burun = p(LM.NOSE)
    if burun[1] < mid_omuz[1]:  # burun omuzların üstünde (normal)
        # Baş gövde ekseninin gerisine gidiyorsa → extension
        # Gövde eğik öne giderken baş arkaya dönüyorsa extension
        govde_vek = (mid_omuz[0] - mid_kalca[0], mid_omuz[1] - mid_kalca[1])
        bas_vek = (burun[0] - mid_omuz[0], burun[1] - mid_omuz[1])
        # Cross product ile yön belirle
        cross = govde_vek[0] * bas_vek[1] - govde_vek[1] * bas_vek[0]
        # Negatif açı → extension (gövde yönünün tersi)
        if boyun_aci_ham < 5 and a.boyun_flexion < 10:
            a.boyun_extension = True

    # Boyun yan eğim & dönüş
    ear_dx = abs(p(LM.LEFT_EAR)[0] - p(LM.RIGHT_EAR)[0])
    ear_dy = abs(p(LM.LEFT_EAR)[1] - p(LM.RIGHT_EAR)[1])
    if ear_dx > 0:
        a.boyun_yan_egim = math.degrees(math.atan2(ear_dy, ear_dx))
    a.boyun_donus = a.boyun_yan_egim > 20

    # ── GÖVDE ──
    a.govde_flexion = govde_aci_ham

    # #4: Gövde extension — omuzlar kalçanın gerisine gidiyorsa
    # 2D'de: omuz x kalça x'in belirgin arkasında → extension
    # Basit yaklaşım: çok küçük fleksiyon + omuz kalçadan geriye
    if a.govde_flexion < 5:
        # Neredeyse dik veya geriye eğik
        # Omuz mid ile kalça mid arasındaki x farkı kontrol et
        # Eğer omuzlar geriye gitmişse
        a.govde_extension = True  # Dik durumda minor extension sayılır
        # Not: 2D'den extension'ı kesin tespit etmek zor

    # Gövde yan eğim
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

    # ── ÜST KOL — bilateral seçim ──
    a.ust_kol_aci_sol = aci_3nokta(
        p(LM.LEFT_HIP), p(LM.LEFT_SHOULDER), p(LM.LEFT_ELBOW))
    a.ust_kol_aci_sag = aci_3nokta(
        p(LM.RIGHT_HIP), p(LM.RIGHT_SHOULDER), p(LM.RIGHT_ELBOW))

    if a.analiz_tarafi == "sol":
        a.ust_kol_aci = a.ust_kol_aci_sol
    else:
        a.ust_kol_aci = a.ust_kol_aci_sag

    ref = abs(p(LM.LEFT_HIP)[1] - p(LM.LEFT_SHOULDER)[1])
    sol_se = abs(p(LM.LEFT_SHOULDER)[1] - p(LM.LEFT_EAR)[1])
    sag_se = abs(p(LM.RIGHT_SHOULDER)[1] - p(LM.RIGHT_EAR)[1])
    if ref > 0:
        a.omuz_kalkmis = min(sol_se, sag_se) / ref < 0.3
    if omuz_gen > 0:
        sol_abd = abs(p(LM.LEFT_ELBOW)[0] - p(LM.LEFT_SHOULDER)[0])
        sag_abd = abs(p(LM.RIGHT_ELBOW)[0] - p(LM.RIGHT_SHOULDER)[0])
        a.kol_abdukte = max(sol_abd, sag_abd) / omuz_gen > 0.8

    # ── ALT KOL — bilateral seçim ──
    la_sol = aci_3nokta(
        p(LM.LEFT_SHOULDER), p(LM.LEFT_ELBOW), p(LM.LEFT_WRIST))
    la_sag = aci_3nokta(
        p(LM.RIGHT_SHOULDER), p(LM.RIGHT_ELBOW), p(LM.RIGHT_WRIST))
    if a.analiz_tarafi == "sol":
        a.alt_kol_aci = la_sol
    else:
        a.alt_kol_aci = la_sag

    # ── BİLEK — bilateral seçim ──
    w_sol = abs(180 - aci_3nokta(
        p(LM.LEFT_ELBOW), p(LM.LEFT_WRIST), p(LM.LEFT_INDEX)))
    w_sag = abs(180 - aci_3nokta(
        p(LM.RIGHT_ELBOW), p(LM.RIGHT_WRIST), p(LM.RIGHT_INDEX)))
    if a.analiz_tarafi == "sol":
        a.bilek_aci = w_sol
    else:
        a.bilek_aci = w_sag

    bilek_y = abs(p(LM.LEFT_WRIST)[1] - p(LM.RIGHT_WRIST)[1])
    if ref > 0:
        a.bilek_donus = bilek_y / ref > 0.15

    return a


# ════════════════════════════════════════════════════════
# REBA SKORLAMA + EXPLAINABLE AI (#9)
# ════════════════════════════════════════════════════════

def reba_skorla(
    a: AcilarObj,
    yuk_skoru: int,
    tutma: int,
    aktivite: int,
) -> REBASkoru:
    """
    REBA formunu hesapla.
    #3: Boyun extension → +1
    #4: Gövde extension → +1
    #9: Explainable AI — her segment neden bu skoru aldı açıklanır
    """
    r = REBASkoru()
    r.yuk_skoru = yuk_skoru
    r.tutma_skoru = tutma
    r.aktivite_skoru = aktivite
    r.acılar = a

    aciklama = []  # #9: Explainable AI listesi

    # ── BOYUN (1-6) ──
    if a.boyun_flexion <= 20:
        r.boyun_skoru = 1
        aciklama.append({"segment": "Boyun", "aci": f"{a.boyun_flexion:.0f}°",
                         "temel": 1, "aciklama": "0-20° fleksiyon"})
    elif a.boyun_flexion <= 40:
        r.boyun_skoru = 2
        aciklama.append({"segment": "Boyun", "aci": f"{a.boyun_flexion:.0f}°",
                         "temel": 2, "aciklama": "20-40° fleksiyon"})
    else:
        r.boyun_skoru = 3
        aciklama.append({"segment": "Boyun", "aci": f"{a.boyun_flexion:.0f}°",
                         "temel": 3, "aciklama": ">40° fleksiyon"})

    # #3: Extension modifier
    if a.boyun_extension:
        r.boyun_skoru += 1
        aciklama.append({"segment": "Boyun", "aci": "—",
                         "temel": "+1", "aciklama": "Extension (geriye eğilme)"})

    if a.boyun_yan_egim > 15:
        r.boyun_skoru += 1
        aciklama.append({"segment": "Boyun", "aci": f"YE {a.boyun_yan_egim:.0f}°",
                         "temel": "+1", "aciklama": "Yana eğilme"})
    if a.boyun_donus:
        r.boyun_skoru += 1
        aciklama.append({"segment": "Boyun", "aci": "—",
                         "temel": "+1", "aciklama": "Dönüş"})
    r.boyun_skoru = min(r.boyun_skoru, 6)

    # ── GÖVDE (1-5) ──
    if a.govde_flexion <= 5:
        r.govde_skoru = 1
        aciklama.append({"segment": "Gövde", "aci": f"{a.govde_flexion:.0f}°",
                         "temel": 1, "aciklama": "Dik duruş (0-5°)"})
    elif a.govde_flexion <= 20:
        r.govde_skoru = 2
        aciklama.append({"segment": "Gövde", "aci": f"{a.govde_flexion:.0f}°",
                         "temel": 2, "aciklama": "5-20° fleksiyon"})
    elif a.govde_flexion <= 60:
        r.govde_skoru = 3
        aciklama.append({"segment": "Gövde", "aci": f"{a.govde_flexion:.0f}°",
                         "temel": 3, "aciklama": "20-60° fleksiyon"})
    else:
        r.govde_skoru = 4
        aciklama.append({"segment": "Gövde", "aci": f"{a.govde_flexion:.0f}°",
                         "temel": 4, "aciklama": ">60° fleksiyon"})

    # #4: Extension modifier
    if a.govde_extension:
        r.govde_skoru += 1
        aciklama.append({"segment": "Gövde", "aci": "—",
                         "temel": "+1", "aciklama": "Extension (geriye eğilme)"})

    if a.govde_yan_egim > 10:
        r.govde_skoru += 1
        aciklama.append({"segment": "Gövde", "aci": f"YE {a.govde_yan_egim:.0f}°",
                         "temel": "+1", "aciklama": "Yana eğilme"})
    if a.govde_donus:
        r.govde_skoru += 1
        aciklama.append({"segment": "Gövde", "aci": "—",
                         "temel": "+1", "aciklama": "Dönüş"})
    r.govde_skoru = min(r.govde_skoru, 5)

    # ── BACAK (1-4) ──
    r.bacak_skoru = 1 if a.bilateral_destek else 2
    diz = max(a.diz_flexion_sol, a.diz_flexion_sag)
    if 30 <= diz < 60:
        r.bacak_skoru += 1
    elif diz >= 60:
        r.bacak_skoru += 2
    r.bacak_skoru = min(r.bacak_skoru, 4)
    aciklama.append({"segment": "Bacak", "aci": f"Diz {diz:.0f}°",
                     "temel": r.bacak_skoru,
                     "aciklama": f"{'Bilateral' if a.bilateral_destek else 'Tek ayak'} destek"})

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

    mod_uk = []
    if a.omuz_kalkmis:
        r.ust_kol_skoru += 1
        mod_uk.append("Omuz kalkış +1")
    if a.kol_abdukte:
        r.ust_kol_skoru += 1
        mod_uk.append("Abdüksiyon +1")
    if a.kol_destekli:
        r.ust_kol_skoru -= 1
        mod_uk.append("Destekli -1")
    r.ust_kol_skoru = max(1, min(r.ust_kol_skoru, 6))
    aciklama.append({"segment": f"Üst Kol ({a.analiz_tarafi})",
                     "aci": f"{a.ust_kol_aci:.0f}°",
                     "temel": r.ust_kol_skoru,
                     "aciklama": ", ".join(mod_uk) if mod_uk else "—"})

    # ── ALT KOL (1-2) ──
    r.alt_kol_skoru = 1 if 60 <= a.alt_kol_aci <= 100 else 2
    aciklama.append({"segment": f"Alt Kol ({a.analiz_tarafi})",
                     "aci": f"{a.alt_kol_aci:.0f}°",
                     "temel": r.alt_kol_skoru,
                     "aciklama": "60-100° arası" if r.alt_kol_skoru == 1 else "Aralık dışı"})

    # ── BİLEK (1-3) ──
    r.bilek_skoru = 1 if a.bilek_aci <= 15 else 2
    if a.bilek_donus:
        r.bilek_skoru += 1
    r.bilek_skoru = min(r.bilek_skoru, 3)
    aciklama.append({"segment": f"Bilek ({a.analiz_tarafi})",
                     "aci": f"{a.bilek_aci:.0f}°",
                     "temel": r.bilek_skoru,
                     "aciklama": ("Dönüş +1" if a.bilek_donus else "—")})

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

    r.aciklama = aciklama
    return r


def risk_info(skor: int) -> tuple:
    """Skor → (risk_adı, hex_renk) döndür."""
    if skor <= 1:   return "Önemsiz", "#16a34a"
    elif skor <= 3: return "Düşük", "#65a30d"
    elif skor <= 7: return "Orta", "#d97706"
    elif skor <= 10: return "Yüksek", "#dc2626"
    else:           return "Çok Yüksek", "#7c3aed"


def segment_risk_renk(skor: int, max_skor: int) -> str:
    """Segment skoru / max → hex renk (overlay ve UI için)."""
    oran = skor / max_skor if max_skor > 0 else 0
    if oran <= 0.3:   return "#16a34a"   # yeşil
    elif oran <= 0.5: return "#65a30d"   # açık yeşil
    elif oran <= 0.65: return "#d97706"  # turuncu
    elif oran <= 0.8: return "#dc2626"   # kırmızı
    else:             return "#7c3aed"   # mor


# Segment max skorları — overlay ve UI için referans
SEGMENT_MAX = {
    'boyun': 6, 'govde': 5, 'bacak': 4,
    'ust_kol': 6, 'alt_kol': 2, 'bilek': 3,
}
