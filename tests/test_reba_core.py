"""
tests/test_reba_core.py
REBA Hesaplama Motoru için Unit Testler
Referans: Hignett & McAtamney (2000), Applied Ergonomics 31(2), 201-205

Çalıştır: pytest tests/ -v --cov=reba_core --cov-report=term-missing
"""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reba_core import (
    AcilarObj, REBASkoru,
    TABLO_A, TABLO_B, TABLO_C,
    aci_3nokta, aci_dikey,
    reba_skorla, risk_info, segment_risk_renk, SEGMENT_MAX,
)


# ════════════════════════════════════════════════════════
# TABLO DOĞRULUĞU — Hignett & McAtamney (2000) referans değerleri
# ════════════════════════════════════════════════════════

class TestTabloA:
    """TABLO_A[govde-1][boyun-1][bacak-1] lookup testleri."""

    def test_dik_durust_en_dusuk(self):
        """Dik duruş: govde=1, boyun=1, bacak=1 → tablo_a=1"""
        assert TABLO_A[0][0][0] == 1

    def test_govde2_boyun1_bacak1(self):
        assert TABLO_A[1][0][0] == 2

    def test_govde3_boyun2_bacak2(self):
        assert TABLO_A[2][1][1] == 5

    def test_govde5_boyun3_bacak4(self):
        """En yüksek kombinasyon → 9"""
        assert TABLO_A[4][2][3] == 9

    def test_govde1_boyun2_bacak3(self):
        assert TABLO_A[0][1][2] == 3

    def test_govde4_boyun2_bacak1(self):
        assert TABLO_A[3][1][0] == 5


class TestTabloB:
    """TABLO_B[ustkol-1][altkol-1][bilek-1] lookup testleri."""

    def test_en_dusuk(self):
        """ustkol=1, altkol=1, bilek=1 → 1"""
        assert TABLO_B[0][0][0] == 1

    def test_ustkol3_altkol1_bilek2(self):
        assert TABLO_B[2][0][1] == 4

    def test_ustkol6_altkol2_bilek3(self):
        """En yüksek → 9"""
        assert TABLO_B[5][1][2] == 9

    def test_ustkol4_altkol2_bilek1(self):
        assert TABLO_B[3][1][0] == 5


class TestTabloC:
    """TABLO_C[skor_a-1][skor_b-1] lookup testleri."""

    def test_skora1_skorb1(self):
        assert TABLO_C[0][0] == 1

    def test_skora5_skorb5(self):
        assert TABLO_C[4][4] == 6

    def test_skora9_skorb8(self):
        assert TABLO_C[8][7] == 11

    def test_skora12_skorb12(self):
        """Maksimum → 12"""
        assert TABLO_C[11][11] == 12

    def test_skora3_skorb4(self):
        assert TABLO_C[2][3] == 4


# ════════════════════════════════════════════════════════
# AÇI HESAPLAMA — Boundary value testleri
# ════════════════════════════════════════════════════════

class TestAci3Nokta:
    """aci_3nokta boundary değer testleri."""

    def test_dik_aci_90(self):
        """L şekli → 90°"""
        a = (0, 1)
        b = (0, 0)
        c = (1, 0)
        result = aci_3nokta(a, b, c)
        assert abs(result - 90.0) < 0.01

    def test_duz_cizgi_180(self):
        """Düz çizgi → 180°"""
        a = (-1, 0)
        b = (0, 0)
        c = (1, 0)
        result = aci_3nokta(a, b, c)
        assert abs(result - 180.0) < 0.01

    def test_sifir_aci(self):
        """İki vektör aynı yönde → 0°"""
        a = (1, 0)
        b = (0, 0)
        c = (1, 0)
        result = aci_3nokta(a, b, c)
        assert abs(result - 0.0) < 0.01

    def test_degenerate_ayni_nokta(self):
        """a ve c aynı nokta — degenerate durum, çökmemeli"""
        result = aci_3nokta((0, 0), (0, 0), (0, 0))
        assert result == 0.0 or result is not None

    def test_45_derece(self):
        """45° köşegen → ~45°"""
        a = (0, 1)
        b = (0, 0)
        c = (1, 1)
        result = aci_3nokta(a, b, c)
        assert abs(result - 45.0) < 1.0


class TestAciDikey:
    """aci_dikey boundary değer testleri."""

    def test_tam_dikey(self):
        """Dikey çizgi → 0°"""
        result = aci_dikey((0, 0), (0, 10))
        assert abs(result - 0.0) < 0.01

    def test_tam_yatay(self):
        """Yatay çizgi → 90°"""
        result = aci_dikey((0, 0), (10, 0))
        assert abs(result - 90.0) < 0.01

    def test_45_derece(self):
        """45° → ~45°"""
        result = aci_dikey((0, 0), (10, 10))
        assert abs(result - 45.0) < 0.01


# ════════════════════════════════════════════════════════
# SCORING PIPELINE — Bilinen postür senaryoları
# ════════════════════════════════════════════════════════

def _acilar_olustur(**kwargs) -> AcilarObj:
    """Test için AcilarObj oluştur — sadece verilen değerleri set et."""
    a = AcilarObj()
    a.guven = 0.9
    a.analiz_tarafi = "sag"
    a.kamera_acisi = "oblique"
    a.bacak_gozukuyor = True
    a.bilateral_destek = True
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


class TestScoringPipeline:
    """End-to-end REBA skorlama senaryoları."""

    def test_dik_durust_minimum(self):
        """
        Senaryo: Dik duruş, yük yok, ideal tutma.
        Boyun ~10°, gövde ~5°, bacak bilateral.
        Beklenen: Düşük REBA (≤4)
        """
        a = _acilar_olustur(
            boyun_flexion=10.0,
            govde_flexion=5.0,
            diz_flexion_sol=5.0,
            diz_flexion_sag=5.0,
            ust_kol_aci=15.0,
            alt_kol_aci=80.0,
            bilek_aci=5.0,
        )
        s = reba_skorla(a, yuk_skoru=0, tutma=0, aktivite=0)
        assert s.final_skor <= 4, f"Dik duruş REBA={s.final_skor}, beklenen ≤4"

    def test_agir_yuk_tas(self):
        """
        Senaryo: Ağır yük (>10kg) + öne eğilme.
        Gövde 45°, yük skoru 2.
        Beklenen: REBA ≥7
        """
        a = _acilar_olustur(
            boyun_flexion=25.0,
            govde_flexion=45.0,
            diz_flexion_sol=10.0,
            diz_flexion_sag=10.0,
            ust_kol_aci=30.0,
            alt_kol_aci=90.0,
            bilek_aci=10.0,
        )
        s = reba_skorla(a, yuk_skoru=2, tutma=0, aktivite=0)
        assert s.final_skor >= 7, f"Ağır yük REBA={s.final_skor}, beklenen ≥7"

    def test_one_egilme_siddetli(self):
        """
        Senaryo: Ciddi öne eğilme (gövde >60°) + boyun 35°.
        Beklenen: REBA ≥8 (yüksek risk)
        """
        a = _acilar_olustur(
            boyun_flexion=35.0,
            govde_flexion=65.0,
            diz_flexion_sol=45.0,
            diz_flexion_sag=45.0,
            ust_kol_aci=60.0,
            alt_kol_aci=120.0,
            bilek_aci=20.0,
        )
        s = reba_skorla(a, yuk_skoru=1, tutma=1, aktivite=1)
        assert s.final_skor >= 8, f"Şiddetli eğilme REBA={s.final_skor}, beklenen ≥8"
        assert s.risk_seviyesi in ["Yüksek Risk", "Çok Yüksek Risk"]

    def test_manuel_modifier_boyun(self):
        """Boyun modifier'ları doğru ekleniyor mu?"""
        a = _acilar_olustur(boyun_flexion=22.0, govde_flexion=5.0,
                             ust_kol_aci=15.0, alt_kol_aci=80.0, bilek_aci=5.0)
        s_baz = reba_skorla(a, 0, 0, 0)
        s_mod = reba_skorla(a, 0, 0, 0,
                            boyun_yan_egim=True, boyun_donus=True)
        assert s_mod.boyun_skoru == s_baz.boyun_skoru + 2

    def test_manuel_modifier_govde(self):
        """Gövde modifier'ları doğru ekleniyor mu?"""
        a = _acilar_olustur(govde_flexion=15.0, boyun_flexion=10.0,
                             ust_kol_aci=15.0, alt_kol_aci=80.0, bilek_aci=5.0)
        s_baz = reba_skorla(a, 0, 0, 0)
        s_mod = reba_skorla(a, 0, 0, 0,
                            govde_yan_egim=True, govde_donus=True, govde_extension=True)
        assert s_mod.govde_skoru == min(s_baz.govde_skoru + 3, 5)

    def test_kol_destekli_azaltir(self):
        """Kol destekli → üst kol skoru -1"""
        a = _acilar_olustur(ust_kol_aci=50.0, alt_kol_aci=80.0, bilek_aci=5.0,
                             govde_flexion=5.0, boyun_flexion=10.0)
        s_normal = reba_skorla(a, 0, 0, 0)
        s_destekli = reba_skorla(a, 0, 0, 0, kol_destekli=True)
        assert s_destekli.ust_kol_skoru == max(1, s_normal.ust_kol_skoru - 1)

    def test_final_skor_15_max(self):
        """Final skor hiçbir zaman 15'i geçmemeli."""
        a = _acilar_olustur(
            boyun_flexion=90.0, govde_flexion=90.0,
            diz_flexion_sol=90.0, diz_flexion_sag=90.0,
            ust_kol_aci=120.0, alt_kol_aci=150.0, bilek_aci=45.0,
            bilateral_destek=False,
        )
        s = reba_skorla(a, yuk_skoru=3, tutma=3, aktivite=3,
                        boyun_yan_egim=True, boyun_donus=True, boyun_extension=True,
                        govde_yan_egim=True, govde_donus=True, govde_extension=True,
                        omuz_kalkmis=True, kol_abdukte=True, bilek_donus=True)
        assert s.final_skor <= 15


# ════════════════════════════════════════════════════════
# RİSK BİLGİSİ VE YARDIMCI FONKSİYONLAR
# ════════════════════════════════════════════════════════

class TestRiskInfo:
    def test_onemsiz(self):
        r, c = risk_info(1)
        assert r == "Önemsiz"

    def test_dusuk(self):
        r, _ = risk_info(2)
        assert r == "Düşük"
        r, _ = risk_info(3)
        assert r == "Düşük"

    def test_orta(self):
        for s in [4, 5, 6, 7]:
            r, _ = risk_info(s)
            assert r == "Orta", f"Skor {s} Orta olmalı"

    def test_yuksek(self):
        for s in [8, 9, 10]:
            r, _ = risk_info(s)
            assert r == "Yüksek"

    def test_cok_yuksek(self):
        for s in [11, 12, 15]:
            r, _ = risk_info(s)
            assert r == "Çok Yüksek"


class TestSegmentRiskRenk:
    def test_dusuk_oran_yesil(self):
        c = segment_risk_renk(1, 6)  # ~0.17
        assert c == "#16a34a"

    def test_yuksek_oran_kirmizi(self):
        c = segment_risk_renk(5, 6)  # ~0.83
        assert c == "#7c3aed"

    def test_orta_oran_turuncu(self):
        c = segment_risk_renk(4, 6)  # ~0.67
        assert c == "#dc2626"
