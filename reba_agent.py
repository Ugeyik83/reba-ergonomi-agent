"""
REBA Ergonomi Analiz Ajanı v3.2
================================
MediaPipe Pose + REBA Scoring Engine + Streamlit UI
"""

import streamlit as st
import cv2
import numpy as np
import math
import tempfile
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image
import io
import json
from datetime import datetime
import mediapipe as mp

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="REBA Ergonomi Analizi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# MediaPipe Pose landmark indices
# Ref: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LM = mp.solutions.pose.PoseLandmark
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. DATA CLASSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class BodyAngles:
    """Calculated body segment angles from pose landmarks."""
    neck_flexion: float = 0.0       # Boyun fleksiyonu (derece)
    neck_side_bend: float = 0.0     # Boyun lateral eğilme
    trunk_flexion: float = 0.0      # Gövde fleksiyonu
    trunk_side_bend: float = 0.0    # Gövde lateral eğilme
    trunk_twist: bool = False       # Gövde rotasyonu
    knee_flexion_left: float = 0.0  # Sol diz fleksiyonu
    knee_flexion_right: float = 0.0 # Sağ diz fleksiyonu
    upper_arm_angle: float = 0.0    # Üst kol açısı (gövdeden)
    shoulder_raised: bool = False   # Omuz yükselmesi
    arm_abducted: bool = False      # Kol abdüksiyonu
    lower_arm_angle: float = 0.0    # Dirsek fleksiyonu
    wrist_angle: float = 0.0        # Bilek açısı
    wrist_twist: bool = False       # Bilek rotasyonu
    bilateral_support: bool = True  # İki ayak üzerinde mi
    confidence: float = 0.0         # Pose tespiti güvenilirliği

@dataclass
class REBAResult:
    """Complete REBA assessment result."""
    # Individual scores
    neck_score: int = 0
    trunk_score: int = 0
    leg_score: int = 0
    upper_arm_score: int = 0
    lower_arm_score: int = 0
    wrist_score: int = 0
    # Group scores
    score_a: int = 0          # Table A result
    score_b: int = 0          # Table B result
    load_score: int = 0       # Yük/kuvvet skoru
    coupling_score: int = 0   # Kavrama skoru
    activity_score: int = 0   # Aktivite skoru
    score_c: int = 0          # Table C result
    final_score: int = 0      # Nihai REBA skoru
    risk_level: str = ""      # Risk seviyesi
    action: str = ""          # Aksiyon önerisi
    color: str = "#22c55e"    # Renk kodu
    angles: Optional[BodyAngles] = None
    frame_time: float = 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. ANGLE CALCULATION ENGINE (Trigonometric)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_angle_3p(a, b, c) -> float:
    """3 nokta arasındaki açıyı derece olarak hesapla.
    b = vertex (köşe noktası), a ve c = kol noktaları.
    arccos kullanır. Sonuç: 0-180 derece."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    
    dot = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    
    cos_angle = np.clip(dot / (mag_ba * mag_bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def calc_angle_vertical(a, b) -> float:
    """İki nokta arasındaki vektörün dikey eksenle açısı.
    a = üst nokta, b = alt nokta. Sonuç: 0 = dik, 90 = yatay."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]  # y aşağı doğru artar (piksel koordinatları)
    
    # Dikey eksenle açı
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return abs(angle)


def get_landmark_coords(landmarks, idx, w, h):
    """MediaPipe landmark'ı piksel koordinatlarına çevir."""
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h, lm.z * w, lm.visibility)


def calculate_body_angles(landmarks, img_w, img_h) -> BodyAngles:
    """MediaPipe landmark'larından tüm REBA açılarını hesapla."""
    
    angles = BodyAngles()
    
    # Helper: landmark → (x, y) pixel coords
    def pt(idx):
        lm = landmarks[idx]
        return (lm.x * img_w, lm.y * img_h)
    
    def vis(idx):
        return landmarks[idx].visibility
    
    # Ortalama güvenilirlik
    key_landmarks = [LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
                     LM.LEFT_HIP, LM.RIGHT_HIP, LM.LEFT_KNEE, LM.RIGHT_KNEE]
    angles.confidence = np.mean([vis(lm) for lm in key_landmarks])
    
    # ── Mid-points (orta noktalar) ──
    mid_shoulder = (
        (pt(LM.LEFT_SHOULDER)[0] + pt(LM.RIGHT_SHOULDER)[0]) / 2,
        (pt(LM.LEFT_SHOULDER)[1] + pt(LM.RIGHT_SHOULDER)[1]) / 2
    )
    mid_hip = (
        (pt(LM.LEFT_HIP)[0] + pt(LM.RIGHT_HIP)[0]) / 2,
        (pt(LM.LEFT_HIP)[1] + pt(LM.RIGHT_HIP)[1]) / 2
    )
    
    # ── BOYUN (Neck) ──
    # Burun → omuz orta noktası vektörünün dikey eksenle açısı
    nose = pt(LM.NOSE)
    neck_vec_angle = calc_angle_vertical(nose, mid_shoulder)
    # Boyun fleksiyonu: 0 = nötr, pozitif = öne eğik
    angles.neck_flexion = neck_vec_angle
    
    # Boyun lateral eğilme: kulaklar arası asimetri
    left_ear = pt(LM.LEFT_EAR)
    right_ear = pt(LM.RIGHT_EAR)
    ear_dy = abs(left_ear[1] - right_ear[1])
    ear_dx = abs(left_ear[0] - right_ear[0])
    if ear_dx > 0:
        angles.neck_side_bend = math.degrees(math.atan2(ear_dy, ear_dx))
    
    # ── GÖVDE (Trunk) ──
    # Omuz ortası → kalça ortası vektörünün dikey eksenle açısı
    trunk_angle = calc_angle_vertical(mid_shoulder, mid_hip)
    angles.trunk_flexion = trunk_angle
    
    # Gövde lateral eğilme: omuz ve kalça ortalarının x farkı
    lateral_offset = abs(mid_shoulder[0] - mid_hip[0])
    trunk_height = abs(mid_shoulder[1] - mid_hip[1])
    if trunk_height > 0:
        angles.trunk_side_bend = math.degrees(math.atan2(lateral_offset, trunk_height))
    
    # Gövde rotasyonu: omuz genişliği vs kalça genişliği oranı
    shoulder_width = abs(pt(LM.LEFT_SHOULDER)[0] - pt(LM.RIGHT_SHOULDER)[0])
    hip_width = abs(pt(LM.LEFT_HIP)[0] - pt(LM.RIGHT_HIP)[0])
    if hip_width > 0:
        twist_ratio = shoulder_width / hip_width
        angles.trunk_twist = twist_ratio < 0.7 or twist_ratio > 1.4
    
    # ── BACAKLAR (Legs) ──
    # Diz fleksiyonu: kalça-diz-ayak bileği açısı
    angles.knee_flexion_left = 180 - calc_angle_3p(
        pt(LM.LEFT_HIP), pt(LM.LEFT_KNEE), pt(LM.LEFT_ANKLE)
    )
    angles.knee_flexion_right = 180 - calc_angle_3p(
        pt(LM.RIGHT_HIP), pt(LM.RIGHT_KNEE), pt(LM.RIGHT_ANKLE)
    )
    
    # Bilateral destek: her iki ayak da görünür mü
    angles.bilateral_support = (vis(LM.LEFT_ANKLE) > 0.3 and vis(LM.RIGHT_ANKLE) > 0.3)
    
    # ── ÜST KOL (Upper Arm) ──
    # En kötü tarafı al (sağ veya sol)
    upper_arm_left = calc_angle_3p(
        pt(LM.LEFT_HIP), pt(LM.LEFT_SHOULDER), pt(LM.LEFT_ELBOW)
    )
    upper_arm_right = calc_angle_3p(
        pt(LM.RIGHT_HIP), pt(LM.RIGHT_SHOULDER), pt(LM.RIGHT_ELBOW)
    )
    angles.upper_arm_angle = max(upper_arm_left, upper_arm_right)
    
    # Omuz yükselmesi: omuz noktasının kulak seviyesine yakınlığı
    left_shoulder_ear_dist = abs(pt(LM.LEFT_SHOULDER)[1] - pt(LM.LEFT_EAR)[1])
    right_shoulder_ear_dist = abs(pt(LM.RIGHT_SHOULDER)[1] - pt(LM.RIGHT_EAR)[1])
    ref_dist = abs(pt(LM.LEFT_HIP)[1] - pt(LM.LEFT_SHOULDER)[1])
    if ref_dist > 0:
        angles.shoulder_raised = min(left_shoulder_ear_dist, right_shoulder_ear_dist) / ref_dist < 0.3
    
    # Kol abdüksiyonu: dirsek yatay pozisyonu omuzdan uzakta mı
    left_abd = abs(pt(LM.LEFT_ELBOW)[0] - pt(LM.LEFT_SHOULDER)[0])
    right_abd = abs(pt(LM.RIGHT_ELBOW)[0] - pt(LM.RIGHT_SHOULDER)[0])
    if shoulder_width > 0:
        angles.arm_abducted = max(left_abd, right_abd) / shoulder_width > 0.8
    
    # ── ALT KOL (Lower Arm) ──
    lower_arm_left = calc_angle_3p(
        pt(LM.LEFT_SHOULDER), pt(LM.LEFT_ELBOW), pt(LM.LEFT_WRIST)
    )
    lower_arm_right = calc_angle_3p(
        pt(LM.RIGHT_SHOULDER), pt(LM.RIGHT_ELBOW), pt(LM.RIGHT_WRIST)
    )
    # Dirsek fleksiyonu: 180 = düz kol, 90 = 90° bükülü
    angles.lower_arm_angle = min(lower_arm_left, lower_arm_right)
    
    # ── BİLEK (Wrist) ──
    # Bilek açısı: ön kol - bilek - orta parmak açısı
    wrist_left = abs(180 - calc_angle_3p(
        pt(LM.LEFT_ELBOW), pt(LM.LEFT_WRIST), pt(LM.LEFT_INDEX)
    ))
    wrist_right = abs(180 - calc_angle_3p(
        pt(LM.RIGHT_ELBOW), pt(LM.RIGHT_WRIST), pt(LM.RIGHT_INDEX)
    ))
    angles.wrist_angle = max(wrist_left, wrist_right)
    
    # Bilek twist: sol ve sağ bilek y pozisyonları farklı mı (pronasyon/supinasyon tahmini)
    wrist_y_diff = abs(pt(LM.LEFT_WRIST)[1] - pt(LM.RIGHT_WRIST)[1])
    if ref_dist > 0:
        angles.wrist_twist = wrist_y_diff / ref_dist > 0.15
    
    return angles

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. REBA SCORING ENGINE (Hignett & McAtamney, 2000)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# REBA Table A: Trunk × Neck × Legs
TABLE_A = [
    # Legs→  1    2    3    4
    #       Neck scores for trunk=1
    [[1, 2, 3, 4], [1, 2, 3, 4], [3, 3, 5, 6]],
    # trunk=2
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
    # trunk=3
    [[2, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    # trunk=4
    [[3, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    # trunk=5
    [[4, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
]

# REBA Table B: Upper Arm × Lower Arm × Wrist
TABLE_B = [
    # Wrist→ 1    2    3
    #       Lower arm scores for upper_arm=1
    [[1, 2, 2], [1, 2, 3]],
    # upper_arm=2
    [[1, 2, 3], [2, 3, 4]],
    # upper_arm=3
    [[3, 4, 5], [4, 5, 5]],
    # upper_arm=4
    [[4, 5, 5], [5, 6, 7]],
    # upper_arm=5
    [[6, 7, 8], [7, 8, 8]],
    # upper_arm=6
    [[7, 8, 8], [8, 9, 9]],
]

# REBA Table C: Score A × Score B
TABLE_C = [
    [1,  1,  1,  2,  3,  3,  4,  5,  6,  7,  7,  7],
    [1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  7,  8],
    [2,  3,  3,  3,  4,  5,  6,  7,  7,  8,  8,  8],
    [3,  4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9],
    [4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9,  9],
    [6,  6,  6,  7,  8,  8,  9,  9, 10, 10, 10, 10],
    [7,  7,  7,  8,  9,  9,  9, 10, 10, 11, 11, 11],
    [8,  8,  8,  9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]


def score_reba(angles: BodyAngles, load_kg: float = 0, coupling: int = 0,
               activity: int = 0, load_shock: bool = False) -> REBAResult:
    """REBA puanlama. Açı verilerinden nihai skoru hesaplar."""
    
    result = REBAResult()
    result.angles = angles
    
    # ── BOYUN SKORU ──
    if angles.neck_flexion <= 20:
        result.neck_score = 1
    elif angles.neck_flexion <= 40:
        result.neck_score = 2
    else:
        result.neck_score = 3
    # Yana eğilme modifikatörü
    if angles.neck_side_bend > 15:
        result.neck_score += 1
    result.neck_score = min(result.neck_score, 6)
    
    # ── GÖVDE SKORU ──
    if angles.trunk_flexion <= 5:
        result.trunk_score = 1  # Dik
    elif angles.trunk_flexion <= 20:
        result.trunk_score = 2
    elif angles.trunk_flexion <= 60:
        result.trunk_score = 3
    else:
        result.trunk_score = 4
    # Yana eğilme veya rotasyon
    if angles.trunk_side_bend > 10:
        result.trunk_score += 1
    if angles.trunk_twist:
        result.trunk_score += 1
    result.trunk_score = min(result.trunk_score, 5)
    
    # ── BACAK SKORU ──
    if angles.bilateral_support:
        result.leg_score = 1
    else:
        result.leg_score = 2
    knee_flex = max(angles.knee_flexion_left, angles.knee_flexion_right)
    if 30 <= knee_flex < 60:
        result.leg_score += 1
    elif knee_flex >= 60:
        result.leg_score += 2
    result.leg_score = min(result.leg_score, 4)
    
    # ── TABLE A ──
    t_idx = min(result.trunk_score - 1, 4)
    n_idx = min(result.neck_score - 1, 2)
    l_idx = min(result.leg_score - 1, 3)
    result.score_a = TABLE_A[t_idx][n_idx][l_idx]
    
    # Yük/kuvvet skoru
    if load_kg < 5:
        result.load_score = 0
    elif load_kg < 10:
        result.load_score = 1
    else:
        result.load_score = 2
    if load_shock:
        result.load_score += 1
    result.score_a += result.load_score
    
    # ── ÜST KOL SKORU ──
    if angles.upper_arm_angle <= 20:
        result.upper_arm_score = 1
    elif angles.upper_arm_angle <= 45:
        result.upper_arm_score = 2
    elif angles.upper_arm_angle <= 90:
        result.upper_arm_score = 3
    else:
        result.upper_arm_score = 4
    if angles.shoulder_raised:
        result.upper_arm_score += 1
    if angles.arm_abducted:
        result.upper_arm_score += 1
    result.upper_arm_score = max(1, min(result.upper_arm_score, 6))
    
    # ── ALT KOL SKORU ──
    if 60 <= angles.lower_arm_angle <= 100:
        result.lower_arm_score = 1
    else:
        result.lower_arm_score = 2
    
    # ── BİLEK SKORU ──
    if angles.wrist_angle <= 15:
        result.wrist_score = 1
    else:
        result.wrist_score = 2
    if angles.wrist_twist:
        result.wrist_score += 1
    result.wrist_score = min(result.wrist_score, 3)
    
    # ── TABLE B ──
    u_idx = min(result.upper_arm_score - 1, 5)
    la_idx = min(result.lower_arm_score - 1, 1)
    w_idx = min(result.wrist_score - 1, 2)
    result.score_b = TABLE_B[u_idx][la_idx][w_idx]
    result.coupling_score = coupling
    result.score_b += coupling
    
    # ── TABLE C ──
    ca_idx = min(result.score_a - 1, 11)
    cb_idx = min(result.score_b - 1, 11)
    result.score_c = TABLE_C[ca_idx][cb_idx]
    result.activity_score = activity
    result.final_score = min(result.score_c + activity, 15)
    
    # ── RİSK SEVİYESİ ──
    s = result.final_score
    if s == 1:
        result.risk_level = "Göz ardı edilebilir"
        result.color = "#22c55e"
        result.action = "Aksiyon gerekmez"
    elif s <= 3:
        result.risk_level = "Düşük"
        result.color = "#86efac"
        result.action = "Değişiklik gerekebilir"
    elif s <= 7:
        result.risk_level = "Orta"
        result.color = "#fbbf24"
        result.action = "İnceleme ve değişiklik gerekli"
    elif s <= 10:
        result.risk_level = "Yüksek"
        result.color = "#f97316"
        result.action = "Acil inceleme ve değişiklik gerekli"
    else:
        result.risk_level = "Çok Yüksek"
        result.color = "#ef4444"
        result.action = "Hemen müdahale gerekli"
    
    return result

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. VIDEO/IMAGE PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_skeleton(image, landmarks, reba_score: int, frame_time: float):
    """Görüntü üzerine iskelet, açılar ve skor overlay'i çiz."""
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # Renk: skora göre
    if reba_score <= 3:
        color = (34, 197, 94)    # Yeşil
    elif reba_score <= 7:
        color = (251, 191, 36)   # Sarı
    elif reba_score <= 10:
        color = (249, 115, 22)   # Turuncu
    else:
        color = (239, 68, 68)    # Kırmızı
    
    # BGR dönüşümü
    bgr_color = (color[2], color[1], color[0])
    
    # Bağlantı çizgileri
    for connection in POSE_CONNECTIONS:
        start_lm = landmarks[connection[0]]
        end_lm = landmarks[connection[1]]
        if start_lm.visibility > 0.3 and end_lm.visibility > 0.3:
            start_pt = (int(start_lm.x * w), int(start_lm.y * h))
            end_pt = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(overlay, start_pt, end_pt, bgr_color, 2, cv2.LINE_AA)
    
    # Eklem noktaları
    for lm in landmarks:
        if lm.visibility > 0.3:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(overlay, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, pt, 4, bgr_color, 1, cv2.LINE_AA)
    
    # Skor overlay (sol üst)
    cv2.rectangle(overlay, (10, 10), (200, 70), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (200, 70), bgr_color, 2)
    cv2.putText(overlay, f"REBA: {reba_score}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr_color, 2, cv2.LINE_AA)
    cv2.putText(overlay, f"t={frame_time:.1f}s", (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Blend overlay
    result = cv2.addWeighted(overlay, 0.85, image, 0.15, 0)
    return result


def process_frames(frames: list, fps: float, load_kg: float, coupling: int,
                   activity: int, progress_bar, status_text) -> Tuple[List[REBAResult], list]:
    """Kare listesini MediaPipe ile analiz et, REBA skorlarını hesapla."""
    
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,         # Her kare bağımsız
        model_complexity=1,             # orta - default, indirme gerekmez
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    results_list = []
    annotated_frames = []
    
    for i, (frame, frame_time) in enumerate(frames):
        progress_bar.progress((i + 1) / len(frames))
        status_text.text(f"Kare {i+1}/{len(frames)} analiz ediliyor... (t={frame_time:.1f}s)")
        
        # RGB dönüşümü (MediaPipe RGB bekler)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)
        
        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            # Açı hesaplama
            angles = calculate_body_angles(landmarks, w, h)
            
            # REBA skorlama
            reba = score_reba(angles, load_kg, coupling, activity)
            reba.frame_time = frame_time
            results_list.append(reba)
            
            # İskelet çizimi
            annotated = draw_skeleton(frame.copy(), landmarks, reba.final_score, frame_time)
            annotated_frames.append((annotated, reba))
        else:
            # Kişi tespit edilemedi → orijinal kare
            annotated_frames.append((frame.copy(), None))
    
    pose.close()
    return results_list, annotated_frames


def extract_video_frames(video_path: str, sample_fps: int = 5, max_duration: float = 15.0):
    """Videodan belirli FPS ile kare örnekleme."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = min(total_frames / orig_fps, max_duration)
    
    # Hangi kareleri alacağız
    frame_interval = max(1, int(orig_fps / sample_fps))
    
    frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_idx / orig_fps
        if current_time > max_duration:
            break
        
        if frame_idx % frame_interval == 0:
            # Boyut optimizasyonu (max 720p)
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))
            frames.append((frame, current_time))
        
        frame_idx += 1
    
    cap.release()
    return frames, orig_fps

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. STREAMLIT UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&display=swap');
    
    .stApp {
        background-color: #020817;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .main-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 16px 0 24px 0;
    }
    
    .header-icon {
        width: 48px; height: 48px; border-radius: 12px;
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        display: flex; align-items: center; justify-content: center;
        font-size: 24px; font-weight: 700; color: white;
    }
    
    .header-text h1 {
        margin: 0; font-size: 22px; font-weight: 800;
        color: #f1f5f9; letter-spacing: -0.5px;
    }
    
    .header-text p {
        margin: 0; font-size: 11px; color: #475569;
        letter-spacing: 0.12em;
    }
    
    .score-card {
        background: #0f172a; border-radius: 16px; padding: 24px;
        border: 1px solid #1e293b; margin: 8px 0;
    }
    
    .risk-badge {
        display: inline-block; padding: 6px 16px; border-radius: 20px;
        font-weight: 700; font-size: 13px; letter-spacing: 0.05em;
    }
    
    .metric-grid {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
    }
    
    .metric-box {
        background: #020817; border-radius: 10px; padding: 14px;
        text-align: center; border: 1px solid #1e293b;
    }
    
    .metric-box .icon { font-size: 22px; }
    .metric-box .label { font-size: 10px; color: #475569; margin-top: 4px; letter-spacing: 0.05em; }
    .metric-box .value { font-size: 20px; font-weight: 800; margin-top: 2px; }
    
    .footer-note {
        text-align: center; font-size: 10px; color: #1e293b;
        line-height: 2; margin-top: 24px;
    }
    
    /* Streamlit overrides */
    .stFileUploader > div { border-color: #1e293b !important; }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #94a3b8 !important; font-size: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="header-icon">R</div>
    <div class="header-text">
        <h1>REBA Ergonomi Analizi</h1>
        <p>RAPID ENTIRE BODY ASSESSMENT · MEDIAPIPE POSE · v3</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Parametreler ──
with st.sidebar:
    st.markdown("### ⚙️ Analiz Parametreleri")
    
    sample_fps = st.slider("Örnekleme FPS", 2, 10, 5,
                           help="Saniyede kaç kare analiz edilecek")
    
    load_kg = st.number_input("Yük (kg)", 0.0, 50.0, 0.0, 1.0,
                              help="Taşınan/tutulan yük ağırlığı")
    
    coupling = st.selectbox("Kavrama Kalitesi", 
                            options=[0, 1, 2, 3],
                            format_func=lambda x: ["İyi (0)", "Orta (1)", "Kötü (2)", "Kabul edilemez (3)"][x],
                            help="Obje kavrama kalitesi")
    
    activity = st.selectbox("Aktivite Skoru",
                            options=[0, 1, 2, 3],
                            format_func=lambda x: [
                                "Statik >1dk veya tekrarsız (0)",
                                "Tekrarlı hareket (1)",
                                "Hızlı geniş açılı hareketler (2)",
                                "Dengesiz/kararsız pozisyon (3)"
                            ][x],
                            help="Hareket tipi")
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px; color:#475569; line-height:1.8">
    <strong>REBA Skor Tablosu</strong><br>
    1 → Göz ardı edilebilir<br>
    2-3 → Düşük risk<br>
    4-7 → Orta risk<br>
    8-10 → Yüksek risk<br>
    11-15 → Çok yüksek risk
    </div>
    """, unsafe_allow_html=True)

# ── Ana Alan: Dosya Yükleme ──
uploaded = st.file_uploader(
    "📹 Video veya 📸 Görüntü Yükle",
    type=["mp4", "mov", "m4v", "webm", "avi", "mkv",
          "jpg", "jpeg", "png", "webp", "bmp"],
    help="MP4, MOV, WEBM, AVI | JPG, PNG, WEBP — Maks. 15 sn video"
)

if uploaded is not None:
    file_ext = uploaded.name.lower().split(".")[-1]
    is_video = file_ext in ["mp4", "mov", "m4v", "webm", "avi", "mkv", "3gp"]
    is_image = file_ext in ["jpg", "jpeg", "png", "webp", "bmp", "heic"]
    
    if is_video:
        # ── VIDEO İŞLEME ──
        st.markdown(f"**📹 Video:** `{uploaded.name}`")
        
        # Geçici dosyaya yaz (OpenCV file path istiyor)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        
        try:
            # Kare çıkarma
            progress = st.progress(0)
            status = st.empty()
            status.text("Kareler çıkarılıyor...")
            
            frames, orig_fps = extract_video_frames(tmp_path, sample_fps)
            
            if not frames:
                st.error("❌ Video okunamadı. Farklı bir format deneyin veya ekran görüntüsü yükleyin.")
            else:
                st.info(f"📊 {len(frames)} kare çıkarıldı (orijinal FPS: {orig_fps:.0f})")
                
                # Analiz
                reba_results, annotated = process_frames(
                    frames, orig_fps, load_kg, coupling, activity, progress, status
                )
                
                progress.empty()
                status.empty()
                
                if not reba_results:
                    st.error("❌ Hiçbir karede kişi tespit edilemedi. Kameranın kişiyi tam gösterdiğinden emin olun.")
                else:
                    # ── SONUÇLAR ──
                    worst = max(reba_results, key=lambda r: r.final_score)
                    avg_score = sum(r.final_score for r in reba_results) / len(reba_results)
                    
                    # Ana skor kartı
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="score-card" style="border-color: {worst.color}33">
                            <div style="font-size:10px; color:#475569; letter-spacing:0.12em; margin-bottom:4px">
                                EN YÜKSEK RİSK SKORU
                            </div>
                            <div style="font-size:56px; font-weight:800; color:{worst.color}; line-height:1">
                                {worst.final_score}
                            </div>
                            <div style="font-size:11px; color:#475569">/15</div>
                            <div class="risk-badge" style="background:{worst.color}22; color:{worst.color}; margin-top:12px">
                                {worst.risk_level}
                            </div>
                            <div style="font-size:12px; color:#64748b; margin-top:8px">
                                → {worst.action}
                            </div>
                            <div style="margin-top:12px; font-size:11px; color:#475569">
                                Kare zamanı: {worst.frame_time:.1f}s · 
                                Ortalama: {avg_score:.1f} · 
                                Güven: {worst.angles.confidence:.0%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="score-card">
                            <div style="font-size:10px; color:#475569; letter-spacing:0.12em; margin-bottom:14px">
                                SEGMENT DETAYI · EN KÖTÜ KARE
                            </div>
                            <div class="metric-grid">
                                <div class="metric-box">
                                    <div class="icon">🔝</div>
                                    <div class="label">Boyun</div>
                                    <div class="value" style="color:{'#22c55e' if worst.neck_score <= 1 else '#fbbf24' if worst.neck_score <= 2 else '#ef4444'}">{worst.neck_score}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="icon">🧍</div>
                                    <div class="label">Gövde</div>
                                    <div class="value" style="color:{'#22c55e' if worst.trunk_score <= 2 else '#fbbf24' if worst.trunk_score <= 3 else '#ef4444'}">{worst.trunk_score}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="icon">🦵</div>
                                    <div class="label">Bacak</div>
                                    <div class="value" style="color:{'#22c55e' if worst.leg_score <= 1 else '#fbbf24' if worst.leg_score <= 2 else '#ef4444'}">{worst.leg_score}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="icon">💪</div>
                                    <div class="label">Üst Kol</div>
                                    <div class="value" style="color:{'#22c55e' if worst.upper_arm_score <= 2 else '#fbbf24' if worst.upper_arm_score <= 3 else '#ef4444'}">{worst.upper_arm_score}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="icon">🦾</div>
                                    <div class="label">Alt Kol</div>
                                    <div class="value" style="color:{'#22c55e' if worst.lower_arm_score <= 1 else '#ef4444'}">{worst.lower_arm_score}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="icon">🤚</div>
                                    <div class="label">Bilek</div>
                                    <div class="value" style="color:{'#22c55e' if worst.wrist_score <= 1 else '#fbbf24' if worst.wrist_score <= 2 else '#ef4444'}">{worst.wrist_score}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Zaman çizelgesi
                    st.markdown("---")
                    st.markdown(f"""
                    <div style="font-size:10px; color:#475569; letter-spacing:0.12em; margin-bottom:8px">
                        ZAMAN ÇİZELGESİ · {len(reba_results)} KARE
                    </div>
                    """, unsafe_allow_html=True)
                    
                    timeline_cols = st.columns(min(len(reba_results), 10))
                    for i, r in enumerate(reba_results[:10]):
                        with timeline_cols[i]:
                            st.markdown(f"""
                            <div style="background:#0f172a; border-radius:10px; padding:10px;
                                        border:1px solid {r.color if r == worst else '#1e293b'};
                                        text-align:center">
                                <div style="font-size:20px; font-weight:800; color:{r.color}">{r.final_score}</div>
                                <div style="font-size:9px; color:#475569">{r.frame_time:.1f}s</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # İskelet overlay'li kare gösterimi
                    st.markdown("---")
                    st.markdown("""
                    <div style="font-size:10px; color:#475569; letter-spacing:0.12em; margin-bottom:8px">
                        İSKELET OVERLAY
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # En kötü kareyi göster
                    worst_idx = reba_results.index(worst)
                    for i, (ann_frame, reba) in enumerate(annotated):
                        if reba and reba.frame_time == worst.frame_time:
                            rgb_frame = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
                            st.image(rgb_frame, caption=f"En riskli kare (t={worst.frame_time:.1f}s, REBA={worst.final_score})",
                                     use_container_width=True)
                            break
                    
                    # Ham açı verileri
                    with st.expander("📐 Ham Açı Verileri"):
                        if worst.angles:
                            angle_data = {
                                "Boyun fleksiyonu": f"{worst.angles.neck_flexion:.1f}°",
                                "Boyun yan eğilme": f"{worst.angles.neck_side_bend:.1f}°",
                                "Gövde fleksiyonu": f"{worst.angles.trunk_flexion:.1f}°",
                                "Gövde yan eğilme": f"{worst.angles.trunk_side_bend:.1f}°",
                                "Gövde rotasyonu": "Evet" if worst.angles.trunk_twist else "Hayır",
                                "Sol diz fleksiyonu": f"{worst.angles.knee_flexion_left:.1f}°",
                                "Sağ diz fleksiyonu": f"{worst.angles.knee_flexion_right:.1f}°",
                                "Üst kol açısı": f"{worst.angles.upper_arm_angle:.1f}°",
                                "Omuz yükselmesi": "Evet" if worst.angles.shoulder_raised else "Hayır",
                                "Kol abdüksiyonu": "Evet" if worst.angles.arm_abducted else "Hayır",
                                "Dirsek fleksiyonu": f"{worst.angles.lower_arm_angle:.1f}°",
                                "Bilek açısı": f"{worst.angles.wrist_angle:.1f}°",
                                "Bilateral destek": "Evet" if worst.angles.bilateral_support else "Hayır",
                                "Pose güvenilirliği": f"{worst.angles.confidence:.0%}",
                            }
                            for k, v in angle_data.items():
                                st.text(f"{k}: {v}")
                    
                    # JSON export
                    with st.expander("📋 JSON Export"):
                        export = {
                            "dosya": uploaded.name,
                            "tarih": datetime.now().isoformat(),
                            "parametreler": {
                                "yuk_kg": load_kg,
                                "kavrama": coupling,
                                "aktivite": activity
                            },
                            "en_yuksek_skor": worst.final_score,
                            "risk_seviyesi": worst.risk_level,
                            "ortalama_skor": round(avg_score, 1),
                            "kare_sayisi": len(reba_results),
                            "skorlar": [{"zaman": r.frame_time, "skor": r.final_score} for r in reba_results]
                        }
                        st.json(export)
                        st.download_button(
                            "⬇️ JSON İndir",
                            json.dumps(export, ensure_ascii=False, indent=2),
                            f"reba_{uploaded.name.split('.')[0]}.json",
                            "application/json"
                        )
        
        finally:
            os.unlink(tmp_path)
    
    elif is_image:
        # ── GÖRÜNTÜ İŞLEME ──
        st.markdown(f"**📸 Görüntü:** `{uploaded.name}`")
        
        # Görüntüyü oku
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("❌ Görüntü okunamadı.")
        else:
            # Boyut optimizasyonu
            h, w = img.shape[:2]
            if w > 1280:
                scale = 1280 / w
                img = cv2.resize(img, (1280, int(h * scale)))
            
            progress = st.progress(0)
            status = st.empty()
            
            frames = [(img, 0.0)]
            reba_results, annotated = process_frames(
                frames, 1, load_kg, coupling, activity, progress, status
            )
            
            progress.empty()
            status.empty()
            
            if not reba_results:
                st.error("❌ Görüntüde kişi tespit edilemedi. Kişinin tam göründüğü bir kare deneyin.")
            else:
                result = reba_results[0]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # İskelet overlay
                    if annotated and annotated[0][1]:
                        rgb_frame = cv2.cvtColor(annotated[0][0], cv2.COLOR_BGR2RGB)
                        st.image(rgb_frame, use_container_width=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="score-card" style="border-color: {result.color}33">
                        <div style="font-size:10px; color:#475569; letter-spacing:0.12em; margin-bottom:4px">
                            REBA SKORU
                        </div>
                        <div style="font-size:64px; font-weight:800; color:{result.color}; line-height:1">
                            {result.final_score}
                        </div>
                        <div style="font-size:11px; color:#475569">/15</div>
                        <div class="risk-badge" style="background:{result.color}22; color:{result.color}; margin-top:12px">
                            {result.risk_level}
                        </div>
                        <div style="font-size:12px; color:#64748b; margin-top:8px">
                            → {result.action}
                        </div>
                        <div style="margin-top:12px; font-size:11px; color:#475569">
                            Güven: {result.angles.confidence:.0%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Segment detayı
                st.markdown(f"""
                <div class="score-card">
                    <div style="font-size:10px; color:#475569; letter-spacing:0.12em; margin-bottom:14px">
                        SEGMENT DETAYI
                    </div>
                    <div class="metric-grid">
                        <div class="metric-box">
                            <div class="icon">🔝</div>
                            <div class="label">Boyun</div>
                            <div class="value" style="color:{'#22c55e' if result.neck_score <= 1 else '#fbbf24' if result.neck_score <= 2 else '#ef4444'}">{result.neck_score}</div>
                        </div>
                        <div class="metric-box">
                            <div class="icon">🧍</div>
                            <div class="label">Gövde</div>
                            <div class="value" style="color:{'#22c55e' if result.trunk_score <= 2 else '#fbbf24' if result.trunk_score <= 3 else '#ef4444'}">{result.trunk_score}</div>
                        </div>
                        <div class="metric-box">
                            <div class="icon">🦵</div>
                            <div class="label">Bacak</div>
                            <div class="value" style="color:{'#22c55e' if result.leg_score <= 1 else '#fbbf24' if result.leg_score <= 2 else '#ef4444'}">{result.leg_score}</div>
                        </div>
                        <div class="metric-box">
                            <div class="icon">💪</div>
                            <div class="label">Üst Kol</div>
                            <div class="value" style="color:{'#22c55e' if result.upper_arm_score <= 2 else '#fbbf24' if result.upper_arm_score <= 3 else '#ef4444'}">{result.upper_arm_score}</div>
                        </div>
                        <div class="metric-box">
                            <div class="icon">🦾</div>
                            <div class="label">Alt Kol</div>
                            <div class="value" style="color:{'#22c55e' if result.lower_arm_score <= 1 else '#ef4444'}">{result.lower_arm_score}</div>
                        </div>
                        <div class="metric-box">
                            <div class="icon">🤚</div>
                            <div class="label">Bilek</div>
                            <div class="value" style="color:{'#22c55e' if result.wrist_score <= 1 else '#fbbf24' if result.wrist_score <= 2 else '#ef4444'}">{result.wrist_score}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Açı verileri
                with st.expander("📐 Ham Açı Verileri"):
                    a = result.angles
                    for label, val in [
                        ("Boyun fleksiyonu", f"{a.neck_flexion:.1f}°"),
                        ("Gövde fleksiyonu", f"{a.trunk_flexion:.1f}°"),
                        ("Üst kol açısı", f"{a.upper_arm_angle:.1f}°"),
                        ("Dirsek fleksiyonu", f"{a.lower_arm_angle:.1f}°"),
                        ("Bilek açısı", f"{a.wrist_angle:.1f}°"),
                        ("Sol diz", f"{a.knee_flexion_left:.1f}°"),
                        ("Sağ diz", f"{a.knee_flexion_right:.1f}°"),
                    ]:
                        st.text(f"{label}: {val}")

# Footer
st.markdown("""
<div class="footer-note">
    REBA (Hignett & McAtamney, 2000) · MediaPipe Pose (Google) · Skor 1–15<br>
    Açı doğruluğu kamera açısına ve görüntü kalitesine bağlıdır (±3-5°)<br>
    Profesyonel ergonomi değerlendirmesinin yerini tutmaz
</div>
""", unsafe_allow_html=True)
