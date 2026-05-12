"""
REBA Ergonomi Analiz Ajanı v4.0
================================
İSG Uzmanları için Profesyonel REBA Değerlendirme Aracı
MediaPipe Pose + Manuel Girdi + PDF Rapor
"""

import streamlit as st
import cv2
import numpy as np
import math
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime
import json
import io

import mediapipe as mp

LM = mp.solutions.pose.PoseLandmark
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAYFA AYARI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="REBA Ergonomi Analizi",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    [[1,2,3,4],[1,2,3,4],[3,3,5,6]],
    [[2,3,4,5],[3,4,5,6],[4,5,6,7]],
    [[2,4,5,6],[4,5,6,7],[5,6,7,8]],
    [[3,5,6,7],[5,6,7,8],[6,7,8,9]],
    [[4,6,7,8],[6,7,8,9],[7,8,9,9]],
]

TABLE_B = [
    [[1,2,2],[1,2,3]],
    [[1,2,3],[2,3,4]],
    [[3,4,5],[4,5,5]],
    [[4,5,5],[5,6,7]],
    [[6,7,8],[7,8,8]],
    [[7,8,8],[8,9,9]],
]

TABLE_C = [
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
    )
    mid_hip = (
        (pt(LM.LEFT_HIP)[0] + pt(LM.RIGHT_HIP)[0]) / 2,
        (pt(LM.LEFT_HIP)[1] + pt(LM.RIGHT_HIP)[1]) / 2
    )

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
}
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BAŞLIK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<div class="reba-header">
    <div style="font-size:40px">🦺</div>
    <div>
        <h1>REBA Ergonomi Risk Analiz Ajanı</h1>
        <p>Rapid Entire Body Assessment | MediaPipe Pose AI | ISG Uzmanı Aracı | v4.0</p>
    </div>
</div>
""", unsafe_allow_html=True)

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
</div>
""", unsafe_allow_html=True)
