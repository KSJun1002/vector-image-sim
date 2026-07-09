# ==============================================================================
# [기하 연계 심화 탐구 프로그램 - app.py]
# 주제: 벡터의 내적을 이용한 픽셀 패턴 유사도 측정 및 3D 기하학적 공간 시각화
# 프레임워크: Streamlit
# 특징: 사용자가 클릭해서 그린 8x8 픽셀(64차원 벡터)의 내적과 코사인 유사도, 사잇각을 실시간 계산
# ==============================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 🚨 Matplotlib 백엔드 고정 (서버 내 충돌 및 경고 방지)
import matplotlib
matplotlib.use('Agg')

# ------------------------------------------------------------------------------
# 1. 웹 레이아웃 설정
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Vector Dot Product & Image Similarity", layout="wide")

st.title("📐 기하 심화 연계: 벡터 내적 기반 이미지 유사도 분석기")
st.markdown("""
이 프로그램은 **기하 교과서의 '평면벡터의 내적과 크기, 사잇각'** 개념이 인공지능 이미지/얼굴 인식 기술에서 어떻게 활용되는지 시각적으로 보여줍니다.
좌측과 우측의 $8 \\times 8$ 격자판(64차원 벡터 공간)에 마우스로 클릭하여 패턴을 그려보세요. 두 이미지 벡터의 내적과 크기, $\\cos\\theta$ 값이 실시간으로 계산됩니다.
""")
st.divider()

# Session State를 이용해 사용자가 클릭한 격자판 데이터를 유지합니다. (8x8 크기)
if 'grid_A' not in st.session_state:
    st.session_state.grid_A = np.zeros((8, 8), dtype=int)
if 'grid_B' not in st.session_state:
    st.session_state.grid_B = np.zeros((8, 8), dtype=int)

# ------------------------------------------------------------------------------
# 2. 이미지 격자 드로잉 인터페이스 구축
# ------------------------------------------------------------------------------
col_input1, col_input2 = st.columns(2)

with col_input1:
    st.subheader("🖼️ 이미지 벡터 A (Vector A)")
    st.caption("격자판 칸을 클릭하여 패턴을 채워보세요. (주황색: 활성화)")
    
    # Grid A 인터랙티브 생성
    grid_A_new = np.zeros((8, 8), dtype=int)
    for r in range(8):
        cols = st.columns(8)
        for c in range(8):
            with cols[c]:
                # 이전 클릭 상태에 따른 버튼 색상 시각화
                val = st.session_state.grid_A[r, c]
                btn_label = "⬛" if val == 0 else "🟧"
                
                # 버튼을 클릭하면 상태 반전
                if st.button(btn_label, key=f"A_{r}_{c}"):
                    st.session_state.grid_A[r, c] = 1 - val
                    st.rerun()

with col_input2:
    st.subheader("🖼️ 이미지 벡터 B (Vector B)")
    st.caption("격자판 칸을 클릭하여 패턴을 채워보세요. (파란색: 활성화)")
    
    # Grid B 인터랙티브 생성
    grid_B_new = np.zeros((8, 8), dtype=int)
    for r in range(8):
        cols = st.columns(8)
        for c in range(8):
            with cols[c]:
                val = st.session_state.grid_B[r, c]
                btn_label = "⬛" if val == 0 else "🟦"
                
                if st.button(btn_label, key=f"B_{r}_{c}"):
                    st.session_state.grid_B[r, c] = 1 - val
                    st.rerun()

# 격자 리셋 버튼 레이아웃
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🔄 Vector A 초기화"):
        st.session_state.grid_A = np.zeros((8, 8), dtype=int)
        st.rerun()
with col_btn2:
    if st.button("🔄 Vector B 초기화"):
        st.session_state.grid_B = np.zeros((8, 8), dtype=int)
        st.rerun()

st.divider()

# ------------------------------------------------------------------------------
# 3. 64차원 벡터 공간 내적 및 유사도 연산
# ------------------------------------------------------------------------------
# 8x8 행렬 데이터를 64차원의 1차원 벡터로 일렬 전개(Flatten)
vec_A = st.session_state.grid_A.flatten()
vec_B = st.session_state.grid_B.flatten()

# 벡터 크기 및 내적 계산 (기하 교과 과정 공식 반영)
dot_product = float(np.dot(vec_A, vec_B))
magnitude_A = float(np.linalg.norm(vec_A))
magnitude_B = float(np.linalg.norm(vec_B))

# 코사인 유사도 및 사잇각(theta) 산출
cosine_similarity = 0.0
theta_deg = 90.0

if magnitude_A > 0 and magnitude_B > 0:
    cosine_similarity = dot_product / (magnitude_A * magnitude_B)
    # 수치적 오차 방지용 가드 처리 [-1, 1]
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    theta_rad = np.arccos(cosine_similarity)
    theta_deg = np.degrees(theta_rad)

# ------------------------------------------------------------------------------
# 4. 실시간 수학적 수치 분석 대시보드
# ------------------------------------------------------------------------------
st.subheader("📊 기하학적 정량 데이터 연산 결과")

col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

with col_metric1:
    st.metric(label="1. 벡터 내적 (Vector Dot Product: A • B)", value=f"{dot_product:.1f}")
    st.caption("내적은 같은 자리에 픽셀이 동시에 채워진 개수입니다.")

with col_metric2:
    st.metric(label="2. 벡터 A의 크기 (|A|)", value=f"{magnitude_A:.3f}")
    st.caption("A 격자판에 채워진 총 픽셀 개수의 제곱근입니다.")

with col_metric3:
    st.metric(label="3. 코사인 유사도 (Cos θ)", value=f"{cosine_similarity:.4f}")
    st.caption("1에 가까울수록 같고, 0이면 전혀 다른 패턴입니다.")

with col_metric4:
    st.metric(label="4. 벡터의 사잇각 (Angle θ)", value=f"{theta_deg:.1f}°")
    st.caption("두 이미지 패턴 벡터가 이루는 가상 공간 속 각도입니다.")

# ------------------------------------------------------------------------------
# 5. 시각화 피드백 시스템 (기하학적 사잇각 3D 벡터 투영)
# ------------------------------------------------------------------------------
st.divider()
st.subheader("📐 벡터의 기하학적 사잇각 3D 시각화")
st.markdown("수만 차원의 고차원 공간을 눈으로 볼 수 없기 때문에, 두 벡터가 이루는 평면을 **3차원 공간 상으로 투영하여 사잇각 $\\theta$의 기하학적 벌어짐**을 표현했습니다.")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 원점 세팅
ax.quiver(0, 0, 0, 1, 0, 0, color='gray', linestyle='--', alpha=0.3)
ax.quiver(0, 0, 0, 0, 1, 0, color='gray', linestyle='--', alpha=0.3)
ax.quiver(0, 0, 0, 0, 0, 1, color='gray', linestyle='--', alpha=0.3)

# 두 벡터가 이루는 평면 상의 3차원 투영 벡터 계산
# 벡터 A를 x축 위에 고정시키고, 벡터 B를 두 벡터의 사잇각 θ 만큼 y축 방향으로 회전시켜 기하학적으로 배치함
vector_3d_A = np.array([magnitude_A, 0.0, 0.0]) if magnitude_A > 0 else np.array([0.0, 0.0, 0.0])
vector_3d_B = np.array([magnitude_B * cosine_similarity, magnitude_B * np.sin(np.radians(theta_deg)), 0.0]) if magnitude_B > 0 else np.array([0.0, 0.0, 0.0])

# 3D 화살표 그리기
if magnitude_A > 0:
    ax.quiver(0, 0, 0, vector_3d_A[0], vector_3d_A[1], vector_3d_A[2], 
              color='#e67e22', arrow_length_ratio=0.1, linewidth=3, label=f'Vector A (Mag: {magnitude_A:.2f})')
if magnitude_B > 0:
    ax.quiver(0, 0, 0, vector_3d_B[0], vector_3d_B[1], vector_3d_B[2], 
              color='#3498db', arrow_length_ratio=0.1, linewidth=3, label=f'Vector B (Mag: {magnitude_B:.2f})')

# 사잇각 호(Arc) 및 텍스트 표시
if magnitude_A > 0 and magnitude_B > 0 and theta_deg > 0:
    # 사잇각 표현을 위한 곡선 좌표 생성
    angle_steps = np.linspace(0, np.radians(theta_deg), 50)
    r_arc = min(magnitude_A, magnitude_B) * 0.4  # 벡터 크기에 비례하는 부채꼴 반지름
    arc_x = r_arc * np.cos(angle_steps)
    arc_y = r_arc * np.sin(angle_steps)
    arc_z = np.zeros_like(angle_steps)
    ax.plot(arc_x, arc_y, arc_z, color='#27ae60', linewidth=2, label=f'Angle θ: {theta_deg:.1f}°')
    
    # 텍스트 위치 선정
    text_pos = np.array([r_arc * 1.3 * np.cos(np.radians(theta_deg/2)), r_arc * 1.3 * np.sin(np.radians(theta_deg/2)), 0.1])
    ax.text(text_pos[0], text_pos[1], text_pos[2], f'θ = {theta_deg:.1f}°', color='#27ae60', fontsize=12, fontweight='bold')

# 그래프 스타일 세팅
max_val = max(5.0, magnitude_A, magnitude_B)
ax.set_xlim([-1, max_val + 1])
ax.set_ylim([-1, max_val + 1])
ax.set_zlim([-1, max_val + 1])

ax.set_xlabel('X (Vector A Axis)')
ax.set_ylabel('Y (Orthogonal Axis)')
ax.set_zlabel('Z')
ax.set_title(f'3D Geometric Vector Representation (Cos θ: {cosine_similarity:.4f})', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')

# 스트림릿에 안전하게 Matplotlib 렌더링 전달
st.pyplot(fig)

# ------------------------------------------------------------------------------
# 6. 이미지 인식(잠금해제) 시나리오 시뮬레이션
# ------------------------------------------------------------------------------
st.divider()
st.subheader("🔐 얼굴 인식 잠금해제(Face ID) 임계값 판정 테스트")

# 임계값 조절 슬라이더
threshold = st.slider("인증 보안 기준치(Threshold) 설정", min_value=0.50, max_value=0.99, value=0.85, step=0.01)

if cosine_similarity >= threshold:
    st.success(f"🔓 **[인증 성공]** 유사도 {cosine_similarity:.4f}가 기준선 {threshold:.2f} 이상이므로 잠금을 해제합니다. (사잇각 θ = {theta_deg:.1f}°)")
else:
    st.error(f"🔒 **[인증 실패]** 유사도 {cosine_similarity:.4f}가 기준선 {threshold:.2f}보다 작으므로 접근을 통제합니다. 다른 사람일 확률이 높습니다.")
