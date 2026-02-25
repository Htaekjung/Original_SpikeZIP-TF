import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.linalg import norm
import re

# ==========================================
# 1. 경로 설정
# ==========================================
DIR_QANN = "/home/hyuntaek/Original_SpikeZIP-TF/log/T-SNN_vit_small_patch16_cifar100_gelu_QANN_QAT_act16_weightbit32/numpy_qann"
DIR_SNN = "/home/hyuntaek/Original_SpikeZIP-TF/log/T-SNN_vit_small_patch16_cifar100_gelu_SNN_act16_weightbit32/numpy_snn"
OUTPUT_IMAGE = "analysis_result.png"

# [핵심 수정] 실제 데이터 흐름(Norm -> Attn -> Norm -> MLP) 순서로 정렬하는 함수
def total_sort_key(s):
    # 1. 큰 카테고리 우선순위 (Patch Embed -> Blocks -> Head)
    if 'patch_embed' in s:
        main_priority = 0
    elif 'head' in s:
        main_priority = 2
    else:
        main_priority = 1

    # 2. 블록 번호 추출 (blocks.3.xxx 에서 3 추출)
    block_idx = 0
    block_match = re.search(r'blocks\.(\d+)', s)
    if block_match:
        block_idx = int(block_match.group(1))

    # 3. 블록 내 세부 레이어 우선순위 (물리적 데이터 흐름 순서)
    sub_priority = 99
    if 'norm1' in s: sub_priority = 10
    elif 'attn' in s: sub_priority = 20
    elif 'norm2' in s: sub_priority = 30
    elif 'mlp.fc1' in s: sub_priority = 40
    elif 'mlp.act' in s: 
        sub_priority = 50
        if '_post' in s: sub_priority += 2  # 양자화가 먼저 (52)
        if '_act' in s: sub_priority += 4   # 활성화 함수가 나중 (54)
    elif 'mlp.fc2.0' in s: sub_priority = 60
    elif 'mlp.fc2.1' in s: sub_priority = 70
    
    # _post 파일은 해당 원본 레이어 바로 뒤에 오도록 소수점 부여
    if '_post' in s:
        sub_priority += 5

    # (대분류, 블록번호, 소분류, 전체이름) 순으로 튜플을 만들어 정렬 기준 설정
    return (main_priority, block_idx, sub_priority, s)

def analyze_difference(a, b):
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    if np.isnan(a_flat).any() or np.isnan(b_flat).any():
        return None, None, "NaN detected"
    mse = np.mean((a_flat - b_flat) ** 2)
    norm_a, norm_b = norm(a_flat), norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0, mse, "Norm is 0"
    sim = np.dot(a_flat, b_flat) / (norm_a * norm_b)
    return sim, mse, "OK"

# ==========================================
# 메인 실행 로직
# ==========================================
if not os.path.exists(DIR_QANN):
    print("[Error] 경로를 찾을 수 없습니다.")
    exit()

files = os.listdir(DIR_QANN)
# 강화된 정렬 키 적용
layers = sorted([f for f in files if f.endswith('.npy')], key=total_sort_key)

print(f"\n{'Layer Name':<35} | {'Cos Sim':<10} | {'MSE':<12} | {'Status'}")
print("-" * 105)

layer_names, similarities, mses, indices = [], [], [], []

for i, layer_file in enumerate(layers):
    path_q = os.path.join(DIR_QANN, layer_file)
    path_s = os.path.join(DIR_SNN, layer_file)
    if not os.path.exists(path_s): continue
        
    val_q, val_s = np.load(path_q), np.load(path_s)
    sim, mse, msg = analyze_difference(val_q, val_s)
    clean_name = layer_file.replace('.npy', '')

    if msg != "OK":
        status = f"🚨 ERROR ({msg})"
    elif sim >= 0.9:
        status = "✅ 정상"
    elif 0.1 <= sim < 0.5:
        status = "⚠️ 위험"
    elif sim < 0.1:
        status = "❌ ERROR"
    else:
        status = "중간" 

    print(f"{clean_name:<35} | {sim:<10.4f} | {mse:<12.4e} | {status}")

    layer_names.append(clean_name)
    similarities.append(sim if sim is not None else 0.0)
    mses.append(mse if mse is not None else 0.0)
    indices.append(i)

# ==========================================
# 그래프 시각화 (가독성 최적화)
# ==========================================

# if "patch_embed.proj" in layer_file:
    
#     # 두 데이터의 1번 채널 이미지만 시각적으로 비교
#     plt.subplot(1, 2, 1); plt.imshow(val_q[0, 0]); plt.title("QANN")
#     plt.subplot(1, 2, 2); plt.imshow(val_s[0, 0]); plt.title("SNN")
#     plt.show()



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

# Cosine Similarity
ax1.plot(indices, similarities, marker='o', linestyle='-', color='blue', markersize=4)
ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
ax1.set_title("Cosine Similarity (Flow: Norm -> Attn -> Norm -> MLP)", fontsize=14)
ax1.set_xticks(indices)
ax1.set_xticklabels(layer_names, rotation=90, fontsize=6)
ax1.grid(True, alpha=0.2)

# MSE
ax2.plot(indices, mses, marker='s', linestyle='-', color='red', markersize=4)
ax2.set_yscale('log')
ax2.set_title("Mean Squared Error (Flow: Norm -> Attn -> Norm -> MLP)", fontsize=14)
ax2.set_xticks(indices)
ax2.set_xticklabels(layer_names, rotation=90, fontsize=6)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE)
print(f"\n[Done] 실제 모델 구조 순서대로 정렬 및 분석 완료.")
print("QANN Top-5:", val_q.flatten().argsort()[-5:])
print("SNN  Top-5:", val_s.flatten().argsort()[-5:])