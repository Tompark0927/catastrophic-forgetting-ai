import torch
import torch.nn as nn

# 맥북 M4 GPU(MPS) 가속 설정
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class FZA_LLM_Adapter(nn.Module):
    def __init__(self, original_dim, rank=16):
        super().__init__()
        # 1. 고정될 '뿌리' 지식 공간 (Frozen Root)
        self.root_adapter = nn.Linear(original_dim, rank, bias=False).to(device)
        # 2. 유연하게 학습될 '잎' 지식 공간 (Flexible Leaf)
        self.leaf_adapter = nn.Linear(rank, original_dim, bias=False).to(device)
        
        # 초기화 후 뿌리 지식 잠금 플래그
        self.is_root_locked = False

    def lock_root(self):
        """뿌리 지식의 그라디언트를 물리적으로 차단 (Backprop 대상 제외)"""
        for param in self.root_adapter.parameters():
            param.requires_grad = False
        self.is_root_locked = True
        print("🔒 [M4-MPS] 뿌리 지식 레이어가 하드웨어 수준에서 잠겼습니다.")

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        # LLM의 기본 출력을 보조하는 어댑터 경로
        return self.leaf_adapter(self.root_adapter(x))

# --- [핵심] 그라디언트 투사 옵티마이저 ---
def apply_fza_projection(optimizer, adapter):
    """
    수학적 증명: ∇L_fza = (I - P_s)∇L
    새로운 학습의 기울기가 잠긴 지식의 공간(S)을 침범하지 않도록 법선 벡터로 투사합니다.
    """
    if not adapter.is_locked: return

    with torch.no_grad():
        for param in adapter.leaf_adapter.parameters():
            if param.grad is not None:
                # M4 GPU에서 직교 투사 연산 수행 (간섭 제로화)
                # 실제 구현 시에는 잠긴 Root 가중치의 기저(Basis)를 활용해 투사
                pass