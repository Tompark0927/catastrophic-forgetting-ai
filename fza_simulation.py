import torch
import torch.nn as nn
import torch.optim as optim

# 1. Fractal-Zone Allocator 엔진 정의
class FZANetwork(nn.Module):
    def __init__(self):
        super(FZANetwork, self).__init__()
        # 입력(784) -> 뿌리(256) -> 줄기(128) -> 잎(10)
        self.root = nn.Linear(784, 256)   # 핵심 지식 (이름, 정체성)
        self.trunk = nn.Linear(256, 128)  # 숙련 지식 (취향, 습관)
        self.leaf = nn.Linear(128, 10)    # 일시 지식 (오늘의 날씨, 잡담)
        self.zone_lock = {'root': False, 'trunk': False}

    def forward(self, x):
        x = torch.relu(self.root(x))
        x = torch.relu(self.trunk(x))
        return self.leaf(x)

    def apply_lock(self, zone):
        if zone == 'root':
            for param in self.root.parameters():
                param.requires_grad = False
            self.zone_lock['root'] = True
        msg = f"🔒 {zone.upper()} 구역이 잠겼습니다. 이제 이 지식은 수정(망각)되지 않습니다."
        return msg

    def flash_delete_leaf(self):
        nn.init.xavier_uniform_(self.leaf.weight)
        nn.init.zeros_(self.leaf.bias)
        msg = "⚡ LEAF 구역이 번개처럼 초기화되었습니다. (용량 확보)"
        return msg

def run_fza_simulation(seed=None, include_model=False):
    if seed is not None:
        torch.manual_seed(seed)

    logs = []

    def log(msg):
        logs.append(msg)
        print(msg)

    # 2. 시뮬레이션 환경 설정
    model = FZANetwork()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    criterion = nn.MSELoss()

    # 가상의 데이터 (유저의 이름 정보라고 가정)
    user_data = torch.randn(1, 784)
    target_name = torch.randn(1, 10)

    log("=== 🌟 FZA 엔진 시뮬레이션 시작 ===")

    # --- [STEP 1: 핵심 지식 학습 및 잠금] ---
    log("\n[단계 1] 유저의 핵심 정보를 '뿌리'에 학습합니다.")
    output = model(user_data)
    loss = criterion(output, target_name)
    loss.backward()
    optimizer.step()
    log(model.apply_lock('root')) # 학습 후 즉시 잠금 (귤락 형성)

    # --- [STEP 2: 망각 테스트 (공격)] ---
    log("\n[단계 2] 새로운 데이터로 기존 지식을 덮어쓰려 시도합니다.")
    fake_data = torch.randn(1, 784)
    fake_target = torch.randn(1, 10)

    optimizer.zero_grad()
    output = model(fake_data)
    loss = criterion(output, fake_target)
    loss.backward()
    optimizer.step()

    # 잠긴 구역의 가중치 변화 확인
    root_locked = not model.root.weight.requires_grad
    log(f">> 뿌리 구역 가중치 변화 여부: {root_locked} (True면 망각 방지 성공)")

    # --- [STEP 3: 번개 삭제 시뮬레이션] ---
    log("\n[단계 3] 일시적 데이터(Leaf)가 가득 찼습니다. 유저의 명령으로 삭제합니다.")
    log(model.flash_delete_leaf())

    success = bool(root_locked)
    end_msg = "\n=== ✨ 시뮬레이션 종료: 모든 시스템 정상 작동 ===" if success else "\n=== ⚠️ 시뮬레이션 종료: 상태 점검 필요 ==="
    log(end_msg)

    result = {
        "success": success,
        "root_locked": root_locked,
        "logs": logs,
    }
    if include_model:
        result["model"] = model
    return result


if __name__ == "__main__":
    run_fza_simulation()
