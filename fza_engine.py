import torch
import torch.nn as nn


class FractalZoneNetwork(nn.Module):
    def __init__(self, input_size=784, output_size=10):
        super(FractalZoneNetwork, self).__init__()
        # 1. 계층적 구획
        self.root = nn.Linear(input_size, 256)   # 뿌리: 영구 지식
        self.trunk = nn.Linear(256, 128)         # 줄기: 숙련 지식
        self.leaf = nn.Linear(128, output_size)  # 잎: 일시 정보

        # 2. 중요도 제어 (User-driven)
        self.zone_importance = {'root': 1.0, 'trunk': 0.5, 'leaf': 0.0}

    def forward(self, x):
        x = torch.relu(self.root(x))
        x = torch.relu(self.trunk(x))
        return self.leaf(x)

    def apply_user_shield(self):
        """
        유저가 설정한 등급에 따라 가중치의 '보호막(Shield)'을 물리적으로 적용합니다.
        귤락이 알맹이를 감싸듯, 중요도가 높은 구역의 변화(Gradient)를 차단합니다.
        """
        for name, param in self.named_parameters():
            if 'root' in name:
                param.requires_grad = False if self.zone_importance['root'] == 1.0 else True
            elif 'trunk' in name:
                if param.grad is not None:
                    param.grad *= (1.0 - self.zone_importance['trunk'])

    def lock_knowledge(self, zone='root'):
        """
        유저가 '이건 평생 기억해'라고 하면 귤락처럼 단단하게 굳힙니다.
        """
        if zone == 'root':
            for param in self.root.parameters():
                param.requires_grad = False
            self.zone_importance['root'] = 1.0
            print(f"🔒 [알림] {zone} 구역이 영구 지식으로 잠겼습니다. 망각이 불가능합니다.")

    def flash_delete(self):
        """
        저장 공간이 부족하거나 유저가 원할 때,
        중요도가 낮은 '잎' 구역의 가중치를 번개처럼 초기화(삭제)합니다.
        """
        nn.init.xavier_uniform_(self.leaf.weight)
        nn.init.zeros_(self.leaf.bias)
        self.zone_importance['leaf'] = 0.0
        print("⚡ [알림] 중요도가 낮은 '잎' 구역의 일시 정보가 삭제되었습니다. 용량이 확보되었습니다.")


if __name__ == "__main__":
    engine = FractalZoneNetwork()

    print("--- 기둥 시스템 가동 ---")
    engine.zone_importance['root'] = 1.0
    engine.apply_user_shield()
    print(f"뿌리 구역 보호 상태: {not engine.root.weight.requires_grad}")

    print("\n[시나리오 1: 핵심 지식 습득]")
    engine.lock_knowledge('root')

    print("\n[시나리오 2: 메모리 부족 발생]")
    engine.flash_delete()
