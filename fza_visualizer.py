import os
import torch


class FZAVisualizer:
    @staticmethod
    def print_status_report(model, math_engine=None):
        print("\n" + "=" * 20 + " [FZA 지식 지형도] " + "=" * 20)

        # 1. 뿌리(Root) 상태 - 화석화 정도
        root_weight_sum = torch.sum(torch.abs(model.root.weight)).item()
        lock_status = "🔒 고정됨" if model.is_locked else "🔓 유연함"
        print(f"🌲 뿌리(Root) 구역: [{lock_status}] | 지식 밀도: {root_weight_sum:.4f}")

        # 2. 잎(Leaf) 상태 - 현재 부하량
        leaf_weight_sum = torch.sum(torch.abs(model.leaf.weight)).item()
        print(f"🍃 잎(Leaf) 구역: [활성화] | 현재 데이터 부하: {leaf_weight_sum:.4f}")

        # 3. 효율성 진단
        if leaf_weight_sum > 50.0:
            print("⚠️ 경고: 잎 구역에 노이즈가 많습니다. '다 지워' 명령을 추천합니다.")
        elif model.is_locked:
            print("✨ 최적 상태: 핵심 지식이 안전하게 보호되고 있습니다.")
        else:
            print("ℹ️ 안내: 핵심 지식을 보존하려면 '평생 기억해'를 실행하세요.")

        # 3. 수학 엔진 상태
        if math_engine is not None:
            count = len(math_engine.math_vault)
            seed_exists = os.path.exists("vault/math_seed_v1.fza")
            seed_label = "✅ 씨앗 파일 있음" if seed_exists else "❌ 씨앗 파일 없음"
            print(f"📐 수학(Math) 구역: 등록된 수식 {count}개 | {seed_label}")

        print("=" * 55)
