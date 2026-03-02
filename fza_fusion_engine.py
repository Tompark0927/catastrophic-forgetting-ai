import torch


class FZAFusion:
    @staticmethod
    def fuse_blocks(block_a_path, block_b_path, alpha=0.5):
        """
        두 지식 블록을 가중 평균으로 융합하여 새 블록을 생성합니다.

        Args:
            block_a_path: 첫 번째 .fza 파일 경로
            block_b_path: 두 번째 .fza 파일 경로
            alpha: 블록 A 의 반영 비중 (0.5 = 반반, 1.0 = A 전체)

        Returns:
            fused_path: 생성된 융합 블록 경로 (실패 시 None)
        """
        data_a = torch.load(block_a_path, weights_only=False)
        data_b = torch.load(block_b_path, weights_only=False)

        # payload 에서 weights 딕셔너리만 추출
        # (구 형식: state_dict 그대로 / 신 형식: {'weights': ..., 'hash': ..., ...})
        weights_a = data_a.get('weights', data_a)
        weights_b = data_b.get('weights', data_b)

        # 구조 호환성 검사
        keys_a, keys_b = set(weights_a.keys()), set(weights_b.keys())
        if keys_a != keys_b:
            print("❌ [실패] 두 블록의 레이어 구조가 달라 융합할 수 없습니다.")
            print(f"  블록 A 전용: {list(keys_a - keys_b)[:3]}")
            print(f"  블록 B 전용: {list(keys_b - keys_a)[:3]}")
            return None

        # 가중 평균 융합 (FP32 로 업캐스트 후 연산)
        fused_weights = {
            key: alpha * weights_a[key].float() + (1.0 - alpha) * weights_b[key].float()
            for key in weights_a
        }

        fused_path = "vault/fused_knowledge.fza"
        torch.save(
            {'weights': fused_weights, 'metadata': f'Fused alpha={alpha}'},
            fused_path
        )

        print(f"🧬 [융합 성공] 두 블록이 결합되었습니다. (A 비중: {alpha:.0%})")
        print(f"✨ 새로운 블록 '{fused_path}' 가 탄생했습니다.")
        return fused_path
