import torch
import hashlib
import os


class FZAStorage:
    # ── 내부 유틸 ────────────────────────────────────────────────
    @staticmethod
    def _get_zone_weights(model, zone_patterns=None):
        """
        zone_patterns 리스트가 있으면 패턴 매칭으로,
        없으면 model.root 속성으로 가중치를 추출.
        → 어떤 PyTorch 모델(HuggingFace 포함)에도 동작.
        """
        if zone_patterns:
            return {
                k: v.detach().cpu().clone()
                for k, v in model.named_parameters()
                if any(p in k for p in zone_patterns)
            }
        if hasattr(model, 'root'):
            return model.root.state_dict()
        raise ValueError(
            "zone_patterns 을 지정하거나 model.root 속성이 있어야 합니다.\n"
            "예) zone_patterns=['embed_tokens', 'layers.0', 'layers.1']"
        )

    @staticmethod
    def _compute_hash(weights_dict):
        """가중치 딕셔너리의 SHA-256 해시를 계산합니다."""
        weights_tensor = torch.cat([v.flatten().float() for v in weights_dict.values()])
        return hashlib.sha256(weights_tensor.cpu().numpy().tobytes()).hexdigest()

    # ── 블록 추출 ─────────────────────────────────────────────────
    @staticmethod
    def export_root_block(model, block_name="knowledge_seed_v1", zone_patterns=None):
        """
        지정 구역(zone)의 가중치를 초경량 블록(.fza)으로 추출합니다.

        Args:
            model: 어떤 nn.Module 이든 가능 (FZANetwork, LLaMA, Mistral 등)
            block_name: 저장 파일명 (vault/{block_name}.fza)
            zone_patterns: 루트 구역 레이어 이름 패턴 리스트
                           예) ['embed_tokens', 'layers.0.', 'layers.1.']
                           None 이면 model.root 속성을 사용 (구 방식 호환)
        """
        os.makedirs("vault", exist_ok=True)

        root_weights = FZAStorage._get_zone_weights(model, zone_patterns)
        block_hash = FZAStorage._compute_hash(root_weights)

        payload = {
            'weights': root_weights,
            'hash': block_hash,
            'zone_patterns': zone_patterns or [],
            'metadata': 'FZA Permanent Knowledge Block',
        }

        path = f"vault/{block_name}.fza"
        torch.save(payload, path)

        print(f"💎 [블록화 완료] '{path}' 생성됨.")
        print(f"🔑 지식 지문(Hash): {block_hash[:10]}...")
        return path, block_hash

    # ── 블록 검증 + 이식 ──────────────────────────────────────────
    @staticmethod
    def verify_and_load(model, block_name, zone_patterns=None):
        """
        저장된 블록의 무결성을 검사하고 모델에 이식합니다.
        실제 LLM 에도 strict=False 로 안전하게 적재합니다.
        """
        path = f"vault/{block_name}.fza"
        if not os.path.exists(path):
            print(f"❌ [실패] '{path}' 파일이 없습니다.")
            return False

        # weights_only=False: metadata/hash(string) 포함 payload 역직렬화
        payload = torch.load(path, weights_only=False)
        saved_weights = payload['weights']
        saved_hash = payload.get('hash')

        # 무결성 검사
        current_hash = FZAStorage._compute_hash(saved_weights)
        if saved_hash and saved_hash != current_hash:
            print("❌ [실패] 지식 블록 해시 검증에 실패했습니다. 로드를 중단합니다.")
            return False

        # 이식: 패턴 기반(실제 LLM) 또는 model.root 기반(장난감 모델)
        if zone_patterns or not hasattr(model, 'root'):
            current_state = model.state_dict()
            matched = 0
            for k, v in saved_weights.items():
                if k in current_state:
                    # dtype을 현재 모델에 맞게 변환 (BF16, FP16, FP32 모두 대응)
                    current_state[k] = v.to(current_state[k].dtype)
                    matched += 1
            model.load_state_dict(current_state, strict=False)
            print(f"🧬 [이식 성공] '{block_name}' — {matched}개 파라미터 복원.")
        else:
            model.root.load_state_dict(saved_weights)
            print(f"🧬 [이식 성공] '{block_name}' 지식이 모델의 뿌리에 결합되었습니다.")

        return True

    # ── 구역 잠금 ─────────────────────────────────────────────────
    @staticmethod
    def lock_zone(model, zone_patterns=None):
        """
        지정 구역의 파라미터를 동결(requires_grad=False)합니다.
        이후 optimizer.step() 이 호출돼도 해당 구역은 변하지 않습니다.
        """
        if zone_patterns:
            count = 0
            for name, param in model.named_parameters():
                if any(p in name for p in zone_patterns):
                    param.requires_grad = False
                    count += 1
            print(f"🔒 [{count}개 파라미터] 루트 구역 동결 완료.")
        elif hasattr(model, 'root'):
            for param in model.root.parameters():
                param.requires_grad = False
            print("🔒 [root 구역] 동결 완료.")
        else:
            raise ValueError(
                "zone_patterns 을 지정하거나 model.root 속성이 있어야 합니다."
            )
