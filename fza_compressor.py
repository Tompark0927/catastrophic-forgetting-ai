import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class FZACompressor:
    @staticmethod
    def compress_block(model, amount=0.90, zone_patterns=None):
        """
        지정 구역의 Linear 레이어를 L1 프루닝 + FP16 양자화로 압축합니다.

        Args:
            model: 어떤 nn.Module 이든 가능 (FZANetwork, LLaMA 등)
            amount: 프루닝 비율 (0.90 = 90% 가중치를 0으로)
            zone_patterns: 압축 대상 레이어 이름 패턴 리스트
                           예) ['layers.0', 'layers.1', 'embed']
                           None 이면 model.root 사용 (구 방식 호환)

        저장 경로: vault/compressed_seed.fza
        """
        if zone_patterns:
            pruned_count = 0
            for name, module in model.named_modules():
                if any(p in name for p in zone_patterns) and isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=amount)
                    prune.remove(module, 'weight')
                    pruned_count += 1
            print(f"✂️ [{pruned_count}개 Linear 레이어] {amount * 100:.0f}% 프루닝 완료.")

            compressed = {
                k: v.detach().to(torch.float16)
                for k, v in model.named_parameters()
                if any(p in k for p in zone_patterns)
            }

        elif hasattr(model, 'root'):
            prune.l1_unstructured(model.root, name='weight', amount=amount)
            prune.remove(model.root, 'weight')
            print(f"✂️ [root 레이어] {amount * 100:.0f}% 프루닝 완료.")

            compressed = {
                k: v.to(torch.float16)
                for k, v in model.root.state_dict().items()
            }

        else:
            raise ValueError(
                "zone_patterns 을 지정하거나 model.root 속성이 있어야 합니다.\n"
                "예) zone_patterns=['layers.0', 'embed_tokens']"
            )

        path = "vault/compressed_seed.fza"
        torch.save({'weights': compressed}, path)

        size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        print(f"📉 [양자화 완료] FP32 → FP16. 저장: {path} ({size_kb:.1f} KB)")
