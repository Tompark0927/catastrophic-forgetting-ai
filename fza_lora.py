"""
FZA LoRA — PEFT LoRA 를 이용한 실제 가중치 파인튜닝
────────────────────────────────────────────────────
"내 이름은 홍길동이야" 같은 사실을 LLM 가중치에 직접 새깁니다.
전체 파라미터의 ~1% 만 학습 (LoRA rank=16) → 빠르고 가볍습니다.

[현재 시스템과의 차이]
  fza_llm_bridge  →  프롬프트로 기억  (재시작해도 유지되나 가중치에는 없음)
  FZALoRA         →  가중치로 기억    (모델 자체가 알게 됨 — 진짜 망각 없음)

의존성: pip install peft transformers
GPU 권장 (Apple MPS / CUDA). CPU 도 동작하나 느립니다.

사용 예시:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model     = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    manager   = FZAManager(model=model, tokenizer=tokenizer)
    # 이후 '이름은...' 명령 시 LoRA 파인튜닝 자동 실행
"""
import os
import torch


class FZALoRA:
    def __init__(self, model, tokenizer, target_modules=None, r=16, lora_alpha=32, ewc=None):
        """
        Args:
            model:          HuggingFace CausalLM 모델
            tokenizer:      대응 토크나이저
            target_modules: LoRA 를 적용할 레이어 이름 리스트
                            None 이면 ["q_proj", "v_proj"] 사용 (LLaMA/Mistral 기본값)
            r:              LoRA rank (높을수록 표현력↑, 메모리↑)
            lora_alpha:     LoRA 스케일링 계수 (보통 r*2)
            ewc:            Optional FZAEwc instance. When supplied, the EWC
                            penalty is added to the training loss to protect
                            Root-zone parameters from catastrophic forgetting.
        """
        from peft import LoraConfig, get_peft_model, TaskType

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules or ["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.model     = get_peft_model(model, config)
        self.tokenizer = tokenizer
        self.device    = next(model.parameters()).device
        self.ewc       = ewc  # FZAEwc instance or None
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=2e-4,
        )

        trainable, total = self.model.get_nb_trainable_parameters()
        ewc_status = f" | EWC: {'✅ 활성화' if ewc and ewc.is_active else '⬜ 비활성화'}" 
        print(f"🔧 [LoRA] 준비 완료. 학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}%){ewc_status}")

    # ── 사실 파인튜닝 ─────────────────────────────────────────
    def train_on_fact(self, fact_text: str, steps: int = 20):
        """
        단일 사실 텍스트를 LLM 가중치에 직접 새깁니다.
        EWC 인스턴스가 있으면 Root 구역 보호 패널티를 손실에 추가합니다.

        Args:
            fact_text: 학습시킬 문장 ("홍길동은 뛰어난 개발자입니다.")
            steps:     반복 학습 횟수. 높을수록 강하게 새겨지나 과적합 주의.
        """
        self.model.train()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            fact_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        last_loss      = 0.0
        last_task_loss = 0.0
        last_ewc_loss  = 0.0
        for _ in range(steps):
            self.optimizer.zero_grad()
            outputs   = self.model(**inputs, labels=inputs["input_ids"])
            task_loss = outputs.loss

            # ── EWC Penalty (Root 보호) ───────────────────────
            ewc_penalty = torch.tensor(0.0, device=self.device)
            if self.ewc and self.ewc.is_active:
                ewc_penalty = self.ewc.ewc_loss().to(self.device)

            total_loss = task_loss + ewc_penalty
            total_loss.backward()
            self.optimizer.step()

            last_loss      = total_loss.item()
            last_task_loss = task_loss.item()
            last_ewc_loss  = ewc_penalty.item()

        print(f"🔥 [LoRA 학습] '{fact_text[:40]}' — {steps}스텝 완료.")
        if self.ewc and self.ewc.is_active:
            print(f"   task_loss: {last_task_loss:.4f}  |  EWC 패널티: {last_ewc_loss:.4f}  |  total: {last_loss:.4f}")
        else:
            print(f"   최종 loss: {last_loss:.4f}  ✅ 가중치에 실제로 새겨졌습니다.")

    # ── 어댑터 저장 / 불러오기 ────────────────────────────────
    def save_adapter(self, path="vault/lora_adapter"):
        """LoRA 어댑터 가중치를 저장합니다. (전체 모델 대비 ~1% 크기)"""
        self.model.save_pretrained(path)
        print(f"💎 [LoRA] 어댑터 저장 완료: {path}")

    @staticmethod
    def load_adapter(base_model, path="vault/lora_adapter"):
        """저장된 LoRA 어댑터를 기반 모델에 이식합니다."""
        from peft import PeftModel
        if not os.path.exists(path):
            print("❌ [LoRA] 저장된 어댑터가 없습니다.")
            return None
        model = PeftModel.from_pretrained(base_model, path)
        print(f"📂 [LoRA] 어댑터 복구 완료: {path}")
        return model
