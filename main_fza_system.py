import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
import json
import uuid
from datetime import datetime

from fza_storage_v2 import FZAStorage
from fza_visualizer import FZAVisualizer
from fza_math_compressor import FZAMathEngine
from fza_compressor import FZACompressor
from fza_fusion_engine import FZAFusion
from fza_llm_bridge import FZALLMBridge


# ── 구역(Zone) 설정 ─────────────────────────────────────────────
class FZAConfig:
    """
    실제 LLM 의 구역(Zone) 정의.
    레이어 이름 패턴으로 root / trunk / leaf 구역을 지정합니다.

    None 이면 구 방식(model.root / model.leaf 속성) 을 사용합니다.

    사용 예시:
        # LLaMA / Mistral 계열
        config = FZAConfig(
            root_patterns=['embed_tokens', 'layers.0.', 'layers.1.', 'layers.2.'],
            trunk_patterns=['layers.3.',  ..., 'layers.28.'],
            leaf_patterns=['layers.29.', 'layers.30.', 'layers.31.', 'lm_head', 'norm'],
        )

        # GPT-2 계열
        config = FZAConfig(
            root_patterns=['wte', 'wpe', 'h.0.', 'h.1.'],
            leaf_patterns=['h.10.', 'h.11.', 'ln_f'],
        )

        # 장난감 FZANetwork (기본값 — 패턴 없이 model.root 속성 사용)
        config = FZAConfig()
    """
    def __init__(self, root_patterns=None, trunk_patterns=None, leaf_patterns=None):
        self.root_patterns  = root_patterns   # 영구 지식 구역
        self.trunk_patterns = trunk_patterns  # 숙련 지식 구역 (선택)
        self.leaf_patterns  = leaf_patterns   # 일시 정보 구역


# ── 장난감 FZANetwork (데모 / 기본값) ──────────────────────────
class FZANetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.root = nn.Linear(10, 10)  # 뿌리: 영구 지식
        self.leaf = nn.Linear(10, 2)   # 잎: 일시 정보
        self.is_locked = False

    def forward(self, x):
        x = torch.relu(self.root(x))
        return self.leaf(x)


# ── 통합 관리 시스템 ──────────────────────────────────────────
class FZAManager:
    def __init__(
        self,
        user_id="default",
        model=None,
        tokenizer=None,
        config=None,
        use_local: bool = False,
        local_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    ):
        """
        Args:
            user_id:          사용자 식별자. 멀티유저 지원 (vault/users/{user_id}/).
            model:            실제 LLM(HuggingFace nn.Module) 또는 None (→ 데모 FZANetwork).
            tokenizer:        HuggingFace 토크나이저.
            config:           FZAConfig 인스턴스.
            use_local:        True 이면 FZALocalEngine(로컬 Mistral/Llama).
                              False 이면 FZALLMBridge(Anthropic API) 사용.
            local_model_name: use_local=True 일 때 로드할 HuggingFace 모델 ID.
        """
        self.user_id    = user_id
        self.vault_path = os.path.join("vault", "users", user_id)
        os.makedirs(self.vault_path, exist_ok=True)

        self.config = config or FZAConfig()
        self.use_local = use_local

        if model is not None:
            self.model = model
            trainable = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable, lr=1e-5) if trainable else None
        else:
            self.model = FZANetwork()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        self.criterion   = nn.MSELoss()
        self.math_engine = FZAMathEngine()

        # ── RAG (벡터 기억) 초기화 ──────────────────────────
        try:
            from fza_rag import FZAMemory
            self.rag = FZAMemory()
            self.rag.load(path=os.path.join(self.vault_path, "rag_memory"), silent=True)
            print("✅ [RAG] 벡터 기억 시스템 활성화.")
        except Exception:
            print("⚠️ [RAG] 비활성화 (pip install sentence-transformers faiss-cpu)")
            self.rag = None

        # ── EWC placeholder ─────────────────────────────────
        self.ewc = None

        # ── Local Engine OR Anthropic Bridge ─────────────────
        if use_local:
            from fza_local_engine import FZALocalEngine
            self.bridge = FZALocalEngine(
                model_name=local_model_name,
                math_engine=self.math_engine,
                memory=self.rag,
                vault_path=self.vault_path,
            )
            print(f"✅ [LocalEngine] '{local_model_name}' 연결.")
        else:
            self.bridge = FZALLMBridge(
                self.math_engine, memory=self.rag, vault_path=self.vault_path
            )

        # ── LoRA 초기화 (실제 모델 + 토크나이저가 있을 때만) ──
        self.lora = None
        if model is not None and tokenizer is not None:
            try:
                from fza_lora import FZALoRA
                self.lora = FZALoRA(self.model, tokenizer, ewc=self.ewc)
                print("✅ [LoRA] 가중치 파인튜닝 엔진 활성화.")
            except Exception as e:
                print(f"⚠️ [LoRA] 비활성화 ({e})")

        # 대화 저장 경로
        self.conversations_path = os.path.join(self.vault_path, "conversations")
        os.makedirs(self.conversations_path, exist_ok=True)
        self.current_conv_id = None

        # 시작 시 자동 복구 (silent — 파일 없어도 에러 없음)
        self.bridge.load_profile(silent=True)
        self.math_engine.load_from_seed(silent=True)

        # EWC 체크포인트 자동 복구
        ewc_path = os.path.join(self.vault_path, "ewc_checkpoint.pt")
        if use_local and os.path.exists(ewc_path):
            try:
                from fza_ewc import FZAEwc
                self.ewc = FZAEwc(
                    self.bridge.raw_model,
                    zone_patterns=self.config.root_patterns,
                )
                self.ewc.load(ewc_path)
                if self.lora:
                    self.lora.ewc = self.ewc
            except Exception as e:
                print(f"⚠️ [EWC] 복구 실패: {e}")

    # ── 학습 / 사실 등록 ────────────────────────────────────────
    def learn_info(self, text_data):
        """유저의 정보를 루트 프로필에 저장합니다.
        장난감 모델이면 신경망 학습 시뮬레이션도 추가로 수행합니다."""
        print(f"⚡ [학습] '{text_data}' 정보를 기록합니다...")

        # 장난감 모델 전용: 신경망 학습 시뮬레이션
        if isinstance(self.model, FZANetwork) and self.optimizer:
            input_data = torch.randn(1, 10)
            target     = torch.randn(1, 2)
            self.model.train()
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(input_data), target)
            loss.backward()
            self.optimizer.step()

        # 1. 구조화 파싱 시도 ("X는 Y야" → key-value 사실)
        m = re.search(
            r'(?:내\s*)?(.+?)(?:은|는)\s*(.+?)(?:이야|야|이다|다|입니다)?$',
            text_data.strip()
        )
        if m:
            key   = m.group(1).strip().lstrip('내').strip()
            value = m.group(2).strip()
            self.bridge.set_user_fact(key, value)

        # 2. 원본 텍스트 전체를 일반 기억에도 항상 저장 (구조화 여부와 무관)
        self.bridge.add_memory(text_data)
        self.bridge.save_profile()  # 즉시 디스크 저장

        # 3. RAG: 벡터 기억에도 추가
        if self.rag:
            self.rag.add(text_data)
            self.rag.save(path=os.path.join(self.vault_path, "rag_memory"))

        # LoRA: 실제 LLM 가중치에 파인튜닝 (실제 모델 연결 시)
        if self.lora:
            self.lora.train_on_fact(text_data)
            self.lora.save_adapter()

        print("✅ 학습 완료: 임시 지식으로 저장되었습니다.")

    # ── 지식 잠금 & 블록 추출 ────────────────────────────────────
    def lock_and_export(self, block_name="permanent_knowledge"):
        """지정 구역을 동결하고 물리적 파일(Block)로 추출합니다.
        use_local=True 시 EWC Fisher Information Matrix도 자동 캡쳐."""

        # Which model to lock: local engine raw model or toy FZANetwork
        target_model = (
            self.bridge.raw_model
            if (self.use_local and hasattr(self.bridge, 'raw_model'))
            else self.model
        )

        FZAStorage.lock_zone(target_model, self.config.root_patterns)
        if hasattr(target_model, 'is_locked'):
            target_model.is_locked = True

        block_path = os.path.join(self.vault_path, block_name)
        path, block_hash = FZAStorage.export_root_block(
            target_model,
            block_name=block_path,
            zone_patterns=self.config.root_patterns,
        )

        self.bridge.save_profile()
        if self.math_engine.math_vault:
            self.math_engine.compress_to_seed()
        if self.rag:
            self.rag.save(path=os.path.join(self.vault_path, "rag_memory"))

        # ── EWC: Compute Fisher Information Matrix over permanent memories ──
        if self.use_local and hasattr(self.bridge, 'raw_model'):
            try:
                from fza_ewc import FZAEwc
                memories = self.bridge.user_profile.get('_memories', [])
                device   = str(next(target_model.parameters()).device)
                self.ewc = FZAEwc(
                    target_model,
                    zone_patterns=self.config.root_patterns,
                    ewc_lambda=1000.0,
                )
                self.ewc.compute_fisher(
                    tokenizer=self.bridge.tokenizer,
                    memory_texts=memories,
                    device=device,
                )
                ewc_ckpt = os.path.join(self.vault_path, "ewc_checkpoint.pt")
                self.ewc.save(ewc_ckpt)
                if self.lora:
                    self.lora.ewc = self.ewc
                print("🔒 [EWC] Fisher 보호 활성화 완료 — Root 구역이 수학적으로 보호됩니다.")
            except Exception as e:
                print(f"⚠️ [EWC] Fisher 캡쳐 실패 — {e}")

        print(f"🔒 [잠금] 뿌리 구역이 고정되었습니다. 망각이 차단됩니다.")
        print(f"🔑 지식 지문(Hash): {block_hash[:10]}...")

    # ── 블록 복구 ────────────────────────────────────────────────
    def load_block(self, block_name="permanent_knowledge"):
        """vault 블록에서 구역 지식을 복구합니다."""
        block_path = os.path.join(self.vault_path, block_name)
        ok = FZAStorage.verify_and_load(
            self.model,
            block_name=block_path,
            zone_patterns=self.config.root_patterns,
        )
        if not ok:
            print(f"❌ [복구 실패] '{block_name}' 블록을 찾지 못했거나 검증에 실패했습니다.")

    # ── 수식 관리 ────────────────────────────────────────────────
    def add_formula(self, name, formula):
        self.math_engine.add_formula(name, formula)

    def compress_math(self):
        if not self.math_engine.math_vault:
            print("⚠️ [알림] 등록된 수식이 없습니다. 먼저 '수식 추가' 명령을 사용하세요.")
            return
        self.math_engine.compress_to_seed()

    def list_formulas(self):
        vault = self.math_engine.math_vault
        if not vault:
            print("📭 [알림] 등록된 수식이 없습니다.")
            return
        print(f"📐 [수식 목록] 총 {len(vault)}개:")
        for name, formula in vault.items():
            print(f"  · {name}: {formula}")

    def load_math(self):
        self.math_engine.load_from_seed()

    # ── 대화 세션 관리 ───────────────────────────────────────────
    def new_conversation(self) -> str:
        """새 대화를 시작합니다. conv_id 반환."""
        conv_id = uuid.uuid4().hex[:10]
        self.current_conv_id = conv_id
        self.bridge.conversation_history = []
        return conv_id

    def load_conversation(self, conv_id: str) -> dict:
        """저장된 대화를 불러옵니다."""
        path = os.path.join(self.conversations_path, f"{conv_id}.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.bridge.conversation_history = data.get("history", [])
        self.current_conv_id = conv_id
        return data

    def save_conversation(self, conv_id: str):
        """현재 대화를 저장합니다. 제목은 첫 메시지에서 자동 생성."""
        path = os.path.join(self.conversations_path, f"{conv_id}.json")
        existing = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        # 제목 자동 생성: 첫 user 메시지에서
        if "title" not in existing:
            first = next(
                (m["content"] for m in self.bridge.conversation_history if m["role"] == "user"),
                "새 대화"
            )
            existing["title"] = first[:28] + ("…" if len(first) > 28 else "")

        existing.update({
            "id":         conv_id,
            "history":    self.bridge.conversation_history,
            "updated_at": datetime.now().isoformat(),
        })
        if "created_at" not in existing:
            existing["created_at"] = existing["updated_at"]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def list_conversations(self) -> list:
        """대화 목록을 최신순으로 반환합니다."""
        convs = []
        for fname in os.listdir(self.conversations_path):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(self.conversations_path, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            convs.append({
                "id":         data.get("id", fname[:-5]),
                "title":      data.get("title", "대화"),
                "updated_at": data.get("updated_at", ""),
                "turn_count": len(data.get("history", [])) // 2,
            })
        convs.sort(key=lambda x: x["updated_at"], reverse=True)
        return convs

    def delete_conversation(self, conv_id: str) -> bool:
        """대화를 영구 삭제합니다."""
        path = os.path.join(self.conversations_path, f"{conv_id}.json")
        if os.path.exists(path):
            os.remove(path)
            if self.current_conv_id == conv_id:
                self.current_conv_id = None
                self.bridge.conversation_history = []
            return True
        return False

    # ── AI 대화 ─────────────────────────────────────────────────
    def chat(self, message: str) -> str:
        return self.bridge.chat(message)

    def chat_and_remember(self, message: str, conv_id: str = None) -> dict:
        """대화 후 자동으로 기억할 사실을 추출하여 저장합니다.
        Returns: {"reply": str, "new_memories": list[str], "conv_id": str}
        """
        # 대화 세션 전환
        if conv_id and conv_id != self.current_conv_id:
            self.load_conversation(conv_id)
        if not self.current_conv_id:
            conv_id = self.new_conversation()

        reply = self.bridge.chat(message)

        # Stage 4: 자동 기억 추출 → 스마트 병합
        new_facts = self.bridge.auto_extract_memory(message, reply)
        added, replaced = [], []
        if new_facts:
            added, replaced = self.bridge.smart_merge_memories(new_facts)
            # RAG: 추가된 것은 그대로, 교체된 것은 new 버전만 추가
            for fact in added:
                if self.rag:
                    self.rag.add(fact)
            for pair in replaced:
                if self.rag:
                    self.rag.add(pair["new"])
            if added or replaced:
                self.bridge.save_profile()
                if self.rag:
                    self.rag.save(path=os.path.join(self.vault_path, "rag_memory"))

        # 대화 저장
        self.save_conversation(self.current_conv_id)

        return {
            "reply":        reply,
            "new_memories": added + [p["new"] for p in replaced],
            "replaced":     replaced,
            "conv_id":      self.current_conv_id,
        }

    # ── 잎 구역 삭제 ─────────────────────────────────────────────
    def flush_memory(self):
        """잎(Leaf) 데이터와 대화 기록을 비웁니다."""
        if hasattr(self.model, 'leaf'):
            # 장난감 모델: leaf 속성 직접 초기화
            nn.init.xavier_uniform_(self.model.leaf.weight)
        elif self.config.leaf_patterns:
            # 실제 LLM: 패턴 매칭으로 leaf 구역 Linear 레이어 초기화
            reset_count = 0
            for name, module in self.model.named_modules():
                if any(p in name for p in self.config.leaf_patterns):
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        reset_count += 1
            print(f"  ({reset_count}개 레이어 초기화됨)")

        self.bridge.flush_conversation()
        print("🍂 [삭제] 일시적 정보(나뭇잎)와 대화 기록을 모두 비워 용량을 확보했습니다.")

    # ── 모델 압축 ────────────────────────────────────────────────
    def compress_model(self, amount=0.90):
        """뿌리 가중치를 프루닝 + FP16 양자화로 압축합니다."""
        FZACompressor.compress_block(
            self.model,
            amount=amount,
            zone_patterns=self.config.root_patterns,
        )

    # ── 블록 융합 ────────────────────────────────────────────────
    def fuse_blocks(self, block_a_name, block_b_name, alpha=0.5):
        """두 지식 블록을 가중 평균으로 융합하여 새 블록을 생성합니다."""
        FZAFusion.fuse_blocks(
            f"vault/{block_a_name}.fza",
            f"vault/{block_b_name}.fza",
            alpha=alpha,
        )

    # ── 상태 보고서 ──────────────────────────────────────────────
    def report_status(self):
        FZAVisualizer.print_status_report(self.model, self.math_engine)


# ── CLI 인터페이스 ────────────────────────────────────────────
def start_fza_system(
    use_local: bool = False,
    local_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
):
    manager = FZAManager(use_local=use_local, local_model_name=local_model_name)
    engine_label = f"🔥 LocalEngine ({local_model_name})" if use_local else "☁️  Claude API (Anthropic)"
    print("\n" + "="*55)
    print(f"🌿 FZA System v3.0 — 세상에 없던 망각 없는 AI")
    print(f"   추론 엔진: {engine_label}")
    print("="*55)
    print("='내 이름은 000이야'      -> 사실 학습 (루트 저장)")
    print("='평생 기억해'            -> 지식 잠금 + EWC Fisher 캡쳐")
    print("='불러와'                 -> 저장된 지식 블록 복구")
    print("='다 지워'                -> 일시 정보 삭제 (Leaf 초기화)")
    print("='상태 보고서'            -> 지식 지형도 출력")
    print("--- [AI 대화] ---")
    print("='물어봐 [질문]'          -> AI에게 질문 (루트+RAG 자동 주입)")
    print("='기억해 [텍스트]'        -> RAG 벡터 기억에 직접 추가")
    print("='기억 검색 [쿼리]'       -> 의미 유사 기억 검색")
    print("--- [수학 엔진] ---")
    print("='수식 추가 [이름] [수식]' -> 수식 등록")
    print("='수식 목록'              -> 등록된 수식 확인")
    print("--- [고급 도구] ---")
    print("='모델 압축'              -> 뿌리 가중치 프루닝+양자화")
    print("='블록 융합 [이름A] [이름B]' -> 두 블록 가중 평균 융합")
    print("='EWC 상태'              -> EWC 보호 파라미터 수 확인")
    print("="*55)

    while True:
        cmd = input("\n[명령]: ").strip()

        if "기억 검색" in cmd:
            query = cmd[cmd.index("기억 검색") + 5:].strip()
            if not query:
                print("❓ 형식: 기억 검색 [쿼리]  예) 기억 검색 내 직업")
            elif not manager.rag:
                print("⚠️ [RAG] 비활성화 상태입니다. pip install sentence-transformers faiss-cpu")
            else:
                results = manager.rag.recall(query, top_k=5)
                if results:
                    print(f"🔍 [RAG 검색 결과] '{query}' 와 유사한 기억 {len(results)}개:")
                    for i, r in enumerate(results, 1):
                        print(f"  {i}. {r}")
                else:
                    print("📭 저장된 벡터 기억이 없습니다.")
        elif "기억해" in cmd:
            text = cmd[cmd.index("기억해") + 3:].strip()
            if not text:
                print("❓ 형식: 기억해 [텍스트]  예) 기억해 나는 서울에 살고 있다")
            elif not manager.rag:
                print("⚠️ [RAG] 비활성화 상태입니다. pip install sentence-transformers faiss-cpu")
            else:
                manager.rag.add(text)
                manager.rag.save()
        elif "물어봐" in cmd:
            question = cmd[cmd.index("물어봐") + 3:].strip()
            if not question:
                print("❓ 형식: 물어봐 [질문]  예) 물어봐 내 이름이 뭐야?")
            else:
                print("🤖 [AI 응답]")
                print(manager.chat(question))
        elif "모델 압축" in cmd:
            manager.compress_model()
        elif "블록 융합" in cmd:
            parts = cmd.split()
            if len(parts) < 3:
                print("❓ 형식: 블록 융합 [이름A] [이름B]  예) 블록 융합 permanent_knowledge backup")
            else:
                manager.fuse_blocks(parts[1], parts[2])
        elif "수식 추가" in cmd or "공식 추가" in cmd:
            keyword = "수식 추가" if "수식 추가" in cmd else "공식 추가"
            rest = cmd[cmd.index(keyword) + len(keyword):].strip()
            parts = rest.split(None, 1)
            if len(parts) < 2:
                print("❓ 형식: 수식 추가 [이름] [수식]  예) 수식 추가 피타고라스 a^2+b^2=c^2")
            else:
                manager.add_formula(parts[0], parts[1])
        elif "수식 압축" in cmd or "공식 압축" in cmd:
            manager.compress_math()
        elif "수식 불러와" in cmd or "공식 불러와" in cmd:
            manager.load_math()
        elif "수식 목록" in cmd or "공식 목록" in cmd:
            manager.list_formulas()
        elif "이름은" in cmd or "정보" in cmd:
            manager.learn_info(cmd)
        elif "평생 기억해" in cmd:
            manager.lock_and_export("permanent_knowledge")
        elif "불러와" in cmd:
            manager.load_block("permanent_knowledge")
        elif "다 지워" in cmd:
            manager.flush_memory()
        elif "EWC 상태" in cmd or "ewc" in cmd.lower():
            if manager.ewc and manager.ewc.is_active:
                print(f"🔒 [EWC] 활성화 — {len(manager.ewc.fisher)}개 파라미터 보호 중 (λ={manager.ewc.ewc_lambda})")
            else:
                print("⬜ [EWC] 비활성화 — '평생 기억해' 후 자동 활성화됩니다.")
        elif "상태 보고서" in cmd or "지형도" in cmd:
            manager.report_status()
        elif "종료" in cmd:
            print("👋 시스템을 안전하게 종료합니다.")
            break
        else:
            print("❓ '이름은...', '평생 기억해', '수식 추가' 등의 명령을 내려주세요.")


if __name__ == "__main__":
    import sys
    use_local_flag = "--local" in sys.argv
    model_arg = next(
        (sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--model" and i + 1 < len(sys.argv)),
        "mistralai/Mistral-7B-Instruct-v0.3",
    )
    start_fza_system(use_local=use_local_flag, local_model_name=model_arg)
