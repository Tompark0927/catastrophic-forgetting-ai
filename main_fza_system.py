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

        # ── v6.0: Reflex Node (Jellyfish Edge Router) ──
        try:
            from fza_reflex_node import FZAReflexNode
            self.reflex = FZAReflexNode(confidence_threshold=0.72, use_semantic=True)
            print("⚡ [ReflexNode] 젤리피시 엣지 라우터 활성화. GPU 절약 모드 ON.")
        except Exception as e:
            print(f"⚠️ [ReflexNode] 비활성화: {e}")
            self.reflex = None

        # ── v8.0: Micro Reflex Node (Dynamic Sparsity Router) ───
        try:
            from fza_micro_reflex import FZAMicroReflex
            self.micro_reflex = FZAMicroReflex(confidence_threshold=0.65)
            print("🧠 [MicroReflex] 마이크로 구조적 라우터 활성화. 컨텍스트 바이패스 대기 중.")
        except Exception as e:
            print(f"⚠️ [MicroReflex] 비활성화: {e}")
            self.micro_reflex = None

        # 시작 시 자동 복구 (silent — 파일 없어도 에러 없음)
        self.bridge.load_profile(silent=True)
        self.math_engine.load_from_seed(silent=True)

        # ── v7.0: Warm-up Reflex Node anchors using loaded profile ──
        if self.reflex:
            self.reflex.warm_up(self.bridge.user_profile)

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

        # ── v6.0: Biomimetic Reflex Node (Jellyfish Edge Router) ──
        try:
            from fza_reflex_node import FZAReflexNode
            self.reflex = FZAReflexNode(confidence_threshold=0.72, use_semantic=True)
            print("⚡ [ReflexNode] 젤리피시 엣지 라우터 활성화. GPU 절약 모드 ON.")
        except Exception as e:
            print(f"⚠️ [ReflexNode] 비활성화: {e}")
            self.reflex = None

        # ── v4.0: Adapter Router ──────────────────────────────────
        self.router = None
        if use_local and hasattr(self.bridge, 'raw_model'):
            try:
                from fza_adapter_router import FZAAdapterRouter
                router_vault = os.path.join(self.vault_path, "adapters")
                self.router = FZAAdapterRouter(
                    base_model=self.bridge.raw_model,
                    tokenizer=self.bridge.tokenizer,
                    vault_path=router_vault,
                )
                print(f"✅ [Router] 어댑터 라우터 활성화 ({self.router.adapter_count}개 기존 어댑터).")
                # v8.0: Attach router to bridge for dynamic PageRank generation
                self.bridge.router = self.router
            except Exception as e:
                print(f"⚠️ [Router] 비활성화: {e}")

        # ── v7.0: Associative Memory Graph ──────────────────────
        self.memory_graph = None
        if use_local and self.router:
            try:
                from fza_memory_graph import FZAMemoryGraph
                graph_vault = os.path.join(self.vault_path, "memory_graph")
                self.memory_graph = FZAMemoryGraph(
                    vault_path=graph_vault,
                    similarity_threshold=0.40,
                    max_neighbors=3,
                )
                self.router.memory_graph = self.memory_graph
                print(f"🕸️  [MemoryGraph] 활성화 ({self.memory_graph.node_count}노드 / {self.memory_graph.edge_count}엣지).")
            except Exception as e:
                print(f"⚠️ [MemoryGraph] 비활성화: {e}")

        # ── v7.0: Hebbian Fast-Weight Layer ─────────────────────
        self.hebbian = None
        if use_local and hasattr(self.bridge, 'raw_model'):
            try:
                from fza_hebbian_layer import FZAHebbianLayer
                cfg = getattr(self.bridge.raw_model, 'config', None)
                hidden_dim = getattr(cfg, 'hidden_size', None) or 4096
                self.hebbian = FZAHebbianLayer.inject_into(
                    self.bridge.raw_model,
                    layer_index=-1,
                    hidden_dim=hidden_dim,
                    lr=0.01,
                    decay=0.999,
                )
                hebb_path = os.path.join(self.vault_path, "hebbian_layer.pt")
                self.hebbian.load(hebb_path)
            except Exception as e:
                print(f"⚠️ [Hebbian] 비활성화: {e}")

        # ── v4.0: Smart Replay Daemon ─────────────────────────────
        self.smart_replay = None
        if use_local and hasattr(self.bridge, 'raw_model'):
            try:
                from fza_smart_replay import FZASmartReplay, ReplayMemoryBank
                rb = ReplayMemoryBank(
                    path=os.path.join(self.vault_path, "replay_bank.json")
                )
                device = str(next(self.bridge.raw_model.parameters()).device)
                self.smart_replay = FZASmartReplay(
                    model=self.bridge.raw_model,
                    tokenizer=self.bridge.tokenizer,
                    replay_bank=rb,
                    loss_tolerance=1.30,
                    idle_threshold_s=120.0,
                    max_replay_steps=5,
                    probe_interval_s=300.0,
                    device=device,
                )
                self.smart_replay.start()
            except Exception as e:
                print(f"⚠️ [SmartReplay] 비활성화: {e}")

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

        # 4. Smart Replay Bank 등록 (망각 탐침 대상으로 추가)
        if self.smart_replay:
            self.smart_replay.bank.add(text_data)
            self.smart_replay.ping()

        # 5. v7.0: Hebbian Fast-Weight instant zero-backprop registration
        if self.hebbian and hasattr(self.bridge, 'raw_model') and hasattr(self.bridge, 'tokenizer'):
            try:
                import torch
                tok = self.bridge.tokenizer
                model = self.bridge.raw_model
                inputs = tok(text_data, return_tensors="pt", truncation=True, max_length=64).to(
                    str(next(model.parameters()).device)
                )
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                hidden = out.hidden_states[-1].squeeze(0)   # (seq_len, hidden_dim)
                self.hebbian.hebbian_update(query=hidden, value=hidden)
                if self.hebbian.is_saturated:
                    print(f"⚠️ [Hebbian] 포화도 {self.hebbian.saturation:.0%} — Sleep 후 LoRA 증류 예정.")
            except Exception:
                pass

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

        # ── v4.0: Create an isolated frozen adapter for these memories ──
        if self.router:
            memories = self.bridge.user_profile.get('_memories', [])
            if memories:
                try:
                    adapter_id = self.router.create_and_freeze_adapter(memories[-10:])
                    print(f"🔐 [Router] 격리 어댑터 생성: {adapter_id[:8]}… ({len(memories[-10:])}개 사실)")
                except Exception as e:
                    print(f"⚠️ [Router] 어댑터 생성 실패: {e}")

        # v7.0: Persist Hebbian fast weights
        if self.hebbian:
            hebb_path = os.path.join(self.vault_path, "hebbian_layer.pt")
            self.hebbian.save(hebb_path)

        # v7.0: Re-warm the reflex node with the updated profile
        if self.reflex:
            self.reflex.warm_up(self.bridge.user_profile)

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
        if self.smart_replay:
            self.smart_replay.ping()   # user is active — delay replay

        # ── v6.0: Reflex Node intercept (Jellyfish) ───────────────
        if self.reflex:
            instant = self.reflex.intercept(message, self.bridge.user_profile)
            if instant is not None:
                try:
                    from fza_event_bus import bus
                    bus.emit("reflex_intercept", {"type": "v6.0_jellyfish", "query": message})
                except ImportError:
                    pass
                return instant   # LLM bypassed completely ⚡

        # ── v8.0: Micro Reflex intercept (Dynamic Sparsity) ───────
        if getattr(self, "micro_reflex", None):
            micro_action = self.micro_reflex.intercept(message)
            if micro_action is not None:
                print(f"🧠 [MicroReflex] 구조적 쿼리 감지 ('{micro_action['intent']}'). 동적 스파시티 가동 (RAG/그래프 생략).")
                try:
                    from fza_event_bus import bus
                    bus.emit("micro_reflex", {"intent": micro_action['intent']})
                except ImportError:
                    pass
                # Route directly to raw model with zero-context system prompt
                return self.bridge._generate(
                    system_prompt=micro_action["system_prompt"],
                    user_message=message
                )

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
    print("='반사 통계'              -> Jellyfish 리플렉스 노드 바이패스 통계")
    print("--- [Phase 3: Mycorrhizal Mesh] ---")
    print("='어댑터 전송'            -> 모든 LoRA를 브로커에 업로드")
    print("='어댑터 동기화'          -> 브로커에서 최신 어댑터 다운로드")
    print("='지식 병합 [alpha]'      -> EWC Fisher 대각 크로스폴리네이션")
    print("="*55)

    # ── Phase 3: Mycorrhizal Mesh Helpers ──────────────────────────────────
    BROKER_URL = "http://localhost:8001"

    def _mesh_upload_adapters():
        import requests
        from fza_sync_protocol import pack_adapter
        from fza_event_bus import bus
        if not manager.router:
            print("⬜ [Mesh] 어댑터 라우터가 없습니다.")
            return
        bank = manager.router.bank
        all_ids = bank.get_all_ids() if hasattr(bank, 'get_all_ids') else []
        if not all_ids:
            print("⬜ [Mesh] 업로드할 어댑터가 없습니다. 먼저 '평생 기억해'를 해주세요.")
            return
        uploaded = 0
        for aid in all_ids:
            meta = bank.get_meta(aid)
            adapter_path = meta.get("path", "") if meta else ""
            if not adapter_path:
                continue
            try:
                blob = pack_adapter(aid, adapter_path, metadata=meta)
                resp = requests.post(f"{BROKER_URL}/adapters/upload", json=blob, timeout=30)
                if resp.ok:
                    uploaded += 1
            except Exception as e:
                print(f"⚠️ [Mesh] 전송 실패 ({aid[:8]}): {e}")
        bus.emit("adapter_synced", {"direction": "upload", "count": uploaded})
        print(f"✅ [Mesh] {uploaded}개 어댑터가 브로커({BROKER_URL})에 전송되었습니다.")

    def _mesh_sync_adapters():
        import requests
        from fza_sync_protocol import unpack_adapter
        from fza_event_bus import bus
        try:
            resp = requests.get(f"{BROKER_URL}/adapters/list", timeout=10)
            if not resp.ok:
                print(f"⚠️ [Mesh] 브로커 연결 실패 ({BROKER_URL})")
                return
            catalog = resp.json().get("adapters", [])
            synced = 0
            for entry in catalog:
                aid = entry["adapter_id"]
                blob_resp = requests.get(f"{BROKER_URL}/adapters/{aid}", timeout=30)
                if blob_resp.ok:
                    adapter_base = getattr(manager.router, 'adapter_base', './adapters')
                    unpack_adapter(blob_resp.json(), destination_dir=adapter_base)
                    synced += 1
            bus.emit("adapter_synced", {"direction": "download", "count": synced})
            print(f"✅ [Mesh] {synced}개 어댑터를 브로커에서 동기화했습니다.")
        except Exception as e:
            print(f"⚠️ [Mesh] 동기화 실패: {e}")

    def _mesh_ewc_merge(alpha=0.3):
        from fza_ewc_merge import merge_fisher_diagonals, export_fisher, import_fisher
        from fza_event_bus import bus
        if not manager.ewc or not manager.ewc.is_active:
            print("⬜ [Mesh] EWC가 비활성화 상태입니다. 먼저 '평생 기억해'를 실행하세요.")
            return
        peer_fisher = {
            name: tensor * (1.0 + 0.1 * (hash(name) % 5 - 2))
            for name, tensor in manager.ewc.fisher.items()
        }
        root_params = list(manager.ewc.star_params.keys()) if hasattr(manager.ewc, 'star_params') else []
        merged = merge_fisher_diagonals(manager.ewc.fisher, peer_fisher, alpha=alpha, root_param_names=root_params)
        import_fisher(manager.ewc, merged)
        bus.emit("ewc_merge_complete", {"alpha": alpha, "params_merged": len(merged)})
        print(f"🌿 [Mesh] EWC 병합 완료 — α={alpha}, {len(merged)}개 파라미터 통합, 루트 보호 완료")

    # ── Phase 5 (v11.0): Hive-Mind Helpers ─────────────────────────────────────
    HIVE_BROKER = "http://localhost:8001"

    def _hive_status():
        import requests
        try:
            resp = requests.get(f"{HIVE_BROKER}/nodes/list", timeout=5)
            if not resp.ok:
                print(f"⚠️ [Hive] 브로커 연결 실패")
                return
            data = resp.json()
            nodes = data.get("nodes", [])
            print(f"\n🍄 [Hive-Mind] 라이브 노드: {len(nodes)}개 / 전체: {data.get('total_registered', 0)}개")
            for n in nodes:
                print(f"   • {n['node_id'][:8]} | {n['host']}:{n['port']} | {n['device'].upper()} | 어댑터 {n['adapter_count']}개 | 주제: {n['adapter_topics'][:3]}")
        except Exception as e:
            print(f"⚠️ [Hive] 상태 조회 실패: {e}")

    def _hive_single(query):
        from fza_distributed_inference import DistributedInferenceEngine
        die = DistributedInferenceEngine(broker_url=HIVE_BROKER)
        result = die.query_single_expert(query)
        if result and result.succeeded:
            print(f"\n🌐 [Hive-Single] 노드 {result.node_id[:8]} 응답 ({result.latency_ms:.0f}ms):\n{result.reply}")
        else:
            err = result.error if result else "없음"
            print(f"⚠️ [Hive-Single] 실패 또는 노드 없음: {err}")

    def _hive_ensemble(query):
        from fza_distributed_inference import DistributedInferenceEngine
        die = DistributedInferenceEngine(broker_url=HIVE_BROKER)
        result = die.query_ensemble(query, top_k=2)
        if result.get("merged_reply"):
            print(f"\n🌐 [Hive-앙상블] 비로드 노드 {result.get('best_node','?')[:8]} 응답:\n{result['merged_reply']}")
        else:
            print(f"⚠️ [Hive-앙상블] 반환할 노드가 없습니다.")

    def _hive_chain(query):
        from fza_distributed_inference import DistributedInferenceEngine
        die = DistributedInferenceEngine(broker_url=HIVE_BROKER)
        result = die.chain_of_thought_split(query)
        if result.get("reply"):
            print(f"\n🧠 [체인-오브-소트] 사고 프레임:\n{result.get('thinking_frame','')[:200]}")
            print(f"\n💬 최종 응답:\n{result['reply']}")
        else:
            print(f"⚠️ [체인-오브-소트] 노드가  2개 이상 필요합니다.")

    # ── Phase 7 (v13.0): Spatial Grounding Helpers ──────────────────────────
    def _get_visual_memory():
        """Returns the VisualMemoryPipeline — from engine if local, else standalone."""
        if hasattr(manager, 'bridge') and hasattr(manager.bridge, 'visual_memory') and manager.bridge.visual_memory:
            return manager.bridge.visual_memory
        # Fallback: standalone instance
        from fza_visual_memory import VisualMemoryPipeline
        return VisualMemoryPipeline()

    def _spatial_observe(text):
        """'봐봐' — ingest a scene description into spatial memory."""
        vm = _get_visual_memory()
        result = vm.ingest_description(text)
        if result["objects_detected"]:
            print(f"\n🏠 [공간 기억] 저장 완료:")
            for obj in result["objects_detected"]:
                print(f"   • {obj} → {result['location'].replace('_', ' ')}")
        else:
            print("⚠️ [공간 기억] 객체를 인식하지 못했습니다. 더 구체적으로 말해줘 (예: '책상 위에 노트북이 있어')")

    def _spatial_locate(query):
        """'어디 있어' — answer a "where is X?" query from spatial memory."""
        vm = _get_visual_memory()
        answer = vm.answer_location_query(query)
        print(f"\n📍 [공간 기억] {answer}")

    def _spatial_map():
        """'공간 지도' — show the full world graph."""
        vm = _get_visual_memory()
        summary = vm.world_graph.build_natural_language_summary()
        stats = vm.get_stats()
        print(f"\n🗺️  [세계 지도] 총 {stats['world_graph_objects']}개 객체, {stats['spatial_adapters']}개 공간 어댑터")
        if summary:
            print(summary)
        else:
            print("   (아직 공간 정보가 없습니다. '봐봐, [위치]에 [물건]이 있어'로 알려줘!)")

    # ── Phase 8 (v14.0): Singularity Threshold Helpers ──────────────────────
    def _get_self_architect():
        """Returns SelfArchitect from engine (if available) or standalone."""
        if hasattr(manager, 'bridge') and hasattr(manager.bridge, 'self_architect') and manager.bridge.self_architect:
            return manager.bridge.self_architect
        from fza_self_architect import SelfArchitect
        return SelfArchitect(model=None)

    def _sa_reflect():
        """'자가 진단' — full architectural self-assessment."""
        sa = _get_self_architect()
        sa.reflect()

    def _sa_spawn(domain):
        """'로브 생성 [도메인]' — manually spawn a new domain lobe."""
        if not domain:
            print("❓ 형식: 로브 생성 [도메인]  예) 로브 생성 quantum_physics")
            return
        sa = _get_self_architect()
        lobe_id = sa.spawn_lobe(domain)
        sa.spawner.activate(lobe_id, target_gate=0.1)
        print(f"🧠 [SelfArchitect] '{domain}' 로브 생성 및 웜-업 시작: {lobe_id[:8]}")

    def _sa_lobes():
        """'로브 목록' — list all domain lobes."""
        sa = _get_self_architect()
        lobes = sa.spawner.list_lobes()
        if not lobes:
            print("🧠 [LobeSpawner] 생성된 로브 없음 — '로브 생성 [도메인]'으로 시작하세요")
            return
        print(f"\n🧠 [LobeSpawner] {len(lobes)}개 로브:")
        for l in lobes:
            print(f"   • {l['domain']:<25} {l['status']:<10} gate={l['gate_value']:.2f} "
                  f"활성화={l['activation_count']}회 [{l['lobe_id'][:8]}]")

    def _sa_kernels():
        """'커널 보고서' — show KernelForge bottleneck report."""
        sa = _get_self_architect()
        sa.forge.print_report()

    # ── Phase 9 (v15.0): OS Symbiosis Helpers ───────────────────────────────
    def _get_os_agent():
        """Returns OSAgent from engine (if available) or a dry_run standalone."""
        if hasattr(manager, 'bridge') and getattr(manager.bridge, 'os_agent', None):
            return manager.bridge.os_agent
        from fza_procedural_memory import ProceduralMemory
        from fza_os_agent import OSAgent
        pm = ProceduralMemory()
        return OSAgent(procedural_memory=pm, dry_run=True)

    def _os_observe():
        """'화면 관찰' — take a screenshot and describe the current screen."""
        agent = _get_os_agent()
        desc = agent.observe()
        print(f"\n👁️  [OSAgent] 화면 설명:\n{desc}")

    def _os_execute(goal):
        """'실행 [목표]' — execute an OS task."""
        if not goal:
            print("❓ 형식: 실행 [목표]  예) 실행 Safari 열기")
            return
        agent = _get_os_agent()
        result = agent.execute_task(goal)
        print(f"\n{'✅' if result['success'] else '❌'} [OSAgent] '{goal}' | "
              f"단계: {result['steps_executed']} | 화면 변화: {result.get('screen_changed', '-')}")

    def _os_procedures():
        """'프로시저 목록' — list all learned OS workflows."""
        from fza_procedural_memory import ProceduralMemory
        pm = _get_os_agent().procedural_memory or ProceduralMemory()
        pm.print_summary()

    def _os_enable_control():
        """'컴퓨터 제어 활성화' — switch OSAgent from dry_run to live mode."""
        agent = _get_os_agent()
        agent.executor.dry_run = False
        print("⚠️  [OSAgent] 🔴 실제 컴퓨터 제어 활성화됨 — 'PyAutoGUI FAILSAFE: 마우스 좌상단 이동시 중단'")

    # ── Phase 10 (v16.0): Swarm Replication Helpers ─────────────────────────
    _swarm = None
    def _get_swarm():
        nonlocal _swarm
        if _swarm is None:
            from fza_swarm_provisioner import SwarmProvisioner
            _swarm = SwarmProvisioner(backend="mock")
        return _swarm

    def _swarm_spawn(label):
        """'세포분열 [이유]' — spawn a new child node."""
        label = label or "generic overflow node"
        sp = _get_swarm()
        node_id = sp.spawn_child(label)
        if node_id:
            import time; time.sleep(0.6)   # Wait for background thread
            print(f"\n🦠 [Swarm] 분열 완료: [{node_id[:8]}] '{label}'")
        else:
            print("⚠️  [Swarm] 최대 노드 수 도달 또는 분열 실패")

    def _swarm_status():
        """'떼 상태' — show all active swarm nodes."""
        _get_swarm().print_swarm_status()

    def _swarm_cull():
        """'유휴 종료' — cull idle nodes."""
        sp = _get_swarm()
        culled = sp.cull_idle_nodes(max_idle_seconds=60)
        if culled:
            print(f"🗑️  [Swarm] {len(culled)}개 유휴 노드 종료됨")
        else:
            print("🦠 [Swarm] 종료할 유휴 노드 없음")

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
        # ── Phase 3: Mycorrhizal Mesh Commands ──────────────────────────────
        elif "어댑터 전송" in cmd or "메시 전송" in cmd:
            _mesh_upload_adapters()
        elif "어댑터 동기화" in cmd or "메시 동기화" in cmd:
            _mesh_sync_adapters()
        elif "지식 병합" in cmd:
            _a = 0.3
            for _p in cmd.split():
                try: _a = float(_p); break
                except ValueError: pass
            _mesh_ewc_merge(alpha=max(0.01, min(0.9, _a)))
        # ── Phase 5 (v11.0): Hive-Mind Commands ─────────────────────────────
        elif "하이브 상태" in cmd or "hive" in cmd.lower():
            _hive_status()
        elif "분산 추론" in cmd:
            q = cmd[cmd.index("분산 추론") + 5:].strip()
            if q: _hive_single(q)
            else: print("❓ 형식: 분산 추론 [질문]")
        elif "앙상블 추론" in cmd:
            q = cmd[cmd.index("앙상블 추론") + 6:].strip()
            if q: _hive_ensemble(q)
            else: print("❓ 형식: 앙상블 추론 [질문]")
        elif "체인 추론" in cmd:
            q = cmd[cmd.index("체인 추론") + 5:].strip()
            if q: _hive_chain(q)
            else: print("❓ 형식: 체인 추론 [질문]")
        # ── Phase 7 (v13.0): Spatial Grounding Commands ─────────────────────
        elif "봐봐" in cmd:
            scene = cmd[cmd.index("봐봐") + 3:].strip()
            if scene: _spatial_observe(scene)
            else: print("❓ 형식: 봐봐, [위치]에 [물건]이 있어  예) 봐봐, 책상 위에 노트북이 있어")
        elif "어디 있어" in cmd or "어디에 있어" in cmd:
            _spatial_locate(cmd)
        elif "공간 지도" in cmd or "세계 지도" in cmd:
            _spatial_map()
        # ── Phase 8 (v14.0): Singularity Threshold Commands ─────────────────
        elif "자가 진단" in cmd or "self reflect" in cmd.lower():
            _sa_reflect()
        elif "로브 생성" in cmd:
            domain = cmd[cmd.index("로브 생성") + 5:].strip()
            _sa_spawn(domain)
        elif "로브 목록" in cmd:
            _sa_lobes()
        elif "커널 보고서" in cmd:
            _sa_kernels()
        # ── Phase 9 (v15.0): OS Symbiosis Commands ──────────────────────────
        elif "화면 관찰" in cmd or "스크린샷" in cmd:
            _os_observe()
        elif cmd.startswith("실행 "):
            _os_execute(cmd[3:].strip())
        elif "프로시저 목록" in cmd:
            _os_procedures()
        elif "컴퓨터 제어 활성화" in cmd:
            _os_enable_control()
        # ── Phase 10 (v16.0): Swarm Replication Commands ────────────────────
        elif "세포분열" in cmd:
            label = cmd[cmd.index("세포분열") + 4:].strip()
            _swarm_spawn(label or "overflow node")
        elif "떼 상태" in cmd or "swarm" in cmd.lower():
            _swarm_status()
        elif "유휴 종료" in cmd:
            _swarm_cull()
        # ── Phase 11 (v17.0): Latent Telepathy Commands ─────────────────────
        elif "텔레파시 시작" in cmd:
            # '텔레파시 시작 9200' — start a TelepathyNode listener on the given port
            raw = cmd.split()
            port = int(raw[-1]) if raw[-1].isdigit() else 9200
            from fza_latent_telepathy import TelepathyNode
            _tp_node = TelepathyNode(host="0.0.0.0", port=port)
            _tp_node.start_listener(
                on_receive=lambda t, addr: print(f"\n🧠← [텔레파시] {addr}로부터 수신: {t.shape}")
            )
            print(f"📡 [텔레파시] 리스너 시작 @ port {port}")
        elif "텔레파시 전송" in cmd:
            # '텔레파시 전송 [port]' — send a random test vector to loopback
            raw = cmd.split()
            port = int(raw[-1]) if raw[-1].isdigit() else 9200
            import torch
            from fza_latent_telepathy import TelepathyNode
            _tp_send = TelepathyNode(host="0.0.0.0", port=9299)
            vec = torch.randn(8, 256)
            _tp_send.send_thought(vec, peer_host="127.0.0.1", peer_port=port)
        elif "텔레파시 상태" in cmd:
            print("📡 [텔레파시] 모듈 로드 확인 중...")
            from fza_latent_telepathy import TelepathyNode
            from fza_vector_compression import compress_payload, compression_ratio
            import torch
            test = torch.randn(4, 256)
            enc = compress_payload(test)
            ratio = compression_ratio(test, enc)
            print(f"   fp16 압축률: {ratio:.1f}x | 패킷 크기: {len(enc)}B | 준비 완료 ✅")
        # ── Phase 12 (v18.0): Physical Embodiment Commands ──────────────────
        elif "로봇 실행" in cmd:
            intent = cmd[cmd.index("로봇 실행") + 5:].strip()
            if not intent:
                print("❓ 형식: 로봇 실행 [동작]  예) 로봇 실행 책상으로 이동해")
            else:
                from fza_motor_cortex import MotorCortex
                from fza_ros2_bridge import FZAROS2Bridge
                _bridge = FZAROS2Bridge(robot_name="fza_bot", dry_run=True)
                _mc = MotorCortex(world_graph=None, ros2_bridge=_bridge)
                result = _mc.execute_intent(intent)
                print(f"{'✅' if result['success'] else '❌'} [로봇] '{result['skill_name']}' | {result['commands_executed']}개 명령")
        elif "운동 기억" in cmd:
            from fza_motor_cortex import MotorAdapterBank
            MotorAdapterBank().print_summary()
        elif "로봇 상태" in cmd:
            from fza_ros2_bridge import FZAROS2Bridge, _ROS2_AVAILABLE
            bridge_stat = FZAROS2Bridge(robot_name="fza_bot", dry_run=True)
            bridge_stat.print_status()
        # ── Phase XIV (v20.0): Synthetic Natural Selection ──────────────────────
        elif "진화" in cmd and "계보" in cmd:
            # '진화 계보' — show evolutionary lineage
            from fza_evolution_arena import EvolutionArena
            _arena = EvolutionArena(dry_run=True)
            _arena.print_lineage()
        elif "진화" in cmd:
            # '진화' — trigger one evolutionary cycle
            def _run_evolution():
                pop = 5
                dry = "--live" not in cmd.lower()
                from fza_evolution_arena import EvolutionArena
                _arena = EvolutionArena(
                    seed_path="./jellyfish_seed/core_seed.json",
                    workspace_dir=".",
                    population=pop,
                    dry_run=dry,
                )
                winner = _arena.run()
                print(f"\n🧬 진화 완료! 승자: {winner['mutation_id']} | 점수: {winner['winning_score']:.3f}")
                print(f"   새 하이퍼파라미터: {winner.get('mutation_params', {})}")
            import threading; threading.Thread(target=_run_evolution, daemon=True).start()
        # ── Phase XV (v21.0): P2P Mesh Intelligence ─────────────────────────────
        elif "메시 시작" in cmd:
            def _start_mesh():
                from fza_mesh_node import MeshNode
                mesh_port = 10001
                _mesh = MeshNode(
                    node_id=f"fza-{__import__('socket').gethostname()}",
                    port=mesh_port,
                    on_query=lambda q: manager.engine.chat(q) if hasattr(manager, 'engine') else "메시 응답",
                )
                _mesh.start()
                manager._mesh_node = _mesh
                print(f"🕸️  메시 시작 완료. 포트: {mesh_port}")
            import threading; threading.Thread(target=_start_mesh, daemon=True).start()
        elif "메시 상태" in cmd:
            mesh = getattr(manager, '_mesh_node', None)
            if mesh:
                mesh.print_status()
            else:
                print("🕸️  메시 없음 — '메시 시작'을 먼저 실행하세요.")
        elif "메시 쿼리" in cmd:
            mesh = getattr(manager, '_mesh_node', None)
            if mesh:
                q = cmd.replace("메시 쿼리", "").strip()
                responses = mesh.query_mesh(q)
                if responses:
                    for r in responses:
                        print(f"🕸️  [{r['node_id']}]: {r['answer']}")
                else:
                    print("🕸️  응답 없음 (피어 없음 또는 타임아웃)")
            else:
                print("🕸️  메시 없음 — '메시 시작'을 먼저 실행하세요.")
        # ── Phase XVI (v22.0): Post-Biological Kernel Forge ─────────────────────
        elif "단조 목록" in cmd:
            from fza_kernel_forge import KernelForge
            KernelForge(dry_run=True).print_ledger()
        elif "단조" in cmd:
            # Usage: 단조 cosine_similarity 4096
            parts = cmd.split()
            kernel = "cosine_similarity"
            dim = 4096
            for p in parts:
                if p.isdigit():
                    dim = int(p)
                elif p in ("cosine_similarity", "dot_product", "softmax", "l2_normalize", "euclidean_distance"):
                    kernel = p
            def _forge():
                live = "--live" in cmd.lower()
                from fza_kernel_forge import KernelForge
                f = KernelForge(dry_run=not live)
                fn = f.profile_and_compile(kernel, dim=dim, benchmark=True)
                if fn:
                    print(f"⚙️  '{kernel}_{dim}' 단조 완료! 함수 핫스왑 가능")
                f.print_ledger()
            import threading; threading.Thread(target=_forge, daemon=True).start()
        elif "반사 통계" in cmd or "reflex" in cmd.lower():
            if manager.reflex:
                manager.reflex.print_stats()
            else:
                print("⬜ [ReflexNode] 비활성화.")
            if getattr(manager, "micro_reflex", None):
                manager.micro_reflex.print_stats()
        elif "그래프" in cmd or "graph" in cmd.lower():
            if manager.memory_graph:
                manager.memory_graph.print_graph_summary()
            else:
                print("⬜ [MemoryGraph] 비활성화.")
        elif "헤비안" in cmd or "hebbian" in cmd.lower():
            if manager.hebbian:
                print(f"🧬 [Hebbian] 업데이트 {manager.hebbian.update_count}회 / 포화도 {manager.hebbian.saturation:.1%}")
            else:
                print("⬜ [Hebbian] 비활성화.")
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
