"""
FZA LLM Bridge — Stage 1 + Stage 2 + Stage 3 (RAG) + Stage 4 (Auto-Memory)
────────────────────────────────────────────────────────────
Stage 1: math_vault 가 LLM 응답을 override
Stage 2: FZA 지식 구역 → 개인화 시스템 프롬프트
Stage 3: RAG — 의미 검색으로 관련 기억만 꺼내 주입
Stage 4: Auto-Memory — 대화 후 중요한 사실을 자동 추출하여 저장
────────────────────────────────────────────────────────────
"""
import json
import os
from datetime import datetime


class FZALLMBridge:
    def __init__(self, math_engine, user_profile=None, memory=None, vault_path="vault"):
        """
        Args:
            memory:     FZAMemory 인스턴스 (RAG). None 이면 Stage 3 비활성화.
            vault_path: 프로필 저장 디렉토리 (멀티유저: vault/users/{user_id})
        """
        self._client = None
        self.math_engine = math_engine
        self.user_profile = user_profile or {}
        self.conversation_history = []
        self.memory = memory
        self.vault_path = vault_path

    # ── Anthropic 클라이언트 (lazy) ──────────────────────────
    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    # ── 루트 지식 관리 ───────────────────────────────────────
    def set_user_fact(self, key: str, value: str):
        """user_profile에 영구 사실 저장 (root 구역)"""
        self.user_profile[key] = value
        print(f"🧠 [루트 지식] '{key}: {value}' 개인화 프로필에 저장되었습니다.")

    @staticmethod
    def _categorize(text: str) -> str:
        """키워드 기반 기억 카테고리 자동 분류"""
        t = text.lower()
        if any(k in t for k in ['나이', '생일', '이름', '성별', '출생']): return '신상'
        if any(k in t for k in ['직업', '일', '회사', '학교', '전공', '대학', '공부']): return '직업'
        if any(k in t for k in ['가족', '친구', '연인', '부모', '형제', '결혼']): return '관계'
        if any(k in t for k in ['목표', '꿈', '하고싶', '계획', '원해', '바라']): return '목표'
        if any(k in t for k in ['건강', '운동', '병', '다이어트', '체중']): return '건강'
        if any(k in t for k in ['취미', '즐기', '관심', '게임', '음악', '영화', '독서', '좋아']): return '취미'
        if any(k in t for k in ['사는', '살고', '거주', '이사']): return '장소'
        return '기타'

    def add_memory(self, text: str):
        """자유 형식 텍스트를 일반 기억으로 추가합니다."""
        mems  = self.user_profile.setdefault('_memories', [])
        dates = self.user_profile.setdefault('_memory_dates', [])
        cats  = self.user_profile.setdefault('_memory_cats', [])
        if text not in mems:
            mems.append(text)
            dates.append(datetime.now().isoformat(timespec='seconds'))
            cats.append(self._categorize(text))
            print(f"💡 [일반 기억] '{text[:50]}' 저장됨. (총 {len(mems)}개)")

    def save_profile(self):
        """루트 지식을 파일로 영구 저장"""
        os.makedirs(self.vault_path, exist_ok=True)
        path = os.path.join(self.vault_path, "user_profile.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.user_profile, f, ensure_ascii=False, indent=2)
        print(f"💾 [프로필 저장] {path}")

    def load_profile(self, silent=False):
        """저장된 루트 지식 복구."""
        path = os.path.join(self.vault_path, "user_profile.json")
        if not os.path.exists(path):
            if not silent:
                print("❌ [프로필] 저장된 프로필이 없습니다.")
            return
        with open(path, "r", encoding="utf-8") as f:
            self.user_profile = json.load(f)
        structured = {k: v for k, v in self.user_profile.items() if not k.startswith('_')}
        memories = self.user_profile.get('_memories', [])
        print(f"📂 [프로필 복구] 구조화 {len(structured)}개 / 일반 기억 {len(memories)}개 로드 완료.")
        for k, v in structured.items():
            print(f"  · {k}: {v}")

    # ── Stage 2+3: 시스템 프롬프트 빌더 ─────────────────────
    def _build_system_prompt(self, query: str = None) -> str:
        parts = [
            "당신은 사용자를 오래 알아온 개인 AI 어시스턴트입니다.\n"
            "아래에 이 사용자에 대해 알고 있는 정보가 있습니다. 대화할 때 자연스럽게 활용하세요.\n"
        ]

        # [루트 구역] 영구 사용자 프로필 (key-value 사실)
        structured = {k: v for k, v in self.user_profile.items() if not k.startswith('_')}
        if structured:
            parts.append("[사용자 프로필]")
            for k, v in structured.items():
                parts.append(f"- {k}: {v}")
            parts.append("")

        # [일반 기억 구역] 자유 형식 기억 (최근 30개)
        memories = self.user_profile.get('_memories', [])
        if memories:
            parts.append("[기억하고 있는 사실들]")
            for mem in memories[-30:]:
                parts.append(f"- {mem}")
            parts.append("")

        # [RAG 구역] 의미 검색으로 꺼낸 관련 기억 (Stage 3)
        if self.memory and query:
            recalled = self.memory.recall(query, top_k=3)
            if recalled:
                parts.append("[지금 대화와 관련된 기억]")
                for mem in recalled:
                    parts.append(f"- {mem}")
                parts.append("")

        # [수학 구역]
        if self.math_engine.math_vault:
            parts.append("[수식 — 100% 정확도 보장]")
            for name, formula in self.math_engine.math_vault.items():
                parts.append(f"- {name}: {formula}")
            parts.append("")

        parts.append("위의 모든 사실은 절대적으로 정확합니다. 대화 중 사용자가 언급한 것과 위 정보가 맞으면 자연스럽게 활용하세요.")
        return "\n".join(parts)

    # ── Stage 1 + 2 + 3: 통합 채팅 ─────────────────────────
    def chat(self, user_message: str) -> str:
        # Stage 1 — math_vault override
        for name, formula in self.math_engine.math_vault.items():
            if name in user_message:
                return f"📐 {name}: {formula}"

        # Stage 2+3 — 개인화 + RAG Claude 호출
        self.conversation_history.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=self._build_system_prompt(query=user_message),
            messages=self.conversation_history,
        )

        reply = response.content[0].text
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    # ── Stage 4: 자동 기억 추출 ──────────────────────────────
    def auto_extract_memory(self, user_msg: str, ai_response: str) -> list:
        """대화에서 사용자에 관한 기억할 만한 사실을 자동 추출합니다.
        빠른 Haiku 모델 사용 — 비용 최소화."""
        prompt = (
            "다음 대화에서 사용자에 대해 기억해둘 만한 구체적인 사실만 추출해줘.\n"
            "예: 이름, 직업, 나이, 사는 곳, 취미, 좋아하는 것, 목표, 중요한 개인 정보.\n"
            "일반 질문/답변, 날씨, 일반 상식은 제외해줘.\n"
            "없으면 빈 배열 []만 반환. JSON 배열 형식으로만 답해줘.\n\n"
            f"사용자: {user_msg}\n"
            f"AI: {ai_response[:400]}"
        )
        try:
            resp = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # JSON 배열 파싱 (앞뒤에 불필요한 텍스트 제거)
            start = raw.find('[')
            end = raw.rfind(']') + 1
            if start >= 0 and end > start:
                facts = json.loads(raw[start:end])
                return facts if isinstance(facts, list) else []
        except Exception:
            pass
        return []

    # ── Stage 4b: 스마트 기억 병합 ───────────────────────────
    def smart_merge_memories(self, new_facts: list) -> tuple:
        """새 사실을 기존 기억과 스마트하게 병합합니다.

        같은 속성(나이, 이름, 직업 등)에 대한 새 정보 → 기존 것 교체 (update)
        전혀 새로운 정보                              → 추가 (add)

        Returns: (added: list[str], replaced: list[{"old", "new"}])
        """
        existing = self.user_profile.get('_memories', [])

        # 기존 기억이 없으면 그냥 전부 추가
        if not existing:
            for fact in new_facts:
                self.add_memory(fact)
            return new_facts, []

        prompt = (
            "기존 기억 목록과 새 정보를 비교해서 각 새 정보가 기존 기억을 교정하는 것인지 판단해줘.\n\n"
            "판단 기준:\n"
            "- 같은 속성(나이, 이름, 직업, 사는 곳 등)에 대해 값이 달라졌다 → update\n"
            "- 기존 기억에 없는 새로운 속성/사실이다 → add\n\n"
            f"기존 기억:\n{json.dumps(existing, ensure_ascii=False)}\n\n"
            f"새 정보:\n{json.dumps(new_facts, ensure_ascii=False)}\n\n"
            "JSON만 반환 (다른 텍스트 없이):\n"
            '{"actions":[{"type":"add","fact":"새 정보"} 또는 {"type":"update","old":"기존 기억 텍스트 그대로","new":"새 정보"}]}'
        )
        try:
            resp = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            start = raw.find('{')
            end   = raw.rfind('}') + 1
            if start < 0 or end <= start:
                raise ValueError("JSON not found")
            data = json.loads(raw[start:end])

            mems    = self.user_profile.setdefault('_memories', [])
            added   = []
            replaced = []

            for action in data.get("actions", []):
                atype = action.get("type")
                if atype == "add":
                    fact = action.get("fact", "").strip()
                    if fact and fact not in mems:
                        self.add_memory(fact)   # dates + cats 자동 추가
                        added.append(fact)
                elif atype == "update":
                    old = action.get("old", "").strip()
                    new = action.get("new", "").strip()
                    if not new:
                        continue
                    # 기존 기억에서 old 제거 (정확 매치 → 부분 매치 순)
                    rm_done = False
                    if old in mems:
                        idx = mems.index(old)
                        mems.pop(idx)
                        for pkey in ('_memory_dates', '_memory_cats'):
                            pl = self.user_profile.get(pkey, [])
                            if idx < len(pl): pl.pop(idx)
                        rm_done = True
                    else:
                        # 앞 15자 기준 부분 매치
                        prefix = old[:15]
                        for i, m in enumerate(mems):
                            if prefix and (prefix in m or m[:15] in old):
                                old = mems.pop(i)   # 실제 제거된 텍스트로 교체
                                for pkey in ('_memory_dates', '_memory_cats'):
                                    pl = self.user_profile.get(pkey, [])
                                    if i < len(pl): pl.pop(i)
                                rm_done = True
                                break
                    self.add_memory(new)   # dates + cats 자동 추가
                    replaced.append({"old": old, "new": new})
                    if rm_done:
                        print(f"🔄 [기억 수정] '{old[:30]}' → '{new[:30]}'")
                    else:
                        print(f"💡 [기억 추가] '{new[:50]}'")

            return added, replaced

        except Exception:
            # 실패 시 안전하게 그냥 추가
            for fact in new_facts:
                self.add_memory(fact)
            return new_facts, []

    # ── 기억 / 프로필 직접 편집 ───────────────────────────────────
    def delete_memory(self, index: int) -> bool:
        """_memories 리스트에서 특정 인덱스의 기억을 삭제합니다."""
        mems = self.user_profile.get('_memories', [])
        if 0 <= index < len(mems):
            removed = mems.pop(index)
            for pkey in ('_memory_dates', '_memory_cats'):
                pl = self.user_profile.get(pkey, [])
                if index < len(pl):
                    pl.pop(index)
            print(f"🗑 [기억 삭제] '{removed[:40]}'")
            return True
        return False

    def edit_memory(self, index: int, new_text: str) -> bool:
        """_memories 리스트에서 특정 인덱스의 기억을 수정합니다."""
        mems = self.user_profile.get('_memories', [])
        if 0 <= index < len(mems) and new_text.strip():
            old = mems[index]
            mems[index] = new_text.strip()
            cats = self.user_profile.get('_memory_cats', [])
            if index < len(cats):
                cats[index] = self._categorize(new_text.strip())
            print(f"✏️ [기억 수정] '{old[:30]}' → '{new_text.strip()[:30]}'")
            return True
        return False

    def delete_profile_key(self, key: str) -> bool:
        """user_profile에서 특정 키를 삭제합니다 (_memories 제외)."""
        if key in self.user_profile and key != '_memories':
            del self.user_profile[key]
            print(f"🗑 [프로필 삭제] '{key}'")
            return True
        return False

    def edit_profile_key(self, key: str, value: str) -> bool:
        """user_profile의 특정 키 값을 수정합니다."""
        if key and key != '_memories':
            self.user_profile[key] = value.strip()
            print(f"✏️ [프로필 수정] '{key}' = '{value.strip()[:30]}'")
            return True
        return False

    # ── 파일 처리 ─────────────────────────────────────────────
    def process_file(self, filename: str, content_b64: str, mime_type: str) -> str:
        """업로드된 파일에서 내용을 추출하고 Haiku로 요약합니다."""
        import base64

        if mime_type.startswith('image/'):
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": content_b64}},
                    {"type": "text", "text": "이 이미지의 내용을 한국어로 상세히 설명해줘."},
                ]}],
            )
            return response.content[0].text

        try:
            raw = base64.b64decode(content_b64)
        except Exception:
            raw = content_b64.encode()

        if mime_type == 'application/pdf' or filename.lower().endswith('.pdf'):
            try:
                from pypdf import PdfReader
                import io as _io
                text = '\n'.join(p.extract_text() or '' for p in PdfReader(_io.BytesIO(raw)).pages)
            except ImportError:
                try:
                    from PyPDF2 import PdfReader
                    import io as _io
                    text = '\n'.join(p.extract_text() or '' for p in PdfReader(_io.BytesIO(raw)).pages)
                except ImportError:
                    return "❌ PDF 파싱을 위해 'pip install pypdf'를 실행해주세요."
        else:
            text = raw.decode('utf-8', errors='replace')

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content":
                f"다음 파일을 한국어로 요약해줘:\n\n파일: {filename}\n\n{text[:4000]}"}],
        )
        return response.content[0].text

    # ── 잎 구역 삭제 ─────────────────────────────────────────
    def flush_conversation(self):
        """대화 기록(Leaf) 초기화 — 기억은 유지"""
        self.conversation_history = []
        print("🍂 [초기화] 대화 기록을 비웠습니다. 기억은 유지됩니다.")

    # ── 상태 출력 ────────────────────────────────────────────
    def print_status(self):
        structured = {k: v for k, v in self.user_profile.items() if not k.startswith('_')}
        print(f"👤 루트 프로필: {len(structured)}개 항목")
        for k, v in structured.items():
            print(f"  · {k}: {v}")
        memories = self.user_profile.get('_memories', [])
        print(f"💡 일반 기억: {len(memories)}개")
        rag_count = len(self.memory) if self.memory else 0
        print(f"🧠 RAG 벡터 기억: {rag_count}개")
        print(f"💬 현재 대화 턴: {len(self.conversation_history) // 2}회")
