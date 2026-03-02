import os

import torch
import torch.nn as nn


class FZANetwork(nn.Module):
    def __init__(self):
        super(FZANetwork, self).__init__()
        self.root = nn.Linear(10, 10)
        self.leaf = nn.Linear(10, 2)
        self.current_block = "None"

    def forward(self, x):
        return self.leaf(torch.relu(self.root(x)))


class FZALibrary:
    def __init__(self):
        self.model = FZANetwork()
        if not os.path.exists("vault"):
            os.makedirs("vault")

    def save_block(self, block_name):
        """현재의 뿌리 지식을 특정 이름의 블록으로 저장"""
        path = f"vault/{block_name}.fza"
        torch.save(self.model.root.state_dict(), path)
        self.model.current_block = block_name
        print(f"📦 [도서관] '{block_name}' 지식이 보관되었습니다.")

    def load_block(self, block_name):
        """도서관에서 특정 지식 블록을 꺼내 뿌리에 이식"""
        path = f"vault/{block_name}.fza"
        if os.path.exists(path):
            self.model.root.load_state_dict(torch.load(path))
            self.model.current_block = block_name
            print(f"🧬 [이식] '{block_name}' 지식 블록을 활성화했습니다.")
        else:
            print(f"❌ [오류] '{block_name}' 블록을 찾을 수 없습니다.")


def start_system():
    """메인 실행 루프"""
    lib = FZALibrary()
    while True:
        print(f"\n[현재 활성 블록: {lib.model.current_block}]")
        cmd = input("명령(저장 [이름] / 로드 [이름] / 삭제 / 종료): ").split()

        if not cmd:
            continue

        action = cmd[0]
        if action == "저장" and len(cmd) > 1:
            lib.save_block(cmd[1])
        elif action == "로드" and len(cmd) > 1:
            lib.load_block(cmd[1])
        elif action == "삭제":
            nn.init.xavier_uniform_(lib.model.leaf.weight)
            print("🍂 잎 구역을 비웠습니다.")
        elif action == "종료":
            break


if __name__ == "__main__":
    start_system()
