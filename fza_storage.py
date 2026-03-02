# fza_storage.py (레거시 v1 — 신규 코드는 fza_storage_v2.py 사용 권장)
import os
import torch

def save_knowledge_block(model, filename="checkpoints/root_v1.pt"):
    """
    중요도가 높은 '뿌리(Root)' 지식만 따로 추출하여 파일로 저장합니다.
    """
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # 전체 모델이 아닌 'root' 레이어의 상태만 저장 (용량 최적화)
    root_data = {
        'state_dict': model.root.state_dict(),
        'importance': model.zone_importance['root']
    }
    torch.save(root_data, filename)
    print(f"💾 [저장] 핵심 지식이 {filename}에 블록화되어 저장되었습니다.")

def load_knowledge_block(model, filename="checkpoints/root_v1.pt"):
    """
    저장된 지식 블록을 불러와 모델의 뿌리에 이식합니다.
    """
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.root.load_state_dict(checkpoint['state_dict'])
        model.zone_importance['root'] = checkpoint['importance']
        print(f"🧬 [복구] {filename}으로부터 지식 씨앗을 이식했습니다.")