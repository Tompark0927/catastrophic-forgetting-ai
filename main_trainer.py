# main_trainer.py
# 참고: model은 fza_engine.FractalZoneNetwork 인스턴스여야 합니다.
import torch.nn as nn


def train_fza(model, data, target, optimizer):
    model.train()

    # 1. 유저의 보호막(Shield) 적용 (잠긴 구역 확인)
    model.apply_user_shield()

    # 2. 학습 시작
    optimizer.zero_grad()
    output = model(data)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    # 3. 가중치 업데이트 (잠긴 Root는 변하지 않음)
    optimizer.step()

    print(f"🔥 [학습 중] 현재 손실율(Loss): {loss.item():.4f}")
