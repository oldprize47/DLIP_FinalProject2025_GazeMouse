# test_fginet.py

import torch
from torch.utils.data import DataLoader
from fginet import FGINet
from dataset_module import GazeScreenDataset  # 앞서 구현한 Dataset


def main():
    # 1) 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) 데이터로더 준비 (작게, 디버그용으로 batch_size=4 등)
    csv_path = "C:/Users/Sangheon/source/repos/DLIP/DLIP_FinalProject2025_GazeMouse/mpiigaze/mpiigaze_labels.csv"
    dataset = GazeScreenDataset(csv_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # 3) 모델 인스턴스화 및 이동
    model = FGINet().to(device)
    model.eval()  # 순전파만 테스트

    # (옵션) 학습된 가중치가 있다면 로드
    # ckpt = torch.load("fgi_checkpoint.pth", map_location=device)
    # model.load_state_dict(ckpt["model_state_dict"])

    # 4) 한 배치만 꺼내서 순전파
    imgs, labels = next(iter(loader))
    imgs = imgs.to(device)  # (B,3,224,224)
    labels = labels.to(device)  # (B,2)

    with torch.no_grad():
        preds = model(imgs)  # (B,2)

    # 5) 출력 모양·범위·오차 확인
    print("inputs:", imgs.shape)
    print("labels:", labels.shape, labels.cpu().numpy())
    print("preds:", preds.shape, preds.cpu().numpy())
    print("preds min/max:", preds.min().item(), preds.max().item())

    # 간단한 MSE 계산
    mse = torch.nn.functional.mse_loss(preds, labels)
    print(f"MSE on this batch: {mse.item():.4f}")

    # 6) 역전파 그래디언트 테스트 (옵션)
    # 모델.train()
    # imgs.requires_grad_(True)
    # preds2 = model(imgs)
    # loss = torch.nn.functional.mse_loss(preds2, labels)
    # loss.backward()
    # print("Gradients OK, e.g. first layer grad norm:",
    #       model.stage1.conv1.weight.grad.norm().item())


if __name__ == "__main__":
    main()
