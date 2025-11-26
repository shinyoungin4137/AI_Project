import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import BetterNet
import os

def train():
    # 1. 설정 (Hyperparameters)
    BATCH_SIZE = 32      # 로컬 GPU(MX450) 메모리에 맞게 줄임
    LEARNING_RATE = 0.001
    EPOCHS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 2. 데이터 준비 (CIFAR-10)
    print("Preparing Data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 학습용 데이터 다운로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    # 3. 모델 불러오기
    model = BetterNet().to(DEVICE)
    
    # 4. 손실함수 및 최적화 도구 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 5. 학습 시작 (Training Loop)
    print("Start Training...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 변화도(Gradient) 매개변수를 0으로 만듦
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # 100 mini-batches 마다 출력
                print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print("Finished Training")

    # 6. 모델 저장
    SAVE_PATH = './better_net.pth'
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == '__main__':
    train()