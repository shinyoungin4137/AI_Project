import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
from model import BetterNet  # model.py에서 클래스 불러오기

# CIFAR-10 클래스
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_model(model_path, device):
    """모델 로드 함수"""
    model = BetterNet().to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    # CPU/GPU 호환 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path):
    """이미지 전처리 함수"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error: Cannot open image '{image_path}'. ({e})")
        return None
    return transform(image).unsqueeze(0)


def main():
    # 1. 사용법 확인
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        print("Example: python inference.py test_dog.jpg")
        return

    image_path = sys.argv[1]

    # 2. 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 모델 로드
    model_path = "./better_net.pth"  # 가중치 파일 경로
    model = load_model(model_path, device)

    # 4. 추론
    input_tensor = preprocess_image(image_path)
    if input_tensor is not None:
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

            result = CLASSES[predicted_idx.item()]
            score = confidence.item() * 100

            print(f"\n[Result]")
            print(f"Image: {image_path}")
            print(f"Prediction: {result}")
            print(f"Confidence: {score:.2f}%")


if __name__ == "__main__":
    main()