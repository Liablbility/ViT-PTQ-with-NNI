import argparse
import torch
import torchvision
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet
import logging
import time

def validate(model, criterion, data_loader, device, print_freq):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % print_freq == 0:
                current_loss = running_loss / (i + 1)
                current_accuracy = 100.0 * correct / total
                logging.info(f"Validation Batch {i+1}/{len(data_loader)}: Loss {current_loss:.4f}, "
                             f"Accuracy {current_accuracy:.2f}%")

    avg_loss = running_loss / len(data_loader)
    avg_accuracy = 100.0 * correct / total
    return avg_loss, avg_accuracy

if __name__ == '__main__':
    # 配置 logging，保持不变
    # ...

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/jiangn')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--print_freq', type=int, default=20)

    args = parser.parse_args()

    # 图像预处理，对于验证集使用 val_transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载 ImageNet 验证集
    val_set = ImageNet(root=args.data, split='val', transform=val_transform)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batchsize, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 加载预训练的 ViT-B/16 模型
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model = model.to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 只运行验证集
    val_loss, val_accuracy = validate(model, criterion, val_loader, device, args.print_freq)
    logging.info(f"Validation Loss: {val_loss:.4f}, "
                 f"Validation Accuracy: {val_accuracy:.2f}%")


    # Save the Model
    torch.save(model.state_dict(), 'trained_vit_b_16_stanford_dogs.pth')