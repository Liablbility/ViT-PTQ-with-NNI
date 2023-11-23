import argparse
import time
import torch
import torchvision
from nni.compression.utils import Evaluator
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data import StanfordDogs
import logging

def train_one_epoch(model, criterion, optimizer,scheduler, data_loader, device, print_freq):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % print_freq == 0:
            current_loss = running_loss / (i + 1)
            current_accuracy = 100.0 * correct / total
            logging.info(f"Batch {i+1}/{len(data_loader)}: Loss {current_loss:.4f}, Accuracy {current_accuracy:.2f}%")

    avg_loss = running_loss / len(data_loader)
    avg_accuracy = 100.0 * correct / total
    scheduler.step()
    return avg_loss, avg_accuracy


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
                logging.info(f"Validation Batch {i+1}/{len(data_loader)}: Loss {current_loss:.4f}, Accuracy {current_accuracy:.2f}%")

    avg_loss = running_loss / len(data_loader)
    avg_accuracy = 100.0 * correct / total
    return avg_loss, avg_accuracy

def evaluate_fn(model, data_loader):
    model.eval()
    start_time = time.time()  # 记录开始时间
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    end_time = time.time()  # 记录结束时间
    elapsed_time = (end_time - start_time) * 1000  # 计算运行时间，转换为毫秒
    # print("Evaluation time: {:.2f} ms".format(elapsed_time))
    return accuracy, elapsed_time
class MyEvaluator(Evaluator):
    def __init__(self, model, val_loader):
        super().__init__()
        self.model = model
        self.val_loader = val_loader
        self._initialization_complete = True  #

    def bind_model(self, model, param_names_map):
        self.model = model

    def unbind_model(self):
        # 如果没有特殊的解绑操作，此方法可以保持空
        pass

    def train(self, max_steps=None, max_epochs=None):
        # PTQ 不需要训练步骤
        pass

    def evaluate(self):
        accuracy, elapsed_time = evaluate_fn(self.model, self.val_loader)
        return accuracy

if __name__ == '__main__':
    # 保存原始的标准输出引用
    # 配置 logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("./logs/log_%s.txt" % time.asctime().split(':')[0]),
                            logging.StreamHandler()
                        ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/dogs')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--print_freq', type=int, default=20)

    parser.add_argument("-istrain", "--istrain", type=bool, help="train or not", default=False)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument("-finetunedPth", "--finetunedPth", type=str, help="finetunedPth",
                        default='fine_tuned_vit_b_16_stanford_dogs.pth')
    parser.add_argument("-quantizedPth", "--quantizedPth", type=str, help="quantizedPth",
                        default='quantized_vit_b_16_stanford_dogs.pth')

    args = parser.parse_args()


    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(size=(224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224 , 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_set = StanfordDogs(root=args.data, split='train', download=False, transforms=train_transform)
    val_set = StanfordDogs(root=args.data, split='test', download=False, transforms=val_transform)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batchsize, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batchsize, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    num_ftrs = model.heads[0].in_features
    model.heads[0] = nn.Linear(num_ftrs, 120)  # Adjust for Stanford Dogs
    model = model.to(device)

    is_train = args.istrain

    # state_dict = ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict()

    if is_train:
        params_1x = []
        params_10x = []
        for name, param in model.named_parameters():
            if 'fc' in name:
                params_10x.append(param)
            else:
                params_1x.append(param)

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW([{'params': params_1x, 'lr': args.lr},
                                      {'params': params_10x, 'lr': args.lr * 10}], lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs/2), gamma=0.01)

        print_freq = args.print_freq
        # Training Loop
        for epoch in range(args.epochs):
            train_loss, train_accuracy = train_one_epoch(model, criterion, optimizer,scheduler, train_loader, device, print_freq)
            val_loss, val_accuracy = validate(model, criterion, val_loader, device, print_freq)
            logging.info(f"Epoch [{epoch + 1}/{args.epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {val_accuracy:.2f}%")

        # Save the Model
        torch.save(model.state_dict(),  args.finetunedPth)
    else:
        state_dict = torch.load(args.finetunedPth, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(state_dict)
        logging.info('loaded pretrained weights....')
        acc, elapsed_time = evaluate_fn(model, val_loader)#加载完预训练权重后，先跑一次验证集得到量化前的运行速度
        logging.info('Validation Acc: {:.2f} %% ,Evaluation time: {:.2f} ms, Inference Time per img:{:.2f} ms'
              .format(acc, elapsed_time, elapsed_time / len(val_set)))

    from nni.compression.quantization import PtqQuantizer as PostTrainingQuantizer

    acc, elapsed_time = evaluate_fn(model, val_loader)
    logging.info('Validation Acc: {:.2f} %% ,Evaluation time: {:.2f} ms, Inference Time per img:{:.2f} ms'
          .format(acc, elapsed_time, elapsed_time / len(val_set)))
    logging.info('')
    # 设置 PTQ 量化配置
    quant_config = [{
        'quant_types': ['weight', 'bias', 'in_proj_weight', 'out_proj_weight'],#权重类型
        'quant_bits': {'weight': 8, 'bias': 8},
        'quant_dtype': 'int8',
        'op_types': ['conv_proj', 'self_attention', 'linear_1', 'linear_2', 'ln_1', 'ln_2'],#所需量化的层
        'quant_scheme': 'affine',
        'granularity': 'default'
    }]

    evaluator = MyEvaluator(model, val_loader)
    # 创建量化器实例
    quantizer = PostTrainingQuantizer(model, config_list=quant_config, evaluator=evaluator)

    # 执行量化
    quantizer.compress(max_steps=20, max_epochs=None)#max_steps 迭代次数

    logging.info('PTQ finished , validating the model......')

    acc, elapsed_time = evaluate_fn(model, val_loader)
    logging.info('Validation Acc: {:.2f} %% ,Evaluation time: {:.2f} ms, Inference Time per img:{:.2f} ms'
          .format(acc, elapsed_time, elapsed_time / len(val_set)))

    quantized_state_dict = model.state_dict()
    # 保存量化后的模型
    torch.save(quantized_state_dict, args.quantizedPth)