import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import argparse


class AlexNet(nn.Module):
    def __init__(self, output_dim, dropout=0.0):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded_text = self.embedding(text)
        rnn_output, _ = self.rnn(embedded_text)
        output = self.fc(rnn_output[:, -1, :])  # 使用最后一个隐藏层
        return output


# 定义多模态模型
class MultiModalModel(nn.Module):
    def __init__(self, vocab_size, text_embed_dim, text_hidden_dim, output_dim, num_classes):
        super(MultiModalModel, self).__init__()
        # 文本处理部分
        self.lstm = LSTM(vocab_size, text_embed_dim, text_hidden_dim, output_dim)
        # 图像处理部分
        self.resnet = AlexNet(output_dim)
        # 全连接层
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2 * output_dim, num_classes)

        # 初始化全连接层参数
        init.xavier_uniform_(self.fc.weight)
        init.constant_(self.fc.bias, 0)

    def forward(self, image, text):
        # 图像处理
        image_features = self.resnet(image)
        # 文本处理
        text_features = self.lstm(text)
        # 合并图像和文本特征
        combined_features = torch.cat((image_features, text_features), dim=1)
        combined = self.relu(combined_features)
        # 分类
        output = self.fc(combined)
        return output


# 文本数据的预处理步骤
class TextProcessor:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_vocab(self, texts):
        for text in texts:
            tokens = text.split()
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = self.vocab_size
                    self.idx2word[self.vocab_size] = token
                    self.vocab_size += 1

    def text_to_tensor(self, text, max_length):
        tokens = text.split()
        token_ids = [self.word2idx[token] for token in tokens if token in self.word2idx]
        token_ids += [0] * (max_length - len(token_ids))  # Padding
        return torch.tensor(token_ids, dtype=torch.long)


# 图像数据的预处理步骤
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 自定义数据集类
class MultiModalDataset(Dataset):
    def __init__(self, dataframe, text_processor, transform=None):
        self.dataframe = dataframe
        self.text_processor = text_processor
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = self.load_image(row['image'])
        text = self.text_processor.text_to_tensor(row['text'], max_length=100)  # 使用文本处理器将文本转换为张量
        label = self.get_label(row['tag'])
        return image, text, label

    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def get_label(self, tag):
        if tag == 'positive':
            return 1
        elif tag == 'negative':
            return 0
        elif tag == 'neutral':
            return 2
        else:
            return 3


def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=10, device="cpu"):
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, texts, labels in train_loader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, texts, labels in valid_loader:
                images = images.to(device)
                texts = texts.to(device)
                labels = labels.to(device)

                outputs = model(images, texts)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)

        valid_loss /= len(valid_loader.dataset)
        accuracy = correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Valid Loss: {valid_loss:.4f}, '
              f'Valid Accuracy: {accuracy:.4f}')


def predict(model, test_loader, device="cpu"):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, texts, _ in test_loader:
            images = images.to(device)
            texts = texts.to(device)

            outputs = model(images, texts)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test different models')
    parser.add_argument('--model', type=str, choices=['multimodal'], default='multimodal',
                        help='Choose which model to test: lstm, alexnet, or multimodal')
    args = parser.parse_args()

    # 加载数据集
    df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # 划分训练集和验证集
    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)

    # 初始化文本处理器并构建词表
    text_processor = TextProcessor()
    text_processor.build_vocab(df['text'])

    # 创建训练集和验证集的数据集和数据加载器
    train_dataset = MultiModalDataset(train_df, text_processor, transform=image_transform)
    valid_dataset = MultiModalDataset(valid_df, text_processor, transform=image_transform)
    test_dataset = MultiModalDataset(test_df, text_processor, transform=image_transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # print(len(train_loader.dataset))  # 检查训练集样本数量
    # print(len(valid_loader.dataset))  # 检查验证集样本数量

    # 获取词表大小
    vocab_size = text_processor.vocab_size

    # 定义超参数
    embedding_dim = 256
    hidden_dim = 128
    output_dim = 128
    num_classes = 3
    num_epochs = 10
    learning_rate = 0.001

    if args.model == 'multimodal':
        model = MultiModalModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 检查是否有可用的GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 训练模型
    train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs, device=device)

    predictions = predict(model, test_loader, device=device)

    print("predict ok!")

    # # 创建一个字典，将整数值映射到类别标签
    # label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    #
    # # 将整数值的预测结果映射到类别标签
    # predicted_labels = [label_map[p] for p in predictions]
    #
    # test_df['tag'] = predicted_labels
    #
    # test_df[['id', 'tag']].to_csv('prediction.txt', sep=',', header=False, index=False)
