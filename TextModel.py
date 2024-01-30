import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim


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


class TextDataset(Dataset):
    def __init__(self, dataframe, text_processor):
        self.dataframe = dataframe
        self.text_processor = text_processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = self.text_processor.text_to_tensor(row['text'], max_length=100)  # 使用文本处理器将文本转换为张量
        label = self.get_label(row['tag'])
        return text, label

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
        for texts, labels in train_loader:
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in valid_loader:
                texts = texts.to(device)
                labels = labels.to(device)

                outputs = model(texts)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader.dataset)
        accuracy = correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Valid Loss: {valid_loss:.4f}, '
              f'Valid Accuracy: {accuracy:.4f}')


df = pd.read_csv('train.csv')

# 划分训练集和验证集
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)

# 初始化文本处理器并构建词表
text_processor = TextProcessor()
text_processor.build_vocab(df['text'])

# 创建训练集和验证集的数据集和数据加载器
train_dataset = TextDataset(train_df, text_processor)
valid_dataset = TextDataset(valid_df, text_processor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 获取词表大小
vocab_size = text_processor.vocab_size

# 定义超参数
embedding_dim = 256
hidden_dim = 128
output_dim = 128
num_classes = 3
num_epochs = 10
learning_rate = 0.001

model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 检查是否有可用的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 训练模型
train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs, device=device)
