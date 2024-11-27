
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import dataset_generation

# Подготовка данных для обучения и тестирования
dataset = dataset_generation.generate()
train_n = int(len(dataset) * 0.8)
train, test = random_split(dataset, (train_n, len(dataset) - train_n))
train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Создание LSTM модели нейронной сети
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Инициализация модели, критерия расчет ошибки и оптимизатора
model = SimpleLSTM(30, 80, 30).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    train_loss, test_loss = 0, 0
    for seq, target in train_dataloader:
        seq, target = seq.to(device), target.to(device)
        outputs = model(seq)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for seq, target in test_dataloader:
            seq, target = seq.to(device), target.to(device)
            outputs = model(seq)
            loss = criterion(outputs, target)
            test_loss += loss.item()

    train_loss /= len(train_dataloader)
    test_loss /= len(test_dataloader)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Сохранение модели
torch.save(model.state_dict(), 'simple_lstm_model.pth')
