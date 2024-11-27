
import numpy as np
import pygame
import torch
from torch import nn


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


model = SimpleLSTM(30, 80, 30)
model.load_state_dict(torch.load('simple_lstm_model.pth', weights_only=True, map_location=torch.device('cpu')))
model.eval()


def update_predict():
    global pred
    input_sequence = np.zeros((1, 30, 30), dtype=np.float32)
    for _i, _num in enumerate(data):
        input_sequence[0, _i, _num] = 1
    input_sequence = torch.tensor(input_sequence)
    with torch.no_grad():
        output = model(input_sequence)
        pred_raw = [int(float(e) * 30) if float(e) > 1 else 0 for e in output[0]]
        pred = [e / sum(pred_raw) * 100 for e in pred_raw]


pygame.init()
screen = pygame.display.set_mode((800, 600))
num_font = pygame.font.Font(None, 18)
pygame.display.set_caption("LSTM предсказатель")

grad_st, grad_en = (120, 50, 50), (120, 50, 150)

data, pred = [], [0] * 30
running = True
while running:
    for i in range(600):
        color = list((int(grad_st[i1] + (grad_en[i1] - grad_st[i1]) / 600 * i) for i1 in range(3)))
        pygame.draw.line(screen, color, (0, i), (800, i))

    k = min(3, len(data) // 10 + 1)
    dl = [[64, 30, 20], [0, 20, 25], [0, 5, 8], [0, 10, 13], [0, 10, 13], [32, 20, 15], [13, 5, 3], [6, 2, 2]]
    for i in range(k * 10):
        pygame.draw.rect(screen, (30, 30, 30), (100 + i * 60 / k, 150, 50 / k, 10 // k), border_radius=3)

    data_font = pygame.font.Font(None, dl[0][k - 1])
    pred_font = pygame.font.Font(None, dl[5][k - 1])
    for i in range(len(data)):
        num = data_font.render(str(data[i]), True, (30, 30, 30))
        screen.blit(num, (112 + i * 60 / k - dl[2][k - 1] - (dl[6][k - 1] if data[i] > 9 else 0), 110 + dl[1][k - 1]))
    if len(data) < 30:
        num = pred_font.render(str(pred.index(max(pred))), True, (30, 30, 30))
        screen.blit(num, (118 + len(data) * 60 / k - dl[3][k - 1] - (dl[7][k - 1] if pred.index(max(pred)) > 9 else 0),
                          125 + dl[4][k - 1]))

    for i in range(30):
        pygame.draw.rect(screen, (30, 30, 30), (100 + i * 20, 400 - pred[i], 12, pred[i]), border_radius=3)
        num = num_font.render(str(i), True, (30, 30, 30))
        screen.blit(num, (102 + i * 20, 410))
        if pred[i] > 0:
            num = num_font.render(str(round(pred[i])) + '%', True, (30, 30, 30))
            screen.blit(num, (95 + i * 20, 380 - pred[i]))

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if 47 < event.key < 58 and len(data) < 30:
                data.append(event.key - 48)

            if event.key == pygame.K_BACKSPACE and data:
                data.pop()

            if event.key == pygame.K_DELETE:
                data.clear()

            if event.key == pygame.K_LEFT and len(data) > 1 and data[-2] < 10 and data[-2] * 10 + data[-1] < 30:
                data[-2] = data[-2] * 10 + data[-1]
                data.pop()

            update_predict()
