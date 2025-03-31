import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from secretsharing import SecretSharer  

class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.classifier(x)
        return x

def split_secret(weights, num_shares=5, threshold=3):
    """Функция для разделения весов модели на несколько частей по схеме Шамира"""
    # Убедимся, что пороговое значение меньше числа частей
    if threshold >= num_shares:
        raise ValueError("Threshold must be less than the total number of shares.")
    
    shares = {}
    for name, param in weights.items():
        flat_param = param.flatten().tolist()
        param_shares = []
        for w in flat_param:
            # Преобразуем числа в строку в шестнадцатеричной форме для безопасной обработки
            w_hex = format(int(round(w * 10**6)), 'x')  # Преобразуем в шестнадцатеричную строку
            param_shares.append(SecretSharer.split_secret(w_hex, num_shares, threshold))
        shares[name] = param_shares
    return shares

def recover_secret(shares, threshold=3):
    """Восстанавливает веса модели из частей"""
    recovered_weights = {}
    for name, param_shares in shares.items():
        recovered_list = []
        for sh in zip(*param_shares):
            secret_str = SecretSharer.recover_secret(sh[:threshold])
            # Преобразуем восстановленную строку обратно в число
            recovered_list.append(int(secret_str, 16) / 10**6)  # Преобразуем из шестнадцатеричной обратно в число
        recovered_weights[name] = torch.tensor(recovered_list).reshape(-1)
    return recovered_weights

def apply_weights_to_model(model, weights):
    model.load_state_dict(weights)
    return model

def evaluate_model(model, dataloader):
    """Оценивает точность модели на тестовых данных"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Загружаем тестовый датасет CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Загружаем модель и её веса
model = CIFAR10Model()
weights = model.state_dict()

# Разделяем веса
shares = split_secret(weights, num_shares=5, threshold=3)  

# Восстанавливаем веса
restored_weights = recover_secret(shares)
model = apply_weights_to_model(model, restored_weights)

# Оцениваем точность модели
accuracy = evaluate_model(model, testloader)
print(f'Accuracy after recovery: {accuracy:.2f}%')
