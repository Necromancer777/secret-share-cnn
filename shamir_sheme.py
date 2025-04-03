import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

def polynomial(coeffs, x, prime=2**31-1):
    """Вычисление значения полинома в точке x."""
    return sum((c * pow(x, i, prime)) % prime for i, c in enumerate(coeffs)) % prime


def generate_shares(secret, n, k, prime=2**31-1):
    """Генерация n долей, требующих k для восстановления."""
    coeffs = [secret] + [random.randint(1, prime-1) for _ in range(k-1)]
    shares = [(i, polynomial(coeffs, i, prime)) for i in range(1, n+1)]
    return shares


def lagrange_interpolation(x, points, prime=2147483647):
    """Интерполяция Лагранжа для восстановления секрета."""
    total = 0
    for i, (xi, yi) in enumerate(points):
        num, den = 1, 1
        for j, (xj, _) in enumerate(points):
            if i != j:
                num = (num * (x - xj)) % prime
                den = (den * (xi - xj)) % prime
        den = pow(den, prime - 2, prime)  # Обратный элемент по модулю prime
        total = (total + (yi * num * den)) % prime
    return (total + prime) % prime  # Избегаем отрицательных значений


class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def test_model(model, dataloader, device):
    """Оценка точности модели."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def encrypt_weights(model, n, k, prime=2**31-1):
    """Разделение весов модели с учетом отрицательных значений."""
    secret_shares = {}
    for name, param in model.state_dict().items():
        flattened = (param.cpu().numpy().flatten() * 1e6).astype(np.int64)

        flattened = np.mod(flattened, prime)

        shares = [generate_shares(int(w), n, k, prime) for w in flattened]
        secret_shares[name] = shares
    return secret_shares


def decrypt_weights(model, shares, k, prime=2**31-1):
    """Восстановление весов модели с корректным учетом отрицательных чисел."""
    recovered_state = {}
    half_prime = prime // 2  # Граница положительных и отрицательных чисел

    for name, param in model.state_dict().items():
        shape = param.shape
        recovered = np.array([lagrange_interpolation(
            0,
            share[:k],
            prime) for share in shares[name]], dtype=np.int64)

        recovered = np.where(recovered > half_prime, recovered - prime,
                             recovered)
        recovered = recovered.astype(np.float32) / 1e6

        recovered_state[name] = torch.tensor(recovered).view(shape)

    model.load_state_dict(recovered_state)
    return model



# Подготовка данных
def get_dataloader(batch_size=128):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    return test_loader


# Основной процесс
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10Model().to(device)
    model.load_state_dict(torch.load('cifar10_best.pth', map_location=device))
    test_loader = get_dataloader()
    
    acc_before = test_model(model, test_loader, device)
    print(f'Точность до шифрования: {acc_before:.4f}')
    
    start_time = time.time()
    shares = encrypt_weights(model, n=5, k=3)
    encrypt_time = time.time() - start_time
    print(f'Время шифрования: {encrypt_time:.2f} секунд')
    
    model = CIFAR10Model().to(device)

    start_time = time.time()
    model = decrypt_weights(model, shares, k=5)
    decrypt_time = time.time() - start_time
    print(f'Время дешифрования: {decrypt_time:.2f} секунд')

    acc_after = test_model(model, test_loader, device)
    print(f'Точность после расшифровки: {acc_after:.4f}')
