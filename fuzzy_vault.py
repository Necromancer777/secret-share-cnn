import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hashlib import sha256
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm


P = 40009  # Простое число для конечного поля
d = 3      # Степень полинома
t_threshold = 4  # Порог для восстановления (t из n)
noise_ratio = 0.2  # Уменьшенная доля ложных точек (было 0.5)
param_ratio = 0.02  # Доля защищаемых параметров


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

# --- Загрузка и обучение модели ---
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10Model().to(device)
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    trainset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(5):  
        for images, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/5'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return model

# --- Загрузка тестовых данных ---
def load_data():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    return DataLoader(testset, batch_size=100, shuffle=False)

# --- Выбор важных параметров ---
def get_important_params(model, ratio):
    important_params = []
    param_indices = []

    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            flat = param.detach().flatten().numpy()
            k = int(len(flat) * ratio)
            indices = np.argpartition(np.abs(flat), -k)[-k:]
            indices = [int(idx) for idx in indices]  

            important_params.append(flat[indices].tolist())  
            param_indices.append((name, indices))

    return important_params, param_indices




# --- Преобразование параметров ---
def float_to_int(x, max_abs, P):
    """ Преобразует float в int с сохранением масштаба """
    return int((x / max_abs) * (P // 2))  # Нормализация на P//2

def int_to_float(x_int, max_abs, P):
    """ Обратно преобразует int в float """
    return (x_int / (P // 2)) * max_abs  # Обратное преобразование


# --- Генерация Fuzzy Vault ---
def create_vault(params, param_indices, d, P, noise_ratio):
    vault = []
    param_to_vault = {}

    for (name, indices), param_values in zip(param_indices, params):
        for idx, p in zip(indices, param_values):  
            coeffs = [np.random.randint(0, P) for _ in range(d+1)]
            x = int(sha256(str(idx).encode()).hexdigest(), 16) % P
            y = sum(coeffs[i] * (x**i) % P for i in range(d+1)) % P  

            vault.append((x, y))
            param_to_vault[idx] = (x, y, p, coeffs[0])  

    num_noise = int(len(vault) * noise_ratio)
    for _ in range(num_noise):
        vault.append((np.random.randint(0, P), np.random.randint(0, P)))

    np.random.shuffle(vault)
    return vault, param_to_vault


# --- Интерполяция Лагранжа ---
def lagrange_interpolation(points, P):
    secret = 0
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    
    print(f"Точки для интерполяции: {points}")

    for j in range(len(points)):
        num, den = 1, 1
        for m in range(len(points)):
            if m != j:
                num = (num * (-x_values[m] + P) % P) % P  
                den = (den * (x_values[j] - x_values[m]) % P) % P

        # Проверяем, существует ли обратный элемент
        if den % P == 0:
            print(f" Ошибка: деление на ноль при j={j}, x={x_values[j]}")
            return None  # Возвращаем None при ошибке

        lj = (num * pow(den, P - 2, P)) % P  
        secret = (secret + y_values[j] * lj) % P

    print(f"Вычисленный секрет: {secret}")
    return secret


# --- Восстановление модели ---
def restore_model(model, param_indices, param_to_vault, P, max_abs):
    device = next(model.parameters()).device
    restored_model = type(model)().to(device)
    restored_model.load_state_dict(model.state_dict())

    with torch.no_grad():
        for name, indices in param_indices:
            param = restored_model.state_dict()[name]
            flat = param.flatten().cpu().numpy()

            for idx in indices:
                int_idx = int(idx)  
                
                if int_idx in param_to_vault:  
                    x, y, orig_value, secret = param_to_vault[int_idx]
                    flat[idx] = orig_value  
                else:
                    print(f" Индекс {idx} отсутствует в param_to_vault (параметр {name})")

            param.copy_(torch.tensor(flat, device=device).view(param.shape))  

    return restored_model


# --- Тестирование модели ---
def test_model(model, testloader):
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Тестирование"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# --- Основной эксперимент ---
def main():
    # Инициализация модели
    model = CIFAR10Model()
    model.load_state_dict(torch.load('cifar10_weights.pth', map_location='cpu'))

    # Загрузка тестовых данных
    testloader = load_data()
    original_accuracy = test_model(model, testloader)
    print(f"Исходная точность модели: {original_accuracy:.4f}")

    # Выборка важных параметров
    important_params, param_indices = get_important_params(model, param_ratio)
    
    
    important_params_dict = {idx: p for idx, p in zip(range(len(important_params)), important_params)}

    flat_important_params = np.concatenate(important_params)
    max_abs = np.max(np.abs(flat_important_params))
    int_params = [[float_to_int(p_val, max_abs, P) for p_val in param_list] for param_list in important_params]



    
    vault, param_to_vault = create_vault(int_params, param_indices, d, P, noise_ratio)


    print("Первые 5 точек хранилища:", vault[:5])

    # Эксперимент с восстановлением
    for t in range(1, 6):
        print(f"\n--- Число участников: {t} ---")
        
       
        available_indices = list(param_to_vault.keys())
        if len(available_indices) < t:
            print(f" Недостаточно данных для {t} участников, пропускаем")
            continue
        
        selected_indices = np.random.choice(available_indices, size=t, replace=False)
        user_points = [param_to_vault[idx][:2] for idx in selected_indices]

        if len(user_points) >= t_threshold:
            secret = lagrange_interpolation(user_points, P)
            print(f"Восстановленный секрет: {secret}")

            
            for name, indices in param_indices:
                print(f"Параметр: {name}, всего элементов: {len(model.state_dict()[name].flatten())}")
                print(f"Выбранные индексы: {indices[:10]}")

            restored_model = restore_model(
                model=model,
                param_indices=param_indices,
                param_to_vault=param_to_vault,
                P=P,
                max_abs=max_abs
            )
            accuracy = test_model(restored_model, testloader)
        else:
            print("Секрет не восстановлен (недостаточно точек)")
            accuracy = test_model(model, testloader)

        print(f"Точность восстановленной модели: {accuracy:.4f}")


if __name__ == "__main__":
    main()