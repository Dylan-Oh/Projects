import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
from ClassificationModelGenerator import CNN, device
import matplotlib.pyplot as plt
import random

# 모델의 최종 성능을 기록할 변수
train_losses, val_losses, accuracies = [], [], []

def objective(trial):
    output_folder = './augmented_images'  # 사용된 이미지 폴더 위치

    # 하이퍼파라미터 범위 설정
    batch_size = trial.suggest_int('batch_size', 32, 128, step=32)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 10, 30, step=5)
    
    # 데이터 전처리 및 로딩
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지 크기 조정
        transforms.ToTensor(),          # 텐서로 변환
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
    ])
    dataset = datasets.ImageFolder(root=output_folder, transform=transform)
    
    class_samples = {}
    
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(i)
    
    # 클래스 간의 샘플 수가 동일하도록 분할
    train_samples = []
    val_samples = []
    
    for label, samples in class_samples.items():
        random.shuffle(samples)
        
        n_samples = len(samples)
        n_train = int(0.65 * n_samples)
        n_val = int(0.2 * n_samples)
        
        # 클래스 별로 분할된 샘플 추가
        train_samples.extend(samples[:n_train])
        val_samples.extend(samples[n_train:n_train+n_val])
    
    # 분할된 샘플 인덱스를 사용하여 데이터셋 분할
    train_dataset = Subset(dataset, train_samples)
    val_dataset = Subset(dataset, val_samples)
    
    # 분할된 데이터셋을 데이터 로더로 변환합니다.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.85)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, total, correct = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        accuracies.append(correct / total)
        scheduler.step()

    return val_losses[-1]

def run_hyperparameter_tuning():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)  # Reduced number of trials for quicker testing

    print('Best hyperparameters:', study.best_params)
    print('Best validation loss:', study.best_value)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(val_losses, label='Validation')
    plt.title('Loss Over Trials')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.title('Accuracy Over Trials')
    plt.legend()

    plt.show()

    
run_hyperparameter_tuning()