import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
from tqdm import tqdm
import seaborn as sns

output_folder = './augmented_images'
batch_size = 64
learning_rate = 0.0077
num_epochs = 25

# Model definition (using VGG16)
class ResNetTransfer(nn.Module):
    def __init__(self, num_classes=9):
        super(ResNetTransfer, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# 데이터 전처리 및 로딩
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),         # 텐서로 변환
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
])

# 데이터셋 생성
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
test_samples = []
for label, samples in class_samples.items():
    random.shuffle(samples)
    
    n_samples = len(samples)
    n_train = int(0.65 * n_samples)
    n_val = int(0.2 * n_samples)
    n_test = n_samples - n_train - n_val
    
    # 클래스 별로 분할된 샘플 추가
    train_samples.extend(samples[:n_train])
    val_samples.extend(samples[n_train:n_train+n_val])
    test_samples.extend(samples[n_train+n_val:])

# 분할된 샘플 인덱스를 사용하여 데이터셋 분할
train_dataset = Subset(dataset, train_samples)
val_dataset = Subset(dataset, val_samples)
test_dataset = Subset(dataset, test_samples)

# 분할된 데이터셋을 데이터 로더로 변환합니다.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# GPU 사용 가능 여부 확인
device = torch.device("cuda")

# 모델 인스턴스 생성
model = ResNetTransfer(num_classes=9).to(device)

print(torch.cuda.is_available())

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)  # ReduceLROnPlateau 설정
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.85)

# 손실 함수 및 정확도 기록
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

patience = 5  # 허용횟수
early_stopping_counter = 0  # 얼리스탑 카운터
best_val_loss = float('inf')  # 최고 검증 손실값

# 학습
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
        
    train_accuracy = correct_train / total_train

    # 테스트
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # 입력 데이터를 GPU로 이동
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted_val = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted_val.cpu().numpy())
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()
    # 손실 함수 및 정확도 평균 계산
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    train_accuracies.append(train_accuracy)
    val_accuracies.append(correct_val / total_val)
    # 정확도 계산
    accuracy = accuracy_score(y_true, y_pred)
    # 정밀도 계산
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    # 재현율 계산
    recall = recall_score(y_true, y_pred, average='weighted')
    # f1 스코어 계산
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {running_loss / len(train_loader)}")
    print(f"Val Loss: {val_loss / len(val_loader)}")
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Val Accuracy: {correct_val / total_val * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    
    # epoch이 절반 이상 진행됐을 때
    if epoch > num_epochs / 2:
        # early stop 조건 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # 얼리스탑 카운터 초기화
        else:
            early_stopping_counter += 1
        
        # early stop 허용 횟수를 초과하면 학습 중단
        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            #break

    # 학습률 감소
    #scheduler.step(val_loss)
    scheduler.step()
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')

# 모델의 최종 성능 평가
model.eval()
test_loss = 0.0
correct = 0
total = 0
y_true_test = []
y_pred_test = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(predicted.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 평가 지표 계산
test_accuracy = accuracy_score(y_true_test, y_pred_test)
test_precision = precision_score(y_true_test, y_pred_test, average='weighted', zero_division=1)
test_recall = recall_score(y_true_test, y_pred_test, average='weighted')
test_f1 = f1_score(y_true_test, y_pred_test, average='weighted')
print("모델의 최종 성능")

# 평가 결과 출력
print("Test Loss: {:.4f}".format(test_loss / len(test_loader)))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
print("Test Precision: {:.2f}%".format(test_precision * 100))
print("Test Recall: {:.2f}%".format(test_recall * 100))
print("Test F1 Score: {:.2f}%".format(test_f1 * 100))

# 혼동 행렬 계산
cm = confusion_matrix(y_true_test, y_pred_test, normalize='true')

# 그래프 그리기
plt.figure(figsize=(10, 5))

# 손실 함수 그래프
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# 혼동 행렬 시각화
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()