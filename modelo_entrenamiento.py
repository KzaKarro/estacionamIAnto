import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# Ruta donde se encuentra el dataset
data_dir = r'C:\Users\cooki\Documents\Proyecto IA\parking\clf-data'

# Transformaciones para preprocesar y aumentar los datos de imagen
transform = transforms.Compose([
    transforms.Resize((68, 29)),  # Redimensionar a tamaño nativo
    transforms.RandomHorizontalFlip(),  # Volteo horizontal aleatorio
    transforms.RandomRotation(5),       # Rotación aleatoria de hasta ±5°
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Ajustes de color
    transforms.ToTensor()  # Convertir imagen a tensor
])

# Cargar dataset y dividirlo 80% entrenamiento, 20% validación
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders para lotes de datos
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Definición de una CNN pequeña personalizada
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Capa convolucional 1
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reducción de tamaño: 34x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Capa convolucional 2
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reducción de tamaño: 17x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 17 * 7, 64),  # Capa densa intermedia
            nn.ReLU(),
            nn.Linear(64, 2)  # Capa de salida (2 clases: vacío u ocupado)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Configuración de entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU si está disponible
model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss()  # Función de pérdida para clasificación
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizador

# Ciclo de entrenamiento (30 épocas)
for epoch in range(30):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()            # Limpiar gradientes anteriores
        outputs = model(inputs)          # Hacer predicción
        loss = criterion(outputs, labels)  # Calcular pérdida
        loss.backward()                  # Calcular gradientes
        optimizer.step()                 # Actualizar pesos

    # Evaluación en validación
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f} | Val Accuracy = {acc:.2f}%")

# Guardar el modelo entrenado
torch.save(model.state_dict(), "modelo_cnn_ligero.pth")
