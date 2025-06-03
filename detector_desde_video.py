import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from zonas_estacionamiento import zonas  # Coordenadas extraídas de la máscara visual

# Modelo CNN personalizada igual al usado durante el entrenamiento
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Convolución 1
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reducción espacial
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Convolución 2
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 17 * 7, 64),  # Capa totalmente conectada
            nn.ReLU(),
            nn.Linear(64, 2)  # Salida con 2 clases: vacío u ocupado
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Cargar el modelo previamente entrenado
model = SmallCNN()
model.load_state_dict(torch.load("modelo_cnn_ligero.pth", map_location=torch.device("cpu")))
model.eval()

# Transformaciones para preprocesar cada zona
transform = transforms.Compose([
    transforms.ToPILImage(),          # Convertir a imagen PIL
    transforms.Resize((68, 29)),      # Redimensionar al tamaño del modelo
    transforms.ToTensor()             # Convertir a tensor
])

# Cargar el video para análisis (asegúrate de que la ruta sea correcta)
cap = cv2.VideoCapture(r"C:\Users\cooki\Documents\Proyecto IA\parking\parking_1920_1080_loop.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ocupados = 0
    vacios = 0

    # Evaluar cada zona definida
    for (x, y, w, h) in zonas:
        zona = frame[y:y+h, x:x+w]
        if zona.shape[0] != h or zona.shape[1] != w:
            continue  # Ignorar si la región está incompleta (en bordes)

        entrada = transform(zona).unsqueeze(0)  # Añadir dimensión batch
        with torch.no_grad():
            salida = model(entrada)
            pred = torch.argmax(salida, dim=1).item()

        # 🟩 Vacío / 🟥 Ocupado
        estado = "Vacio" if pred == 0 else "Ocupado"
        color = (0, 255, 0) if pred == 0 else (0, 0, 255)

        if pred == 0:
            vacios += 1
        else:
            ocupados += 1

        # Dibujar resultado en el video
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, estado, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Mostrar conteo total
    cv2.putText(frame, f'Ocupados: {ocupados}  Vacios: {vacios}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar ventana redimensionable
    cv2.namedWindow("Detección", cv2.WINDOW_NORMAL)
    cv2.imshow("Detección", frame)

    if cv2.waitKey(25) & 0xFF == 27:
        break  # ESC para salir

cap.release()
cv2.destroyAllWindows()
