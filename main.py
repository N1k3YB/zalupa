import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget, QLineEdit, QPushButton, QTabWidget, QFrame
from PyQt5.QtGui import QPixmap
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Загрузка предварительно обученной модели Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cpu")  # Перенос модели на CPU

# Определение класса для загрузки данных
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image

# Определение преобразований для данных
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Определение интерфейса
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Добавление картинок для обучения и генерация")
        self.setGeometry(100, 100, 800, 600)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        # Создание вкладок
        self.tabWidget = QTabWidget()
        self.layout.addWidget(self.tabWidget)

        # Вкладка для загрузки картинок
        self.loadImageFrame = QFrame()
        self.loadImageLayout = QVBoxLayout()
        self.loadImageFrame.setLayout(self.loadImageLayout)

        self.imageLabel = QLabel()
        self.loadImageLayout.addWidget(self.imageLabel)

        self.loadButton = QPushButton("Загрузить изображения")
        self.loadButton.clicked.connect(self.load_image)
        self.loadImageLayout.addWidget(self.loadButton)

        self.trainButton = QPushButton("Обучить модель")
        self.trainButton.clicked.connect(self.train_model)
        self.loadImageLayout.addWidget(self.trainButton)

        self.tabWidget.addTab(self.loadImageFrame, "Загрузка картинок")

        # Вкладка для генерации картинок
        self.generateImageFrame = QFrame()
        self.generateImageLayout = QVBoxLayout()
        self.generateImageFrame.setLayout(self.generateImageLayout)

        self.generateImageLabel = QLabel()
        self.generateImageLayout.addWidget(self.generateImageLabel)

        self.promptEdit = QLineEdit()
        self.generateImageLayout.addWidget(self.promptEdit)

        self.generateButton = QPushButton("Генерировать картинку")
        self.generateButton.clicked.connect(self.generate_image)
        self.generateImageLayout.addWidget(self.generateButton)

        self.tabWidget.addTab(self.generateImageFrame, "Генерация картинок")

        self.image_paths = []

    def load_image(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Открыть изображения", "", "Изображения (*.png *.jpg *.jpeg)")
        if file_paths:
            self.image_paths.extend(file_paths)
            pixmap = QPixmap(file_paths[0])
            self.imageLabel.setPixmap(pixmap)

    def train_model(self):
        # Обучение нейросети на загруженных изображениях
        dataset = ImageDataset(self.image_paths, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        num_epochs = 20
        for epoch in range(num_epochs):
            for images in dataloader:
                images = images.to("cpu")
                loss = pipe(images, guidance_scale=7.5).loss
                loss.backward()
                pipe.optimizer.step()
                pipe.scheduler.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

        # Сохранение обученной модели
        pipe.save_pretrained("path/to/save/model")

    def generate_image(self):
        prompt = self.promptEdit.text()
        image = pipe(prompt)["sample"][0]
        pixmap = QPixmap.fromImage(image)
        self.generateImageLabel.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())