from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGraphicsScene, QGraphicsView,QPushButton, QFileDialog
from PyQt5.QtCore import Qt 
from PyQt5.QtGui import  QImage, QPainter

class CropResultDialog(QDialog): #Fênetre du résultat du crop
    def __init__(self, cropped_image):
        super().__init__()
        self.setWindowTitle("Résultat du recadrage")
        self.setFixedSize(1080, 720)

        self.layout = QVBoxLayout()

        # Créer la scène et la vue
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setFixedSize(1080, 720)  # Fixer la taille de la vue

        # Récupérer les dimensions de l'image crop
        image_width = cropped_image.width()
        image_height = cropped_image.height()

        # Calculer les facteurs d'échelle pour remplir la vue tout en conservant l'aspect ratio
        view_width = self.view.width()
        view_height = self.view.height()
        scale_factor = min(view_width / image_width,
                           view_height / image_height)
        scaled_width = int(image_width * scale_factor)
        scaled_height = int(image_height * scale_factor)

        # Ajouter l'image à la scène en la redimensionnant
        pixmap = cropped_image.scaled(scaled_width, scaled_height)
        self.scene.addPixmap(pixmap)

        # Centrer l'image dans la vue
        self.view.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.view)

        self.save_button = QPushButton("Enregistrer")
        self.save_button.clicked.connect(self.saveImage)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)

    def saveImage(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Enregistrer l'image", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_path:
            self.scene.clearSelection()
            self.scene.setSceneRect(self.scene.itemsBoundingRect())
            image = QImage(self.scene.sceneRect(
            ).size().toSize(), QImage.Format_ARGB32)
            image.fill(Qt.transparent)
            painter = QPainter(image)
            self.scene.render(painter)
            painter.end()
            image.save(file_path) #Save l'image
        self.reject()
