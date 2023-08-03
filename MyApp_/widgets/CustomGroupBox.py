from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,QPushButton
from PyQt5.QtGui import  QIcon

from widgets.CustomGraphicsView import CustomGraphicsView 

class CustomGroupBox(QGroupBox):
    def __init__(self, app_instance, parent=None):
        super().__init__(parent)

        self.app_instance = app_instance
        self.parent_ = parent
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Créer un sous-layout horizontal pour le label et le bouton de fermeture
        header_layout = QHBoxLayout()
        layout.addLayout(header_layout)

        self.label = QLabel(self)
        self.label.setFixedHeight(20)

        header_layout.addWidget(self.label)

        # Ajoutez le bouton de fermeture
        self.close_button = QPushButton()
        self.close_button.setFixedSize(15, 15)

        # Remplacez le chemin vers l'icône de croix
        self.close_button.setIcon(QIcon("Icon/fermer.png"))
        self.close_button.clicked.connect(self.on_close_button_clicked) 
        header_layout.addWidget(self.close_button)

        # Ajoutez la scène 
        self.graphics_view = CustomGraphicsView(self.app_instance, self.parent_)
        

        layout.addWidget(self.graphics_view)
        self.graphics_view.scene_.clear()
        self.label.clear()
        self.hide() #Cache par défault

    def on_close_button_clicked(self): #Méthode pour le bouton
        self.hide()
        self.graphics_view.scene_.clear()
        self.label.clear()

        for scene in self.parent_.all_scenes:
            if scene is self.graphics_view :
                self.parent_.all_scenes.remove(scene)