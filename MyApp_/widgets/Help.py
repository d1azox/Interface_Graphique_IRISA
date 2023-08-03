from PyQt5.QtWidgets import QDialog, QVBoxLayout,QTableWidget, QTableWidgetItem

class HelpDialog_detection(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Help Detection")
        self.resize(700, 350)

        # Création du layout principal de la fenêtre
        layout = QVBoxLayout()

        # Création du tableau pour afficher les attributs et les descriptions
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Attribute", "Description"])

        # Ajout des attributs et des descriptions dans le tableau
        attributes = ["Size", "confThreshold","nmsTreshold"]
        descriptions = ["It specifies the dimensions of the sub-regions into which the image is divided for processing. The size of these subdivisions can affect the accuracy and speed of the Yolo algorithm.",
                        "It determines the minimum confidence score required for an object detection to be considered valid. Objects with confidence scores below this threshold are filtered out and not included in the final results.",
                        "NMS is a post-processing step that helps eliminate duplicate or overlapping detections of the same object. The threshold determines the amount of overlap allowed between bounding boxes before suppression is applied."]

        self.table.setRowCount(len(attributes))
        for row, (attribute, description) in enumerate(zip(attributes, descriptions)):
            self.table.setItem(row, 0, QTableWidgetItem(attribute))
            self.table.setItem(row, 1, QTableWidgetItem(description))

        # Taille de la premiere colonne
        self.table.setColumnWidth(0, 125)
        self.table.setColumnWidth(1, 400)

        # Adapte le taille des cellules en fonction du contenue
        self.table.resizeRowsToContents()

        # Ajout du tableau au layout
        layout.addWidget(self.table)

        # Création du widget central de la fenêtre et attribution du layout
        self.setLayout(layout)

class HelpDialog_segmentation(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Help Semantic Segmentation")
        self.resize(700, 350)

        # Création du layout principal de la fenêtre
        layout = QVBoxLayout()

        # Création du tableau pour afficher les attributs et les descriptions
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Attribut", "Description"])

        # Ajout des attributs et des descriptions dans le tableau
        attributes = ["Nor", "Model", "Threshold"]
        descriptions = ["This option allows normalizing the image, which is highly recommended to ensure consistent results during processing.",
                        "This attribute allows choosing the type of model to use for the Rednet network, such as rednet4B, rednet5B (See the .py file of Rednet)",
                        "The threshold is a parameter used to filter predictions based on their confidence level. This threshold determines how confident a prediction must be considered valid."]

        self.table.setRowCount(len(attributes))
        for row, (attribute, description) in enumerate(zip(attributes, descriptions)):
            self.table.setItem(row, 0, QTableWidgetItem(attribute))
            self.table.setItem(row, 1, QTableWidgetItem(description))

        # Taille de la premiere colonne
        self.table.setColumnWidth(1, 400)

        # Adapte le taille des cellules en fonction du contenue
        self.table.resizeRowsToContents()

        # Ajout du tableau au layout
        layout.addWidget(self.table)

        # Création du widget central de la fenêtre et attribution du layout
        self.setLayout(layout)

class HelpDialog_ins_segmentation(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Help Instance Segmentation")
        self.resize(700, 350)

        # Création du layout principal de la fenêtre
        layout = QVBoxLayout()

        # Création du tableau pour afficher les attributs et les descriptions
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Attribute", "Description"])

        # Ajout des attributs et des descriptions dans le tableau
        attributes = ["confThreshold","Size" ]
        descriptions = ["It determines the minimum confidence score required for an object detection to be considered valid. Objects with confidence scores below this threshold are filtered out and not included in the final results.",
                        "It specifies the dimensions of the sub-regions into which the image is divided for processing. The size of these subdivisions can affect the accuracy and speed of the Yolo algorithm."]
        
        self.table.setRowCount(len(attributes))
        for row, (attribute, description) in enumerate(zip(attributes, descriptions)):
            self.table.setItem(row, 0, QTableWidgetItem(attribute))
            self.table.setItem(row, 1, QTableWidgetItem(description))

        # Taille de la premiere colonne
        self.table.setColumnWidth(0, 125)
        self.table.setColumnWidth(1, 400)

        # Adapte le taille des cellules en fonction du contenue
        self.table.resizeRowsToContents()

        # Ajout du tableau au layout
        layout.addWidget(self.table)

        # Création du widget central de la fenêtre et attribution du layout
        self.setLayout(layout)