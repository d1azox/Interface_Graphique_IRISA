from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QPushButton,QTableWidgetItem
from PyQt5.QtCore import Qt , QIODevice, QFile
from PyQt5.QtXml import  QDomDocument
import os


class TableWidget_Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Secondary Window")
        self.resize(400, 250)

        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        # Table
        self.table = QTableWidget()

        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Keep", "Property", "Value"])
        main_layout.addWidget(self.table)

        # Add Row Button
        add_row_button = QPushButton("Add Row", self)
        add_row_button.clicked.connect(self.addRow)
        main_layout.addWidget(add_row_button)

        # Delete Row Button
        delete_row_button = QPushButton("Delete Row", self)
        delete_row_button.clicked.connect(self.deleteRow)
        main_layout.addWidget(delete_row_button)

        # Save Button
        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.saveData)
        main_layout.addWidget(save_button)

        # Load Data
        self.file_path = os.path.join(os.getcwd(), "Data_table", "data.xml")
        if os.path.exists(self.file_path):
            self.loadData()

    def addRow(self):
        # Obtenir le nombre de lignes actuel dans le tableau
        row_count = self.table.rowCount()

        # Insérer une nouvelle ligne dans le tableau
        self.table.insertRow(row_count)

        # Créer l'élément de case à cocher par défaut cochée
        checkbox_item = QTableWidgetItem()
        checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        checkbox_item.setCheckState(Qt.Checked)
        self.table.setItem(row_count, 0, checkbox_item)

    def deleteRow(self):
        selected_rows = []

        # Parcours les éléments sélectionnés dans le tableau
        for item in self.table.selectedItems():
            if item.row() not in selected_rows:
                selected_rows.append(item.row())

        # Supprime les lignes sélectionnées
        for row in reversed(selected_rows):
            self.table.removeRow(row)

    # LoadData permet de charger les données du fichier xml vers la table Qt (QTableWidget)
    def loadData(self):
        if self.file_path:
            doc = QDomDocument()
            file = QFile(self.file_path)
            if file.open(QIODevice.ReadOnly):  # Ouvre le fichier xml en lecture seule
                if doc.setContent(file):
                    root_element = doc.documentElement()
                    if root_element.tagName() == "Data":
                        row_elements = root_element.elementsByTagName("Row")
                        # Parcours les lignes du fichier xml
                        for i in range(row_elements.count()):
                            row_element = row_elements.at(i)
                            keep_element = row_element.firstChildElement(
                                "Keep")
                            property_element = row_element.firstChildElement(
                                "Property")
                            value_element = row_element.firstChildElement(
                                "Value")
                            if property_element:
                                # Obtenir le nombre de lignes actuel dans le tableau
                                row_count = self.table.rowCount()

                                # Insérer une nouvelle ligne dans le tableau Qt
                                self.table.insertRow(row_count)

                                # Créer l'élément de case à cocher "Keep"
                                checkbox_item = QTableWidgetItem()
                                checkbox_item.setFlags(
                                    Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                                keep_state = keep_element.text()
                                # Met à jour l'état de la case à cocher
                                check_state = Qt.Checked if keep_state == "Checked" else Qt.Unchecked
                                checkbox_item.setCheckState(check_state)
                                self.table.setItem(row_count, 0, checkbox_item)

                                # Créer l'élément pour la colonne "Property"
                                property_item = QTableWidgetItem(
                                    property_element.text())
                                self.table.setItem(row_count, 1, property_item)

                                # Créer l'élément pour la colonne "Value"
                                value_item = QTableWidgetItem(
                                    value_element.text())
                                self.table.setItem(row_count, 2, value_item)

    def saveData(self):

        if self.file_path:
            doc = QDomDocument()

            # Créer l'élément racine
            root_element = doc.createElement("Data")
            doc.appendChild(root_element)

            # Save les données du tableau
            for row in range(self.table.rowCount()):
                keep_item = self.table.item(row, 0)
                property_item = self.table.item(row, 1)
                value_item = self.table.item(row, 2)

                if property_item:
                    # Créer l'élément pour une ligne de données
                    data_element = doc.createElement("Row")

                    # Save l'état de la case à cocher "Keep"
                    property_element = doc.createElement("Keep")
                    check_state = keep_item.checkState()
                    state_text = "Checked" if check_state == Qt.Checked else "Unchecked"
                    property_element.appendChild(
                        doc.createTextNode(state_text))
                    data_element.appendChild(property_element)

                    # Save le contenu de la colonne "Property"
                    property_element = doc.createElement("Property")
                    property_element.appendChild(
                        doc.createTextNode(property_item.text()))
                    data_element.appendChild(property_element)

                    # Save le contenu de la colonne "Value"
                    value_element = doc.createElement("Value")
                    if value_item is None or value_item.text() == "":
                        value_text = "None"
                    else:
                        value_text = value_item.text()
                    value_element.appendChild(doc.createTextNode(value_text))
                    data_element.appendChild(value_element)

                    root_element.appendChild(data_element)

            # Save le fichier XML
            with open(self.file_path, "w") as file:
                file.write(doc.toString())

            print("Data saved successfully.")
            self.reject()