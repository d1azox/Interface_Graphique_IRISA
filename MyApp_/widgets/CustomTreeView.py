import os 
from PyQt5.QtWidgets import QTreeView, QAbstractItemView,QFileDialog , QApplication, QFileSystemModel
from PyQt5.QtCore import Qt , QFileInfo, QModelIndex
from PyQt5.QtGui import  QPixmap


class CustomFileSystemModel(QFileSystemModel): #Model de l'arborecence adapté pour des path 
    def flags(self, index):
        flags = super().flags(index)
        return flags | Qt.ItemIsEditable

class CustomTreeView(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent_ = parent
        # Désactiver le mode d'édition par défaut
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)


        self.model_ = CustomFileSystemModel() #Création du Model
        self.setModel(self.model_) #Met en place le model dans le QTreeView
        self.doubleClicked.connect(self.itemDoubleClicked)#Affiche élément selectionné

        #Cache les colonnes non nécessaires 
        self.hideColumn(1)
        self.hideColumn(2)
        self.hideColumn(3)

    
    def loadDir(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Sélectionner un dossier", os.getcwd()) #Sélection du path du dossier input

        if folder_path:
            # Met en place la racine du model
            self.model_.setRootPath(folder_path)

            # Filtrer les extensions d'image à afficher
            image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif"]
            self.model_.setNameFilters(image_extensions)
            self.model_.setNameFilterDisables(False)

            # Masquer les colonnes "Size" et "Type"
            self.hideColumn(1)
            self.hideColumn(2)
            self.hideColumn(3)

            # Création du QTreeView
            self.setRootIndex(self.model_.index(folder_path))

    def renameSelectedItem(self): 
        current_index = self.currentIndex() #Elément sélectionné

        self.edit(current_index) #Edit élément

        self.update() #Update l'arborescence

    def copySelectedItem(self):
        current_index = self.currentIndex()
        # Récupérer le chemin complet du fichier
        file_path = self.model_.filePath(current_index)
        if file_path is not None:
            pixmap = QPixmap(file_path)
            clipboard = QApplication.clipboard() #Enregistre dans le presse-papier
            clipboard.setPixmap(pixmap)
            self.file_path_image = file_path #Chemin de l'image copier

    def pasteClipboardData(self):
        # Récupére l'élément dans le presse-papier
        clipboard = QApplication.clipboard()
        image = clipboard.image()

        # Récupére l'index actuel dans l'arborescence
        current_index = self.currentIndex()
        # Récupére le chemin complet du fichier sélectionné
        file_path = self.model_.filePath(current_index)
        file_info = QFileInfo(file_path)

        if not file_info.isDir():
            # Récupére le chemin du dossier parent
            folder_path = os.path.dirname(file_path)
        else:
            folder_path = file_path

        if not image.isNull():  # Vérifier si l'image du presse-papier n'est pas vide

            if not os.path.exists(os.path.join(folder_path, os.path.basename(self.file_path_image))):
                # Si le fichier de destination n'existe pas, enregistrer l'image du presse-papier dans ce dossier
                image.save(os.path.join(
                    folder_path, os.path.basename(self.file_path_image)))
                self.update()  # Mettre à jour la vue de l'arborescence
                return
            else:
                if file_path is not None:
                    # Si le fichier de destination existe déjà, créer une copie de l'image en ajoutant un compteur

                    file_name = "copy_" + os.path.basename(self.file_path_image)

                    file_path = os.path.join(folder_path, file_name)

                    # Vérifier si le fichier existe déjà
                    if os.path.exists(file_path):
                        copy_count = 2
                        while os.path.exists(file_path):
                            # Ajouter le compteur de copie au nom du fichier
                            copy_file_name = f"copy {copy_count}_" + os.path.basename(self.file_path_image)
                            file_path = os.path.join(
                                folder_path, copy_file_name)
                            copy_count += 1

                    # Enregistrer l'image dans le fichier de destination
                    image.save(file_path)
                    self.update()  # Mettre à jour la vue de l'arborescence

    def cut(self):  # Copy + Delete
        self.copySelectedItem()
        self.delete()

    def delete(self):  # Supprimer un élément dans l'arborescence
        current_index = self.currentIndex()

        # Récupére le chemin complet du fichier
        file_path = self.model_.filePath(current_index)
        if file_path is not None:
            if os.path.isfile(file_path):
                os.remove(file_path)  # Supprime dans le disque

                self.update()

    # Evenement quand un élément est double click dans l'arborescence
    def itemDoubleClicked(self, index):
        # Récupére le chemin complet du fichier
        file_path = self.model_.filePath(index)
        file_info = QFileInfo(file_path)

        if not file_info.isDir():
            self.setCurrentIndex(index)

            self.parent_.openPicture(file_path)

    def isFileInTree(self, file_path, parent_index=QModelIndex()): #Méthode pour savoir si path est dans l'arborescence
        num_rows = self.model_.rowCount(parent_index)
        for row in range(num_rows):
            index = self.model_.index(row, 0, parent_index)
            if self.model_.filePath(index) == file_path:
                return index
            if self.model_.hasChildren(index):
                result = self.isFileInTree(file_path, index)
                if result.isValid():
                    return result
        return QModelIndex()
    
    def openTree_view(self):  # Méthode pour l'action "Open" dans l'arborescence
        current_index = self.currentIndex()  # Index courrant

        # Récupére le chemin complet du fichier
        file_path = self.model_.filePath(current_index)
        
        self.parent_.openPicture(file_path)  # Ouvre l'image dans la scène