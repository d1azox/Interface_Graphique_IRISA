from PyQt5.QtWidgets import QMainWindow, QAction, QHBoxLayout, QSplitter, QGroupBox, QVBoxLayout, QWidget, QListWidget, QGridLayout, QListWidgetItem, QMenu, QGraphicsView, QFileDialog
from PyQt5.QtCore import Qt, QModelIndex
import os
import shutil
import cv2
import sys
from widgets.CustomGroupBox import CustomGroupBox 
from widgets.CustomGroupBoxOverride import CustomGroupBoxOverride 
from widgets.CustomTreeView import CustomTreeView
from widgets.TerminalTextEdit import TerminalTextEdit

from windows.SecondaryWindow import SecondaryWindow

class MainWindow(QMainWindow):
    """Main Window."""

    def __init__(self, app_instance, parent=None):
        """Initializer."""
        super().__init__(parent)

        self.app_instance = app_instance #Instance de l'Application

        self.resize(1280, 1024) #Taille du la fênetre principale
        self.setWindowTitle("Interface")

        #Mise en place de l'interface
        self._createLayouts()
        self.setupUI()

    def _createLayouts(self):
        # Layout principal horizontal
        main_layout = QHBoxLayout()

        # Création du QSplitter
        self.splitter = QSplitter(Qt.Horizontal)

        # Layout pour le premier groupe de widgets
        self.group_box1 = QGroupBox()
        self.group_box1.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 9px; margin-top: 0.5em; }")
        self.layout1 = QVBoxLayout(self.group_box1)

        # Deuxième layout
        self.group_box2 = QGroupBox()
        self.group_box2.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 9px; margin-top: 0.5em; }")
        self.layout2 = QVBoxLayout(self.group_box2)

        # Ajouter les conteneurs au QSplitter
        self.splitter.addWidget(self.group_box1)
        self.splitter.addWidget(self.group_box2)
        self.splitter.setSizes(
            [int(self.width() * 0.8), int(self.width() * 0.2)]) #Disposition par défaut des conteneurs

        # Ajouter le layout principal au layout de la fenêtre
        main_layout.addWidget(self.splitter)

        # Défini le layout principal comme layout de la fenêtre
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def setupUI(self):

        # Création du QSplitter
        self.splitter2 = QSplitter(Qt.Vertical)

        self.splitter3 = QSplitter(Qt.Vertical)

        self.all_scenes = []


        # Création du layout en grille
        self.grille = QGridLayout()

        self.view = CustomGroupBoxOverride(self.app_instance, self) #Scène principale

        

        #Initialisation des variables de la scène principale
        self.graphicsView = self.view.graphics_view 
        self.scene = self.view.graphics_view.scene_

        self.all_scenes.append(self.graphicsView) #Ajoute la première scène a la liste des scènes 

        #Création des scènes secondaires
        self.view2 = CustomGroupBox(self.app_instance, self)
        self.view3 = CustomGroupBox(self.app_instance, self)
        self.view4 = CustomGroupBox(self.app_instance, self)

        #Disposition en grille
        self.grille.addWidget(self.view, 0, 0)
        self.grille.addWidget(self.view2, 0, 1)
        self.grille.addWidget(self.view3, 1, 0)
        self.grille.addWidget(self.view4, 1, 1)

        
        #List_ widget pour afficher les classes et la confiance des détections
        self.list_widget = QListWidget()
        self.list_widget.setVisible(False)

        # Connecter le signal itemClicked à un slot pour mettre en surbrillance le rectangle correspondant
        self.list_widget.itemPressed.connect(self.highlightRectangle)
        self.index = None 

        #Arborescence
        self.tree_view = CustomTreeView(self)  # Création du QTreeView pour afficher l'arborescence

        #Menu de l'arborescence(clic droit)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.showContextMenu)

        #Disposition 50/50 entre la list_widget et l'arborescence
        self.splitter3.addWidget(self.list_widget)
        self.splitter3.addWidget(self.tree_view)

        self.splitter3.setSizes(
            [int(self.width() * 0.5), int(self.width() * 0.5)])

        self.layout2.addWidget(self.splitter3) #Ajout au layout

        #Terminal
        self.textEdit = TerminalTextEdit()
        sys.stdout = self.textEdit.emit_stream #Þermet de dirigé les outputs vers le terminal de l'interface 

        #Disposition entre la scène et le terminal
        widget = QWidget()
        widget.setLayout(self.grille)

        self.splitter2.addWidget(widget)
        self.splitter2.addWidget(self.textEdit)

        self.splitter2.setSizes(
            [int(self.width() * 0.8), int(self.width() * 0.2)])

        self.layout1.addWidget(self.splitter2) #Ajoute le spliter au layout

    def showContextMenu(self, position): #Menue qui s'affiche quand on fait clic droit sur l'arborescence
        # Création d'un objet menu avec la QTreeView comme parent
        menu = QMenu(self.tree_view)

        # Ajout des actions au menu en utilisant self.app_instance pour accéder à l'instance de l'Application
        menu.addAction(self.app_instance.openTree_viewAction)
        menu.addAction(self.app_instance.saveAction)
        menu.addAction(self.app_instance.copyAction)
        menu.addAction(self.app_instance.pasteAction)
        separator = QAction(self)
        separator.setSeparator(True)
        menu.addAction(separator)
        menu.addAction(self.app_instance.cutAction)
        menu.addAction(self.app_instance.deleteAction)
        menu.addAction(self.app_instance.renameAction)
        menu.addAction(self.app_instance.addsceneAction)

        # Affichage du menu contextuel à la position donnée
        menu.exec_(self.tree_view.viewport().mapToGlobal(position))

    def apply_zoom_in_action_to_all_scenes(self):
        for scene in self.all_scenes:  # all_scenes est une liste contenant toutes les instances de la scène principale
            if scene.contains_menu == True or scene.contains == True:
                scene.zoom_in()  # Appliquer l'action de zoom in sur chaque instance de la scène
                return
        for scene in self.all_scenes: 
            scene.zoom_in()

    def apply_zoom_out_action_to_all_scenes(self):
        for scene in self.all_scenes:  # all_scenes est une liste contenant toutes les instances de la scène principale
            if scene.contains_menu == True or scene.contains == True:
                scene.zoom_out()  # Appliquer l'action de zoom out sur chaque instance de la scène
                return
        for scene in self.all_scenes: 
            scene.zoom_out()

    def apply_crop_action_to_all_scenes(self):
        for scene in self.all_scenes:  
            scene.Crop() # Appliquer l'action de crop sur chaque instance de la scène

    def apply_crop_zoom_action_to_all_scenes(self):
        for scene in self.all_scenes: 
            scene.CropZoom() # Appliquer l'action de crop plus zoom sur chaque instance de la scène

    def clear_crop(self): # Clear chaque instance de la scène
        for scene in self.all_scenes:  
            scene.crop_item = None
            scene.crop = False
            scene.cropzoom = False
            

            scene.setDragMode(QGraphicsView.ScrollHandDrag) #Met le curseur par défault

    def saveImage(self): #Méthode pour le tool "Save"
        if self.graphicsView.current_scene_path is not None:
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", self.graphicsView.current_scene_path, "Images (*.png *.jpg *.jpeg *.tif)")

            if save_path:
                save_folder = os.path.dirname(save_path)  # Récupérer le dossier de destination

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)  # Créer le dossier de destination s'il n'existe pas

                shutil.copy2(self.graphicsView.current_scene_path, save_path)  # Copier l'image actuelle vers le fichier de destination

    def addScene(self): #Permet d'ajouter un scène sur l'interface
        current_index_ = self.tree_view.currentIndex()
        # Récupérer le chemin complet du fichier
        file_path = self.tree_view.model_.filePath(current_index_)

        if file_path:
            if len(self.view2.graphics_view.scene_.items()) == 0: #Si la scène 2 est vide alors ajoute une image
                self.all_scenes.append(self.view2.graphics_view)
                self.view2.show()
                self.view2.graphics_view.openPicture(file_path)

            elif len(self.view3.graphics_view.scene_.items()) == 0: #Si la scène 3 est vide alors ajoute une image
                self.all_scenes.append(self.view3.graphics_view)
                self.view3.show()
                self.view3.graphics_view.openPicture(file_path)

            elif len(self.view4.graphics_view.scene_.items()) == 0: #Si la scène 4 est vide alors ajoute une image
                self.all_scenes.append(self.view4.graphics_view)
                self.view4.show()
                self.view4.graphics_view.openPicture(file_path)

    def run(self):  # Execute la deuxième fenêtre 'Run Option"
        self.secondary_window = SecondaryWindow(self)
        self.secondary_window.exec_()

    def open_tool(self):  # Méthode pour ouvrir une image avec le tool "Open"

        # Choisir l'image dans son disque
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp *.tif)")

        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]

            # Vérifier si le fichier est déjà présent dans l'arborescence
            index = self.tree_view.isFileInTree(file_name)

            if index.isValid():
                # Si oui, alors juste sélectionner l'élement dans l'arborescence
                self.tree_view.itemDoubleClicked(index)
            else:
                # Sinon ouvre l'image dans la scène principale
                self.openPicture(file_name)


    def openPicture(self, file_path):  # Ouvre une image dans la scène principale

        test_path = os.path.join(os.path.dirname(file_path), os.path.splitext(
            os.path.basename(file_path))[0]+".txt")
        # Vérifie si l'image a un fichier .txt correspondant à un détection déjà effectué
        if os.path.exists(test_path):
            self.picture_path = file_path
            self.data_dict = self.read_txt_file(test_path)  # Read .txt
            if self.data_dict:
                self.list_widget.clear()
                self.load_list_widget(self.data_dict)  # Charge les élements
                # Affiche la list_widet sur l'interface
                self.list_widget.setVisible(True)
        else:
            self.list_widget.clear()
            self.list_widget.setVisible(False)
        self.graphicsView.openPicture(file_path)

    # Charge les éléments du dico dans la list_widget
    def load_list_widget(self, data_dico):

        for class_id, class_data in data_dico.items():

            for bbox in class_data:

                conf_score = bbox['conf_score']

                item = QListWidgetItem(f"{class_id} ({conf_score:.2f})")

                self.list_widget.addItem(item)

    def read_txt_file(self, file_path):  # Read le fichier .txt output de la détection
        data_dict = {}

        with open(file_path, 'r') as file:  # Ouvre le fichier en lecture
            for line in file:  # Parcours les lignes
                line = line.strip()
                if line:
                    # Extrait les données dans chaque ligne
                    class_id, x, y, width, height, conf_score = line.split()
                    # Identifiant de la classe détectée
                    class_id = int(class_id)
                    x = int(x)  # Coordonnée x du centre de la boxe
                    y = int(y)  # Coordonnée y du centre de la boxe
                    width = int(width)  # Łargeur de la boxe
                    height = int(height)  # Hauteur de la boxe
                    conf_score = float(conf_score)  # Confiance de la détection

                    # Convertir pour récupérer la position du périmetre de la boxe
                    x1 = int(x - width/2)
                    x2 = int(x + width/2)
                    y1 = int(y - height/2)
                    y2 = int(y + height/2)

                    if class_id not in data_dict:
                        data_dict[class_id] = []

                    data_dict[class_id].append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'conf_score': conf_score
                    })
        return data_dict

    # Fonction qui met en surbrillance l'élement sélectionné dans la list_widget
    def highlightRectangle(self, item):
        # Obtenir l'indice de l'élément cliqué dans la liste
        index_current = self.list_widget.row(item)

        ind = 0  # Indicateur pour vérifier s'il y a eu un changement d'élément sélectionné
        if self.index == None or self.index != index_current:

            self.index = self.list_widget.row(item)
            ind = 1

        t = 0  # Compteur pour parcourir les éléments dans le dictionnaire data_dict
        image = cv2.imread(self.picture_path)  # Charge l'image

        image_modified = image.copy()  # Créer une copie de l'image

        self.index = index_current  # Mettre à jour l'indice actuel avec index_current
        for _, class_data in self.data_dict.items():
            for bbox in class_data:
                if t == index_current and ind == 1:
                    # Créer un rectangle en surbrillance
                    cv2.rectangle(
                        image_modified, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 1, lineType=cv2.LINE_AA)
                t += 1  # Incrémenter le compteur

        if ind == 0:
            # Si ind est 0, cela signifie qu'il n'y a pas eu de changement d'élément sélectionné,
            self.index = None
            self.list_widget.setCurrentIndex(QModelIndex())
            # On ouvre l'image origine sans modif
            self.graphicsView.openImage(image)
            return


        # Afficher l'image modifiée avec le rectangle sélectionné mis en évidence
        self.graphicsView.openImage(image_modified)