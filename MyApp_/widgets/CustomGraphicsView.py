from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem,QMessageBox, QAction,QMenu
from PyQt5.QtCore import Qt , QRectF, QSizeF, QFileInfo
from PyQt5.QtGui import  *
import cv2
from widgets.CropResultDialog import CropResultDialog 


class CustomGraphicsView(QGraphicsView):
    def __init__(self,app_instance, parent=None):
        super().__init__(parent)

        self.parent_ = parent
        self.scene_ = QGraphicsScene() #Scene de la classe
        self.setScene(self.scene_) # Associe la scène à la vue

        self.setFocusPolicy(Qt.WheelFocus) # Définit le focus sur la molette
        self.setDragMode(QGraphicsView.ScrollHandDrag) # Active le défilement de la vue en mode "main"

        self.app_instance = app_instance
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu_view)


        # Initialiser des variables des méthodes
        self.crop_item = None
        self.crop = False
        self.cropzoom = False
        self.start_pos = None
        self.end_pos = None
        self.current_scene_path = None # Chemin de la scène actuelle
        self.image = None # Image affichée dans la vue
        self.min_zoom_factor = None  # Facteur de zoom minimal
        self.contains = False
        self.contains_menu = False

    
    def enterEvent(self, event):
        # Survol de la souris sur la vue
        super().enterEvent(event)
        self.contains = True
        self.setCursor(Qt.ArrowCursor) # Définir le curseur en flèche

    def leaveEvent(self, event):
        # Sortie de la souris de la vue
        super().leaveEvent(event)
        self.contains = False
        self.setCursor(Qt.BlankCursor) # Masquer le curseur    

    def mousePressEvent(self, event): #Evenement qui permet de créer la zone de sélection sur la scène pour le crop
        if event.button() == Qt.LeftButton and self.crop == True and self.crop_item is None:
            self.crop_item = QGraphicsRectItem() #Rectangle
            self.crop_item.setFlag(QGraphicsRectItem.ItemIsMovable, True) # Permet le déplacement de l'item
            self.crop_item.setFlag(QGraphicsRectItem.ItemIsSelectable, True) # Permet la sélection de l'item
            self.crop_item.setPen(QPen(Qt.blue, 2)) # Définit la couleur et l'épaisseur du contour
            self.scene_.addItem(self.crop_item)

            self.start_pos = self.mapToScene(event.pos())
            self.crop_item.setRect(QRectF(self.start_pos, QSizeF()))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event): #Evenement qui permet d'agrandir le rectangle en fonction de la souris
        if self.crop == True and self.crop_item is not None:
            current_pos = self.mapToScene(event.pos()) # Convertit la position du curseur en coordonnées de la scène
            self.crop_item.setRect(
                QRectF(self.start_pos, current_pos).normalized()) # Met à jour le rectangle de recadrage en fonction de la position actuelle
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event): #Evenement quand la souris n'est plus active sur la scène
        if self.crop == True and self.crop_item is not None:
            self.end_pos = self.mapToScene(event.pos()) # Convertit la position de relâchement en coordonnées de la scène
            self.crop_item.setRect(
                QRectF(self.start_pos, self.end_pos).normalized())
            self.crop_area = self.crop_item.boundingRect() # Récupère la zone de recadrage délimitée par le rectangle

            if self.cropzoom == True:
                # Adapte la scène en fonction du rectangle (outil crop zoom)
                self.fitInView(self.crop_area, Qt.KeepAspectRatio)

            else:
                cropped_image = self.image.copy(self.crop_area.toRect())

                # Afficher la fenêtre de résultat du recadrage
                result_dialog = CropResultDialog(cropped_image)
                result_dialog.exec_()

            
            self.scene_.removeItem(self.crop_item) #Clean la scène du rectangle

            self.parent_.clear_crop()

            self.contains_menu = False
        else:
            super().mouseReleaseEvent(event)

    def Crop(self):
        if not self.image:
            return

        if self.crop:
            # Désactive le crop
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ArrowCursor)
            self.crop = False

        else:
            # Active le crop
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)
            self.crop = True
            self.cropzoom = False

    def CropZoom(self):
        if not self.image:
            return
        if self.cropzoom:
            # Désactive le crop zoom
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ArrowCursor)

            self.cropzoom = False

        else:
            # Active le crop zoom
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)

            self.cropzoom = True
            self.crop = True

    def wheelEvent(self, event): #Evenement de la molette de la souris sur la scène
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y() / 120
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            super().wheelEvent(event)

    def zoom_in(self): #Méthode "Ctrl+" sur la scène
        if self.image != None:
            self.zoom_factor = 1.1
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
            self.scale(self.zoom_factor, self.zoom_factor)

            self.contains_menu = False

    def zoom_out(self): #Méthode "Ctrl-" sur la scène
        if self.image != None:
            self.zoom_factor = 0.9
            self.current_zoom = self.transform().m11()
            if self.current_zoom >= self.min_zoom_factor:
                self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
                self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
                self.scale(self.zoom_factor, self.zoom_factor)
            self.contains_menu = False

    def openPicture(self, file_path):

        # Fonction pour ouvrir et afficher une nouvelle image à partir d'un path
        file_info = QFileInfo(file_path)
        file_name = file_info.fileName()

        self.image = QPixmap(file_path) # Charge l'image à partir du path

        if self.image.isNull():
            QMessageBox.warning(self, "Error", "Failed to load image.")
        else:

            self.current_scene_path = file_path
            item = QGraphicsPixmapItem(self.image)
            self.scene_.clear()
            self.scene_.addItem(item)

            self.zoom_factor = 1.0

            self.parent().label.setText(file_name) #Label du nom de l'image

            self.setSceneRect(QRectF(self.image.rect()))
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

            self.min_zoom_factor = self.transform().m11() #Valeur du zoom initial a l'ouverture de l'image
            
            

    def openImage(self, image): # Permet de convertir une image OpenCV vers un format compatible avec Qt
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convertit l'image OpenCV de BGR à RGB
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_rgb.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)  # Crée une QImage à partir des données de l'image convertie

        self.image = QPixmap.fromImage(q_image) # Convertit la QImage en QPixmap pour l'affichage
        if self.image.isNull():
            QMessageBox.warning(self, "Error", "Failed to load image.")
        else:
            item = QGraphicsPixmapItem(self.image)
            self.scene_.clear()
            self.scene_.addItem(item)

    def showContextMenu_view(self, position):
        # Création d'un objet menu avec la QTreeView comme parent
        menu = QMenu(self)

        # Ajout des actions au menu en utilisant self.app_instance pour accéder à MyApp
        menu.addAction(self.app_instance.zoominAction)
        menu.addAction(self.app_instance.zoomoutAction)
        separator = QAction(self)
        separator.setSeparator(True)
        menu.addAction(separator)
        menu.addAction(self.app_instance.cropAction)
        menu.addAction(self.app_instance.cropzoomAction)

        # Variable booléenne pour suivre l'état du menu
        self.contains_menu = False

        def menu_closed():
            self.contains_menu = True

        # Connecter le signal `aboutToHide` pour détecter quand le menu est fermé
        menu.aboutToHide.connect(menu_closed)

        # Affichage du menu contextuel à la position donnée
        menu.exec_(self.viewport().mapToGlobal(position))