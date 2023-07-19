from PyQt5.QtCore import pyqtSignal , QThread
from ultralytics import YOLO
import cv2,subprocess, random, numpy as np

class Thread_yolo_det(QThread): #Thread pour la détection de Yolo par découpage des images
    progressChanged = pyqtSignal(int)
    finished = pyqtSignal()
    resultsReady = pyqtSignal(dict)

    def __init__(self, size_, model, input, confThreshold, nmsThreshold):
        super().__init__()

        self.model_ = YOLO(model) #Initialisation du model
        self.image = cv2.imread(input)
        #Variables d'instances
        self.size_ = size_
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold


        #Initialisation des variables de classes
        if self.image is not None:
            # Récupérer les dimensions de l'image
            self.height_, self.width_, _ = self.image.shape

        self.stride = int(self.size_/2)

        
        self.progress = 0
        self.total_steps = 0
        self.current_step = 0
        self.classNames = []
        self.boxes = []
        self.scores = []
        self.classIds = []
        self.class_data = {}
        self.boxes_pos = []

    def run(self):
        try:
            total_steps = ((self.height_ - self.stride + 1) // self.stride) * \
                ((self.width_ - self.stride + 1) // self.stride) #Nombre d'itération pour le découpage
            current_step = 0
            for _y in range(0, self.height_ - self.stride + 1, self.stride): #Parcours l'image en verticale 
                y = _y
                for _x in range(0, self.width_ - self.stride + 1, self.stride): #Parcours l'image en horizontale

                    current_step += 1
                    progress = int(current_step / total_steps * 100)
                    self.progressChanged.emit(progress) #Emet le signal pour la barre de progression

                    x = _x

                    if y + self.size_ >= self.height_:
                        y = self.height_ - self.size_

                    if x + self.size_ >= self.width_:
                        x = self.width_ - self.size_

                    # Extraction de la petite image
                    crop_img = self.image[y:y+self.size_, x:x+self.size_]

                    results = self.model_.predict(
                        crop_img, imgsz=self.size_, conf=self.confThreshold) #Prédition sur l'image découpée

                    for result in results:
                        for i in range(len(result.boxes)): #Parcours des boxes détectées
                            x1, y1, x2, y2 = result.boxes.xyxy[i][:4].int()
                            class_id = int(result.boxes.cls[i].item())
                            class_name = self.model_.names[class_id]
                            confidence = result.boxes.conf[i].item()
                            boxe_x, boxe_y, boxe_w, boxe_h = result.boxes.xywh[i][:4]

                            # Ajouter la boîte englobante et la probabilité de classe à la liste

                            self.classNames.append(class_name)
                            self.boxes.append(
                                [int(x+x1.item()), int(y+y1.item()), int(x+x2.item()), int(y+y2.item())])
                            self.scores.append(float(confidence))
                            self.classIds.append(class_id)
                            self.boxes_pos.append(
                                [int(x+boxe_x.item()), int(y+boxe_y.item()), int(boxe_w.item()), int(boxe_h.item())])

            # Appliquer la suppression des non-maximaux
            self.indices = cv2.dnn.NMSBoxes(
                self.boxes, self.scores, self.confThreshold, self.nmsThreshold)

            for i in self.indices: #Indices des boxes restant après NMS

                x1, y1, x2, y2 = self.boxes[i]
                class_name = self.classNames[i]
                confidence = self.scores[i]
                class_id = self.classIds[i]
                x, y, w, h = self.boxes_pos[i]

                # Vérifier si la classe existe déjà dans le dictionnaire
                if class_name in self.class_data:
                    self.class_data[class_name].append(
                        (x1, y1, x2, y2, confidence, class_id, x, y, w, h))
                else:
                    self.class_data[class_name] = [
                        (x1, y1, x2, y2, confidence, class_id, x, y, w, h)]
                    
            self.resultsReady.emit(self.class_data)
            self.finished.emit()

        except subprocess.CalledProcessError as e:
            print("Subprocess error:", e)


class Thread_yolo_det_default(QThread): #Gére la détection de Yolo par défault c'est à dire sans découpage de l'image
    finished = pyqtSignal() #Signal Finish du thread
    resultsReady = pyqtSignal(dict) #Signal pour transmettre les outputs du thread

    def __init__(self, model, input,confThreshold):
        super().__init__()
        self.model_ = YOLO(model) #Initialisation du model
        self.image = cv2.imread(input) 
        self.input = input
        self.confThreshold = confThreshold
        self.class_data = {} #Initialisation du dico

    def run(self):
        try:
            self.results = self.model_.predict(
                source=self.input, conf=self.confThreshold ) #Détection sur l'image 

            for result in self.results:
                for i in range(len(result.boxes)):
                    x1, y1, x2, y2 = result.boxes.xyxy[i][:4].int() #Position de la boxe
                    class_id = int(result.boxes.cls[i].item())
                    class_name = self.model_.names[class_id]
                    confidence = result.boxes.conf[i].item() #Score de la classe
                    boxe_x, boxe_y, boxe_w, boxe_h = result.boxes.xywh[i][:4].int(
                    ) #Information sur la boxe pour le fichier .txt de sortie

                    # Vérifier si la classe existe déjà dans le dictionnaire
                    if class_name in self.class_data:
                        self.class_data[class_name].append((x1.item(), y1.item(), x2.item(), y2.item(
                        ), confidence, class_id, boxe_x.item(), boxe_y.item(), boxe_w.item(), boxe_h.item()))
                    else:
                        self.class_data[class_name] = [(x1.item(), y1.item(), x2.item(), y2.item(
                        ), confidence, class_id, boxe_x.item(), boxe_y.item(), boxe_w.item(), boxe_h.item())]

            self.resultsReady.emit(self.class_data)
            self.finished.emit()
        except subprocess.CalledProcessError as e:
            print("Subprocess error:", e)


class Thread_yolo_seg_default(QThread): #Gére la segmentation de Yolo par défault c'est à dire sans découpage de l'image
    finished = pyqtSignal() # Signal de fin du thread


    def __init__(self, model, input, output, confThreshold, semantic):
        super().__init__()
        self.model_ = YOLO(model) #Initialisation du model
        self.image = cv2.imread(input) #Lecture de l'image 
        self.input = input
        self.output = output
        self.confThreshold = confThreshold
        self.semantic = semantic

        if self.semantic:
            class_names = self.model_.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
                    
        
    def overlay(self,image, mask, color, alpha): #Transpose l'image avec le masque 
        
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        # Combinaison de l'image d'entrée et du masque coloré avec une transparence alpha
        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        return image_combined

    def run(self):
        try:
            self.h, self.w, _ = self.image.shape # Récupération de la hauteur et de la largeur de l'image d'entrée
            self.results = self.model_.predict(
                source=self.input, stream=True, conf=self.confThreshold ) # Détection des objets dans l'image

            for r in self.results:
                masks = r.masks # Liste des masques des objets détectés 
                boxes = r.boxes
            if masks is not None:
                for seg, box in zip(masks.data.cpu().numpy(), boxes):
                    seg = cv2.resize(seg, (self.w, self.h)) # Redimensionnement du masque pour correspondre à l'image d'entrée
                    if self.semantic:
                        self.image = self.overlay(self.image, seg, self.colors[int(box.cls)], 0.4) # Superposition du masque coloré sur l'image d'entrée
                    else:
                        colors = [random.randint(0, 255) for _ in range(3)] # Génération de couleurs aléatoires pour chaque masque
                        self.image = self.overlay(self.image, seg, colors, 0.4) # Superposition du masque coloré sur l'image d'entrée

            cv2.imwrite(self.output, self.image)

            self.finished.emit()
        except subprocess.CalledProcessError as e:
            print("Subprocess error:", e)


class Thread_yolo_seg(QThread): #Thread pour la détection de Yolo par découpage des images
    progressChanged = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, size_, model, input,output, confThreshold,semantic):
        super().__init__()

        self.model_ = YOLO(model) #Initialisation du model
        self.image = cv2.imread(input)
        #Variables d'instances
        self.input = input
        self.output = output
        self.size_ = size_
        self.confThreshold = confThreshold
        self.semantic = semantic

        if self.semantic:
            class_names = self.model_.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

        #Initialisation des variables de classes
        if self.image is not None:
            # Récupérer les dimensions de l'image
            self.height_, self.width_, _ = self.image.shape

        self.stride = int(self.size_/2)
        self.bord = int(self.size_/4)

        self.progress = 0
        self.total_steps = 0
        self.current_step = 0

    def overlay(self, image, mask, color, alpha): #Transpose l'image avec le masque 
        overlay_mask = np.zeros_like(image)

        mask_resized = cv2.resize(mask, (overlay_mask.shape[1], overlay_mask.shape[0]))

        overlay_mask_resized = np.zeros_like(overlay_mask)
        overlay_mask_resized[mask_resized != 0] = color


        # Combinaison de l'image d'entrée et du masque coloré avec une transparence alpha
        overlay = cv2.addWeighted(image, 1 - alpha, overlay_mask_resized, alpha, 0)
        result = np.where(overlay_mask_resized != 0, overlay, image)

        return result

    def run(self):
        try:
            self.total_steps = ((self.height_ - self.stride + 1) // (self.stride - self.bord)) * ((self.width_ - self.stride + 1) // (self.stride - self.bord))
            self.current_step = 0
            for _y in range(0, self.height_ - self.stride + 1, self.stride - self.bord): #Parcours horizontalement l'image 
                y = _y
                for _x in range(0, self.width_ - self.stride + 1, self.stride - self.bord): #Parcours verticalement l'image
                    x = _x

                    self.current_step += 1
                    self.progress = int(self.current_step / self.total_steps * 100)
                    self.progressChanged.emit(self.progress) #Emet le signal pour la barre de progression

                    if y + self.size_ >= self.height_:
                        y = self.height_ - self.size_

                    if x + self.size_ >= self.width_:
                        x = self.width_ - self.size_

                    crop_img = self.image[y:y+self.size_, x:x+self.size_]

                    self.results = self.model_.predict(
                        crop_img, imgsz=self.size_, conf=self.confThreshold)

                    for r in self.results:
                        boxes = r.boxes
                        masks = r.masks

                    if masks is not None:
                        for seg, box in zip(masks.data.cpu().numpy(), boxes):
                            seg = cv2.resize(seg, (self.size_, self.size_))  # Redimensionner à la taille de l'image découpée
                            if self.semantic:
                                overlay_img = self.overlay(crop_img, seg, self.colors[int(box.cls)], 0.5)
                            else:
                                colors = [random.randint(0, 255) for _ in range(3)] #Géneration de couleur aléatoire pour le masque
                                overlay_img = self.overlay(crop_img, seg, colors, 0.5) #Applique le masque sur l'image découpée
                            self.image[y:y+self.size_, x:x+self.size_] = overlay_img #Applique l'image découpée sur l'image d'entrée

            cv2.imwrite(self.output, self.image)
            self.finished.emit()

        except subprocess.CalledProcessError as e:
            print("Subprocess error:", e)