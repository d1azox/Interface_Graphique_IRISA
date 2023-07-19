from PyQt5.QtWidgets import QDialog, QVBoxLayout,QGroupBox, QGridLayout,QMessageBox, QLabel,QFrame,QPushButton,QCheckBox,QComboBox,QSpinBox,QDoubleSpinBox,QFileDialog
from PyQt5.QtCore import Qt , QEventLoop, QFile, QIODevice
from PyQt5.QtXml import  QDomDocument
import cv2
import os
from widgets.ProgressDialog import ProgressDialog 
from widgets.TableWidget_Dialog import TableWidget_Dialog
from widgets.Help import HelpDialog_detection,  HelpDialog_segmentation, HelpDialog_ins_segmentation

from threads.Thread_Segmentation import Thread_Segmentation
from threads.Thread_Yolo import Thread_yolo_det, Thread_yolo_det_default, Thread_yolo_seg_default, Thread_yolo_seg

class SecondaryWindow(QDialog):  # Fênetrede de la partie Run de l'interface
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Option")
        self.setFixedSize(1000, 525)

        main_layout = QVBoxLayout()  # Layout principal
        self.setLayout(main_layout)

        group_box = QGroupBox()
        group_box.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 9px; padding: 20px; }")
        main_layout.addWidget(group_box)

        group_layout = QVBoxLayout()
        group_box.setLayout(group_layout)

        #Panel Section
        self.group_box1 = QGroupBox("Panel")
        self.group_box1.setFixedHeight(150)
        self.group_box1.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 9px; padding: 20px; }")

        # Open Image Section
        self.grille = QGridLayout()

        self.group_box1.setLayout(self.grille)

        group_layout.addWidget(self.group_box1)

        self.label_open = QLabel()
        self.label_open.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.label_open.setFixedSize(360, 25)
        self.grille.addWidget(self.label_open, 0, 0)

        self.button_open = QPushButton("Open", self)
        self.button_open.setFixedSize(175, 25)
        self.button_open.clicked.connect(self.openImage)
        self.grille.addWidget(self.button_open, 0, 1)

        # Ouverture d'un dossier
        self.checkbox_open = QCheckBox("Open dir", self)
        self.checkbox_open.stateChanged.connect(self.checkbox_openStateChanged)
        self.checkbox_open.setChecked(False)
        self.checkbox_open.setFixedSize(150, 25)
        self.grille.addWidget(self.checkbox_open, 0, 2)

        # Save File Section
        self.label_save = QLabel()
        self.label_save.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.grille.addWidget(self.label_save, 1, 0)
        self.label_save.setFixedSize(360, 25)

        self.button_save = QPushButton("Save", self)
        self.button_save.clicked.connect(self.output_file)
        self.grille.addWidget(self.button_save, 1, 1)
        self.button_save.setFixedSize(175, 25)

        #Optionnel - détection Yolo
        self.checkbox_text = QCheckBox("Save Text File")
        self.checkbox_image = QCheckBox("Save Image File")

        self.checkbox_text.stateChanged.connect(self.checkbox_textStateChanged)

        self.checkbox_image.stateChanged.connect(
            self.checkbox_imageStateChanged)
        self.checkbox_text.setChecked(True)
        self.checkbox_image.setChecked(True)

        self.checkbox_text.setFixedSize(120, 25)

        self.grille.addWidget(self.checkbox_image, 1, 2)
        self.grille.addWidget(self.checkbox_text, 1, 3)
        

        # Définition des largeurs minimales des colonnes
        self.grille.setColumnMinimumWidth(0, 400)
        self.grille.setColumnMinimumWidth(1, 200)
        self.grille.setColumnMinimumWidth(2, 100)   
        self.grille.setColumnMinimumWidth(3, 100)

        # Parameters Section
        self.group_box2 = QGroupBox("Parameters")
        self.group_box2.setFixedHeight(150)
        self.group_box2.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 9px; padding: 20px; }")
                

        self.options_layout = QGridLayout()

        self.group_box2.setLayout(self.options_layout)

        group_layout.addWidget(self.group_box2)

        self.interface_detection()
        self.interface_segmentation()
        self.interface_instance_segmentation()

        # QComboBox detection tool

        self.combobox_detect = QComboBox(self)
        self.combobox_detect.addItem("Object Detection")
        self.combobox_detect.addItem("Semantic Segmentation")
        self.combobox_detect.addItem("Instance Segmentation")
        self.combobox_detect.currentIndexChanged.connect(self.combobox_detectIndexChanged)  # Correction du nom de la méthode
        
        self.grille.addWidget(self.combobox_detect, 2, 0)
        self.combobox_detect.setFixedSize(200,25)

        # Option avancées
        self.checkbox_advance = QCheckBox("Advanced parameters")
        self.checkbox_advance.stateChanged.connect(
            self.checkboxadvanceStateChanged)
        self.checkbox_advance.setChecked(False)
        
        self.grille.addWidget(self.checkbox_advance, 2, 2)

        # Algorithm
        self.combobox_option = QComboBox()
        self.combobox_option.currentTextChanged.connect(
            self.loadCombobox)
        self.label_algo = QLabel("Algorithm:")
        self.options_layout.addWidget(self.label_algo, 0, 0)
        self.options_layout.addWidget(self.combobox_option, 0, 1)
        self.label_algo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_algo.setFixedSize(100, 25)
        self.combobox_option.adjustSize()
 
        # Run Button
        self.button_ok = QPushButton("OK", self)
        self.button_ok.clicked.connect(self.run_prog)
        group_layout.addWidget(self.button_ok, alignment=Qt.AlignRight)

        # Image dans la scène est l'input par défault
        self.label_open.setText(self.parent().graphicsView.current_scene_path)

        self.combobox_detectIndexChanged(0) #Par défault index 0

        

    # Les méthodes suivant permet de gérer l'états des checkbox et combobox :  que doit être afficher / désafficher sur l'interface

    def combobox_detectIndexChanged(self, index):
        self.checkbox_advance.setChecked(False)
        self.combobox_option.clear()
        self.setupOption()
        self.loadCombobox()
        if index == 0:  # Detection Objet
            self.hideSegmentationButtons()
            self.hideAdvanceSegmentation()
            self.hideAdvanceDetection()
            self.hideInstanceSegmentationButtons()
            self.hideAdvanceInstanceSegmentationButtons()
            self.showDetectionButtons()
        elif index == 1:  # Semantic Segmentation
            self.hideDetectionButtons()
            self.hideAdvanceSegmentation()
            self.hideAdvanceDetection()
            self.hideInstanceSegmentationButtons()
            self.hideAdvanceInstanceSegmentationButtons()
            self.showSegmentationButtons()
        elif index == 2:  # Instance Segmentation
            self.hideDetectionButtons()
            self.hideAdvanceSegmentation()
            self.hideAdvanceDetection()
            self.hideSegmentationButtons()
            self.hideAdvanceInstanceSegmentationButtons()
            self.showInstanceSegmentationButtons()

    def checkbox_openStateChanged(self, state):
        self.label_open.clear()
        self.label_save.clear()

    def checkboxadvanceStateChanged(self, state):

        self.combobox_option.clear()
        self.setupOption()
        self.loadCombobox()
        
        if state == Qt.Checked and self.combobox_detect.currentIndex() ==0:
            self.hideAdvanceSegmentation()
            self.hideDetectionButtons()
            self.hideSegmentationButtons()
            self.hideInstanceSegmentationButtons()
            self.hideAdvanceInstanceSegmentationButtons()
            self.showAdvanceDetection()

            self.group_box2.setFixedHeight(225)
        elif state == Qt.Checked and self.combobox_detect.currentIndex() ==1:
            self.hideAdvanceDetection()
            self.hideDetectionButtons()
            self.hideSegmentationButtons()
            self.hideInstanceSegmentationButtons()
            self.hideAdvanceInstanceSegmentationButtons()
            self.showAdvanceSegmentation()

            self.group_box2.setFixedHeight(225)

        elif state == Qt.Checked and self.combobox_detect.currentIndex() ==2:
            self.hideAdvanceDetection()
            self.hideDetectionButtons()
            self.hideSegmentationButtons()
            self.hideInstanceSegmentationButtons()
            self.hideAdvanceSegmentation()
            self.showAdvanceInstanceSegmentationButtons()

            self.group_box2.setFixedHeight(200)

        elif not state == Qt.Checked and self.combobox_detect.currentIndex() ==0:
            self.hideAdvanceDetection()
            self.group_box2.setFixedHeight(140)
        elif not state == Qt.Checked and self.combobox_detect.currentIndex() ==1:
            self.hideAdvanceSegmentation()
            self.group_box2.setFixedHeight(140)
        elif not state == Qt.Checked and self.combobox_detect.currentIndex() ==2:
            self.hideAdvanceInstanceSegmentationButtons()
            self.group_box2.setFixedHeight(140)

    # Les méthodes suivantes (show / hide) permet d'afficher ou de désafficher l'interface détection / segmentation / option avancée

    def showDetectionButtons(self):
        self.button_open_model.setVisible(True)
        self.label_model.setVisible(True)
        self.checkbox_text.setVisible(True)
        self.checkbox_image.setVisible(True)
        self.spinbox_conf.setVisible(True)
        self.label_conf.setVisible(True)

    def hideDetectionButtons(self):
        self.button_open_model.setVisible(False)
        self.label_model.setVisible(False)
        self.checkbox_text.setVisible(False)
        self.checkbox_image.setVisible(False)
        self.spinbox_conf.setVisible(False)
        self.label_conf.setVisible(False)

    def showAdvanceDetection(self):
        self.spinbox_nms.setVisible(True)
        self.spinbox_conf.setVisible(True)
        self.spinbox_size.setVisible(True)
        self.button_help_detection.setVisible(True)
        self.label_nms.setVisible(True)
        self.label_size.setVisible(True)
        self.label_conf.setVisible(True)
        self.label_model.setVisible(True)
        self.button_open_model.setVisible(True)
        self.checkbox_text.setVisible(True)
        self.checkbox_image.setVisible(True)

    def hideAdvanceDetection(self):
        self.spinbox_nms.setVisible(False)
        self.spinbox_size.setVisible(False)
        self.button_help_detection.setVisible(False)
        self.label_nms.setVisible(False)
        self.label_size.setVisible(False)

    def showAdvanceSegmentation(self):
        self.button_open_file.setVisible(True)
        self.combobox_program.setVisible(True)
        self.button_property.setVisible(True)
        self.button_help_segmentation.setVisible(True)
        self.label_program.setVisible(True)
        self.button_open_weight.setVisible(True)
        self.combobox_weight.setVisible(True)
        self.label_weight.setVisible(True)

    def hideAdvanceSegmentation(self):
        self.button_open_file.setVisible(False)
        self.combobox_program.setVisible(False)
        self.button_property.setVisible(False)
        self.button_help_segmentation.setVisible(False)
        self.label_program.setVisible(False)
        self.deplace_segmentation(True)

    def showSegmentationButtons(self):
        self.button_open_weight.setVisible(True)
        self.combobox_weight.setVisible(True)
        self.label_weight.setVisible(True)
        self.deplace_segmentation(True)

    def hideSegmentationButtons(self):
        self.button_open_weight.setVisible(False)
        self.combobox_weight.setVisible(False)
        self.label_weight.setVisible(False)
        self.deplace_segmentation(False)

    def showInstanceSegmentationButtons(self):
        self.button_open_model_seg.setVisible(True)
        self.label_model_seg.setVisible(True)
        self.combobox_model_seg.setVisible(True)
        self.label_conf_seg.setVisible(True)
        self.spinbox_conf_seg.setVisible(True)

    def hideInstanceSegmentationButtons(self):
        self.button_open_model_seg.setVisible(False)
        self.label_model_seg.setVisible(False)
        self.combobox_model_seg.setVisible(False)
        self.label_conf_seg.setVisible(False)
        self.spinbox_conf_seg.setVisible(False)

    def showAdvanceInstanceSegmentationButtons(self):
        self.label_size_seg.setVisible(True)
        self.spinbox_size_seg.setVisible(True)
        self.label_conf_seg.setVisible(True)
        self.spinbox_conf_seg.setVisible(True)
        self.button_open_model_seg.setVisible(True)
        self.label_model_seg.setVisible(True)
        self.combobox_model_seg.setVisible(True)
        self.button_help_ins_seg.setVisible(True)

    def hideAdvanceInstanceSegmentationButtons(self):
        self.label_size_seg.setVisible(False)
        self.spinbox_size_seg.setVisible(False)
        self.button_help_ins_seg.setVisible(False)

    # Met à jour à l'interface de 'run option' si l'option avancé est cochée
    def deplace_segmentation(self, deplace):
        if deplace:
            self.options_layout.addWidget(self.label_weight, 1, 0)
            self.options_layout.addWidget(self.combobox_weight, 1, 1)
            self.options_layout.addWidget(self.button_open_weight, 1, 2)
        else:
            self.options_layout.addWidget(self.label_weight, 2, 0)
            self.options_layout.addWidget(self.combobox_weight, 2, 1)
            self.options_layout.addWidget(self.button_open_weight, 2, 2)

    def interface_segmentation(self):

        self.combobox_program = QComboBox()
        self.label_program = QLabel("Program:")
        self.options_layout.addWidget(self.label_program, 1, 0)
        self.options_layout.addWidget(self.combobox_program, 1, 1)

        self.combobox_program.setFixedSize(580, 25)
        self.label_program.setFixedSize(100, 25)


        self.button_open_file = QPushButton("Open Program File", self)
        self.button_open_file.clicked.connect(self.openProgramFile)
        self.options_layout.addWidget(self.button_open_file, 1, 2)

        self.button_open_file.setFixedSize(150, 25)

        self.combobox_weight = QComboBox()
        self.label_weight = QLabel("Weights:")
        self.options_layout.addWidget(self.label_weight, 2, 0)
        self.options_layout.addWidget(self.combobox_weight, 2, 1)

        self.combobox_weight.setFixedSize(580, 25)
        self.label_weight.setFixedSize(100, 25)

        self.button_open_weight = QPushButton("Open Weight File", self)
        self.button_open_weight.clicked.connect(self.openWeightFile)
        self.options_layout.addWidget(self.button_open_weight, 2, 2)

        self.button_open_weight.setFixedSize(150, 25)

        self.button_property = QPushButton("Program Parameters", self)
        self.button_property.clicked.connect(self.opentable_segmentation)
        self.options_layout.addWidget(self.button_property, 3, 1, alignment= Qt.AlignCenter )

        self.button_property.setFixedSize(150, 25)

        self.button_help_segmentation = QPushButton("Help", self)
        self.button_help_segmentation.clicked.connect(self.openhelp_segmentation)
        self.options_layout.addWidget(self.button_help_segmentation, 3, 2)

        self.button_help_segmentation.setFixedSize(100, 25)

    def interface_detection(self):

        self.combobox_model = QComboBox()
        self.label_model = QLabel("Model:")
        self.options_layout.addWidget(self.label_model, 1, 0)
        self.options_layout.addWidget(self.combobox_model, 1, 1)

        self.combobox_model.setFixedSize(580, 25)
        self.label_model.setFixedSize(100, 25)


        self.button_open_model = QPushButton("Open Model File", self)
        self.button_open_model.clicked.connect(self.openModelFile)
        self.options_layout.addWidget(self.button_open_model, 1, 2)

        self.button_open_model.setFixedSize(150, 25)

        self.spinbox_conf = QDoubleSpinBox()
        self.label_conf = QLabel("confThreshold:")

        # Définir la plage de valeurs possibles
        self.spinbox_conf.setRange(0.0, 1.0)
        self.spinbox_conf.setValue(0.5)  # Valeur initiale

        # Définir l'incrément/decrement pour chaque clic
        self.spinbox_conf.setSingleStep(0.1)
        self.spinbox_conf.setDecimals(1)
        self.spinbox_conf.setFixedWidth(100)

        self.options_layout.addWidget(self.label_conf, 2, 0)
        self.options_layout.addWidget(self.spinbox_conf,2, 1)

       

        self.button_help_detection = QPushButton("Help", self)
        self.button_help_detection.clicked.connect(self.openhelp_detection)
        self.options_layout.addWidget(self.button_help_detection, 2, 2)

        self.button_help_detection.setFixedSize(100, 25)

        
        self.spinbox_size = QSpinBox()
        self.label_size = QLabel("Size:")

        # Définir la plage de valeurs possibles
        self.spinbox_size.setRange(0, 640)
        self.spinbox_size.setValue(320)  # Valeur initiale

        # Définir l'incrément/decrement pour chaque clic
        self.spinbox_size.setSingleStep(1)
        self.spinbox_size.setFixedWidth(100)

        self.options_layout.addWidget(self.label_size, 3, 0)
        self.options_layout.addWidget(self.spinbox_size, 3, 1)

        self.spinbox_nms = QDoubleSpinBox()
        self.label_nms = QLabel("nmsThreshold:")

        # Définir la plage de valeurs possibles
        self.spinbox_nms.setRange(0.0, 1.0)
        self.spinbox_nms.setValue(0.3)  # Valeur initiale

        # Définir l'incrément/decrement pour chaque clic
        self.spinbox_nms.setSingleStep(0.1)
        self.spinbox_nms.setDecimals(1)
        self.spinbox_nms.setFixedWidth(100)

        self.options_layout.addWidget(self.label_nms, 4, 0)
        self.options_layout.addWidget(self.spinbox_nms, 4, 1)

    def interface_instance_segmentation(self):

        self.combobox_model_seg = QComboBox()
        self.label_model_seg = QLabel("Model:")
        self.options_layout.addWidget(self.label_model_seg, 1, 0)
        self.options_layout.addWidget(self.combobox_model_seg, 1, 1)

        self.combobox_model.setFixedSize(580, 25)
        self.label_model.setFixedSize(100, 25)

        self.button_open_model_seg = QPushButton("Open Model File", self)
        self.button_open_model_seg.clicked.connect(self.openModel_Segmentation_File)
        self.options_layout.addWidget(self.button_open_model_seg, 1, 2)

        self.button_open_model.setFixedSize(150, 25)

        self.spinbox_conf_seg = QDoubleSpinBox()
        self.label_conf_seg = QLabel("confThreshold:")

        # Définir la plage de valeurs possibles
        self.spinbox_conf_seg.setRange(0.0, 1.0)
        self.spinbox_conf_seg.setValue(0.5)  # Valeur initiale

        # Définir l'incrément/decrement pour chaque clic
        self.spinbox_conf_seg.setSingleStep(0.1)
        self.spinbox_conf_seg.setDecimals(1)
        self.spinbox_conf_seg.setFixedWidth(100)

        self.options_layout.addWidget(self.label_conf_seg, 2, 0)
        self.options_layout.addWidget(self.spinbox_conf_seg,2, 1)

        self.button_help_ins_seg = QPushButton("Help", self)
        self.button_help_ins_seg.clicked.connect(self.openhelp_ins_segmentation)
        self.options_layout.addWidget(self.button_help_ins_seg, 2, 2)

        self.button_help_detection.setFixedSize(100, 25)


        self.spinbox_size_seg = QSpinBox()
        self.label_size_seg = QLabel("Size:")

        # Définir la plage de valeurs possibles
        self.spinbox_size_seg.setRange(0, 640)
        self.spinbox_size_seg.setValue(320)  # Valeur initiale

        # Définir l'incrément/decrement pour chaque clic
        self.spinbox_size_seg.setSingleStep(1)
        self.spinbox_size_seg.setFixedWidth(100)

        self.options_layout.addWidget(self.label_size_seg, 3, 0)
        self.options_layout.addWidget(self.spinbox_size_seg, 3, 1)

  
        

    def checkbox_textStateChanged(self, state):
        if state != Qt.Checked and not self.checkbox_image.isChecked():
            self.checkbox_text.setCheckState(Qt.Checked)

    def checkbox_imageStateChanged(self, state):
        if state != Qt.Checked and not self.checkbox_text.isChecked():
            self.checkbox_image.setCheckState(Qt.Checked)

    def setupOption(self):  # Réseau disponible dans l'interface
        if self.combobox_detect.currentIndex() == 0:
            self.combobox_option.addItem("YoloV8")

        elif self.combobox_detect.currentIndex() == 1:
            self.combobox_option.addItem("Segnet")
            self.combobox_option.addItem("Rednet")
            self.combobox_option.addItem("YoloV8_seg")
        elif self.combobox_detect.currentIndex() == 2:
            self.combobox_option.addItem("YoloV8_seg")


    def openhelp_segmentation(self):
        # Initialisation de la fênetre Help pour les paramètres
        self.help_window = HelpDialog_segmentation(self)
        self.help_window.show()

    def openhelp_detection(self):
        # Initialisation de la fênetre Help pour les paramètres
        self.help_window = HelpDialog_detection(self)
        self.help_window.show()

    def openhelp_ins_segmentation(self):
        # Initialisation de la fênetre Help pour les paramètres
        self.help_window = HelpDialog_ins_segmentation(self)
        self.help_window.show()

    # Charge le poids et le programme disponible sur le disque de l'utilisateur
    def loadCombobox(self):
        if self.combobox_detect.currentIndex() == 0:
            self.loadCombobox_detection()
        if self.combobox_detect.currentIndex() == 1:
            
            if self.combobox_option.currentIndex() ==2 and not self.checkbox_advance.isChecked():
                self.hideSegmentationButtons()
                self.hideAdvanceSegmentation()
                self.showInstanceSegmentationButtons()
                self.loadCombobox_instance_segmentation()
            elif not self.checkbox_advance.isChecked():
                self.hideInstanceSegmentationButtons()
                self.hideAdvanceInstanceSegmentationButtons()
                self.showSegmentationButtons()
                self.loadCombobox_segmentation()
            elif self.combobox_option.currentIndex() ==2 and self.checkbox_advance.isChecked():
                self.showAdvanceInstanceSegmentationButtons()
                self.hideAdvanceSegmentation()
                self.hideSegmentationButtons()
                self.loadCombobox_instance_segmentation()
            elif self.checkbox_advance.isChecked():
                self.hideAdvanceInstanceSegmentationButtons()
                self.hideInstanceSegmentationButtons()
                self.showAdvanceSegmentation()
                self.loadCombobox_segmentation()

        if self.combobox_detect.currentIndex() == 2:
            self.loadCombobox_instance_segmentation()

    def loadCombobox_segmentation(self):
        if self.combobox_option.count() != 0:
            weights_dir = "Weights_" + self.combobox_option.currentText()
            # Chemin du dossier contenant les poids dispo
            weights_path = os.path.join(os.getcwd(), "Weights", weights_dir)
            program_dir = "Program_" + self.combobox_option.currentText()
            # Chemin du dossier contenant les programme dispo
            program_path = os.path.join(os.getcwd(), "Program", program_dir)

            self.combobox_program.clear()
            self.combobox_weight.clear()

            # Charge le liste des programme dans le combobox
            for file_name in os.listdir(program_path):
                if file_name.endswith(".py"):
                    file_path = os.path.join(program_path, file_name)
                    self.combobox_program.addItem(file_path)

            # Charge le liste des poids dans le combobox
            for file_name in os.listdir(weights_path):
                if file_name.endswith(".pth") or file_name.endswith(".pt"):
                    file_path = os.path.join(weights_path, file_name)
                    self.combobox_weight.addItem(file_path)

    # Charge les éléments par défault de la détection et les models disponible sur le disque
    def loadCombobox_detection(self):
        if self.combobox_option.count() != 0:
            # Dossier des models sur le disque
            model_dir = "Model_" + self.combobox_option.currentText()
            # Path des models sur le disque
            model_path = os.path.join(os.getcwd(), "Model", model_dir)

            self.combobox_model.clear()

            # Initialisation des valeurs par défault des options de la détection

            self.spinbox_conf.setValue(0.5)  # Confiance du model
            self.spinbox_nms.setValue(0.3)  # Confiance du nms
            self.spinbox_size.setValue(320)  # Taille du découpage des images

            for file_name in os.listdir(model_path):
                if file_name.endswith(".pt"):
                    file_path = os.path.join(model_path, file_name)
                    # Ajoute le model dans le combobox
                    self.combobox_model.addItem(file_path)

    def loadCombobox_instance_segmentation(self):
        if self.combobox_option.count() != 0:
            # Dossier des models sur le disque
            model_dir = "Model_" + self.combobox_option.currentText()
            # Path des models sur le disque
            model_path = os.path.join(os.getcwd(), "Model", model_dir)

            self.combobox_model_seg.clear()

            # Initialisation des valeurs par défault des options de la segmentation d'instance

            self.spinbox_conf_seg.setValue(0.5)  # Confiance du model
            self.spinbox_size_seg.setValue(320)  # Taille du découpage des images

            for file_name in os.listdir(model_path):
                if file_name.endswith(".pt"):
                    file_path = os.path.join(model_path, file_name)
                    # Ajoute le model dans le combobox
                    self.combobox_model_seg.addItem(file_path)

    # Ouvre la table des paramètres de la segmentation
    def opentable_segmentation(self):
        tertiary_window = TableWidget_Dialog(self)
        tertiary_window.exec_()

    def openModelFile(self):  # Méthode pour l'ouverture du model (.pt) sur le disque
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Model File", "", "Model File (*.pt)")
        if file_path:
            self.combobox_model.addItem(file_path)
            self.combobox_model.setCurrentText(file_path)
    
    def openModel_Segmentation_File(self):  # Méthode pour l'ouverture du model (.pt) sur le disque
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Model File", "", "Model File (*.pt)")
        if file_path:
            self.combobox_model_seg.addItem(file_path)
            self.combobox_model_seg.setCurrentText(file_path)

    def openProgramFile(self):  # Méthode pour l'ouverture du programme (.py) sur le disque
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Program File", "", "Program File (*.py)")
        if file_path:
            self.combobox_program.addItem(file_path)
            self.combobox_program.setCurrentText(file_path)

    # Méthode pour l'ouverture du fichier des poids (.pth) du model sur le disque
    def openWeightFile(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Weight File", "", "Weight File (*.pth)")
        if file_path:
            self.combobox_weight.addItem(file_path)
            self.combobox_weight.setCurrentText(file_path)

    def openImage(self):  # Méthode pour l'ouverture de l'image / dossier
        if not self.checkbox_open.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.tif)")
            if file_path:
                self.label_open.setText(file_path)
        else:
            folder_path = QFileDialog.getExistingDirectory(
                self, "Open Folder", "")
            if folder_path:
                self.label_open.setText(folder_path)

    def output_file(self):  # Méthode pour la sauvegarde de l'image / dossier
        if not self.checkbox_open.isChecked():
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save File", "", "Images (*.png *.jpg *.jpeg *.tif)")
            if file_path:
                self.label_save.setText(file_path)
        else:
            folder_path = QFileDialog.getExistingDirectory(
                self, "Open Folder", "")
            if folder_path:
                self.label_save.setText(folder_path)

    def run_prog(self):  # Fonction qui lance le programme choisit par l'utilisateur
        if self.combobox_detect.currentIndex() == 0 : # Vérifie que les conditions sont respectées et lance le programme de la détection
            if not self.label_open.text():
                QMessageBox.warning(self, "Error", "File picture empty.")
            elif not self.label_save.text():
                QMessageBox.warning(self, "Error", "File save empty.")
            elif self.combobox_option.count() == 0:
                QMessageBox.warning(self, "Error", "Combobox option empty.")
            elif self.combobox_model.count() == 0:
                QMessageBox.warning(self, "Error", "Combobox model empty.")
            else:
                self.loadvar_detection()

                if os.path.isdir(self.input):

                    self.image_files = self.readdir(self.input)
                    self.output_path = self.output

                    self.progress_window = ProgressDialog(
                        len(self.image_files), self)

                    self.progress_window.show()

                    self.enum = 0

                    for i, path_file in enumerate(self.image_files):
                        self.enum = i
                        self.input = path_file
                        name = os.path.splitext(os.path.basename(path_file))[
                            0] + "_pred" + os.path.splitext(os.path.basename(path_file))[1]
                        self.output = os.path.join(self.output_path, name)
                        self.runYolo_detection()
                        self.progress_window.progress_updated.emit(i + 1)

                else:
                    self.progress_window = ProgressDialog(1, self)

                    self.progress_window.show()
                    self.runYolo_detection()
                    self.progress_window.progress_updated.emit(1)
                self.progress_window.close()

                self.parent().openPicture(self.output)
                self.reject()

        elif self.combobox_detect.currentIndex() == 1 and not self.combobox_option.currentIndex() == 2 :  # Vérifie que les conditions sont respectées et lance le programme de la segmentation

            if not self.label_open.text():
                QMessageBox.warning(self, "Error", "File picture empty.")
            elif not self.label_save.text():
                QMessageBox.warning(self, "Error", "File save empty.")
            elif self.combobox_option.count() == 0:
                QMessageBox.warning(self, "Error", "Combobox option empty.")
            elif self.combobox_program.count() == 0:
                QMessageBox.warning(self, "Error", "Combobox program empty.")
            elif self.combobox_weight.count() == 0:
                QMessageBox.warning(self, "Error", "Combobox weight empty.")
            else:
                self.loadvar_segmentation()

                if os.path.isdir(self.input):  # Si le fichier input est un dossier

                    self.image_files = self.readdir(self.input)
                    self.output_path = self.output

                    self.progress_window = ProgressDialog(  # Créer la fênetre de progression
                        len(self.image_files), self)

                    self.progress_window.show()

                    self.enum = 0

                    # Parcours les images du dossier
                    for i, path_file in enumerate(self.image_files):
                        self.enum = i
                        self.input = path_file
                        name = os.path.splitext(os.path.basename(path_file))[
                            0] + "_pred" + os.path.splitext(os.path.basename(path_file))[1]

                        self.output = os.path.join(self.output_path, name)
                        self.runSegmentation()
                        self.create_trImage()
                        self.superposition_picture()
                        self.progress_window.progress_updated.emit(
                            i + 1)  # Met à jour le nombre d'image restant

                else:  # Si le input est image
                    # Initialisation de la fênetre de progression avec un seule intération
                    self.progress_window = ProgressDialog(1, self)

                    self.progress_window.show()

                    self.runSegmentation()

                    self.create_trImage()

                    self.superposition_picture()

                    self.progress_window.progress_updated.emit(1)

                self.progress_window.close()  # Ferme le fênetre de progression
                self.parent().openPicture(self.output)  # Ouvre l'image dans la scène
                self.reject()  # Ferme la fênetre "run option"

        if self.combobox_detect.currentIndex() == 2 or ( self.combobox_detect.currentIndex() == 1 and self.combobox_option.currentIndex() == 2 ) : # Vérifie que les conditions sont respectées et lance le programme de la segmentation d'instance
            if not self.label_open.text():
                QMessageBox.warning(self, "Error", "File picture empty.")
            elif not self.label_save.text():
                QMessageBox.warning(self, "Error", "File save empty.")
            elif self.combobox_option.count() == 0:
                QMessageBox.warning(self, "Error", "Combobox option empty.")
            elif self.combobox_model_seg.count() == 0:
                QMessageBox.warning(self, "Error", "Combobox model empty.")
            else:
                self.loadvar_instance_segmentation()
                self.semantic_seg = False
                if self.combobox_detect.currentIndex() == 1:
                    self.semantic_seg = True

                if os.path.isdir(self.input):  # Si le fichier input est un dossier

                    self.image_files = self.readdir(self.input)
                    self.output_path = self.output

                    self.progress_window = ProgressDialog(  # Créer la fênetre de progression
                        len(self.image_files), self)

                    self.progress_window.show()

                    self.enum = 0

                    # Parcours les images du dossier
                    for i, path_file in enumerate(self.image_files):
                        self.enum = i
                        self.input = path_file
                        name = os.path.splitext(os.path.basename(path_file))[
                            0] + "_pred" + os.path.splitext(os.path.basename(path_file))[1]

                        self.output = os.path.join(self.output_path, name)
                        self.runYolo_segmentation()
                        
                        self.progress_window.progress_updated.emit(
                            i + 1)  # Met à jour le nombre d'image restant
                else:

                    self.progress_window = ProgressDialog(1, self)

                    self.progress_window.show()
                    self.runYolo_segmentation()

                    self.progress_window.progress_updated.emit(1)

                self.progress_window.close()  # Ferme le fênetre de progression
                self.parent().openPicture(self.output)  # Ouvre l'image dans la scène
                self.reject()  # Ferme la fênetre "run option"


    def readdir(self, path_dir):
        image_extensions = [".png", ".jpg", ".jpeg", ".tif"]
        image_files = []

        # Parcours les éléments du dossier
        for file_name in os.listdir(path_dir):
            file_path = os.path.join(path_dir, file_name)
            if os.path.isfile(file_path) and any(file_name.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file_path)

        return image_files  # Liste contenant l'ensembles des chemins vers les images

    # Extrait les variables de l'interface utilisateur
    def loadvar_segmentation(self):
        self.input = self.label_open.text()
        self.output = self.label_save.text()
        self.program = self.combobox_program.currentText()
        self.weight = self.combobox_weight.currentText()
        self.readtable()
        self.option = self.property_option

    def loadvar_detection(self):  # Extrait les variables de l'interface utilisateur
        self.input = self.label_open.text()
        self.output = self.label_save.text()
        self.model_d = self.combobox_model.currentText()
        self.size_ = self.spinbox_size.value()
        self.confThreshold = self.spinbox_conf.value()
        self.nmsThreshold = self.spinbox_nms.value()

    def loadvar_instance_segmentation(self):
        self.input = self.label_open.text()
        self.output = self.label_save.text()
        self.model_d = self.combobox_model_seg.currentText()
        self.size_ = self.spinbox_size_seg.value()
        self.confThreshold = self.spinbox_conf_seg.value()

    def runYolo_detection(self):
        self.parent().list_widget.clear()

        self.class_data = {}  # Initialisation du dico
        image = cv2.imread(self.input)  # Read du l'image

        if not self.checkbox_advance.isChecked():  # Programme par défault de Yolo

            self.thread3 = Thread_yolo_det_default(self.model_d, self.input, self.confThreshold)
            # Signal qui fait appel à la méthode handleResults
            self.thread3.resultsReady.connect(self.handleResults)
            self.thread3.start()

            loop = QEventLoop()
            # Connect le signal finished au quit du QEventLoop
            self.thread3.finished.connect(loop.quit)
            loop.exec_()

            # Met à jour la barre de progression à la fin du thread
            self.progress_window.progress_bar.setValue(100)

        else:

            self.thread2 = Thread_yolo_det(
                self.size_, self.model_d, self.input, self.confThreshold, self.nmsThreshold)
            self.thread2.progressChanged.connect(
                self.progress_window.progressbar_updated)
            self.thread2.resultsReady.connect(self.handleResults)

            self.thread2.start()

            loop = QEventLoop()
            # Connecter le signal finished au quit du QEventLoop
            self.thread2.finished.connect(loop.quit)
            loop.exec_()

        # Trier les classes par ordre alphabétique
        self.sorted_classes = sorted(self.class_data.keys())

        # Initialise la variable du chemin vers le fichier txt
        parent_dir = os.path.dirname(self.output)
        name_output = os.path.splitext(
            os.path.basename(self.output))[0] + ".txt"
        self.path_output_txt = os.path.join(parent_dir, name_output)

        # Ajouter les éléments triés au list widget
        if self.checkbox_text.isChecked():  # Si l'option save .txt est cochée
            with open(self.path_output_txt, 'w') as file:  # Créer le fichier txt
                for class_name in self.sorted_classes:
                    for x1, y1, x2, y2, confidence, class_id, x, y, w, h in self.class_data[class_name]:
                        line = f"{class_id} {x} {y} {w} {h} {confidence:.2f}\n"
                        print(line)
                        file.write(line)
                        label = f"{class_name} ({confidence:.2f})"
                        if self.checkbox_image.isChecked():  # Si l'option save image est cochée
                            # Calculer les dimensions du texte
                            self.plot_one_box([x1, y1, x2, y2], image,label)
                            
        else:  # Si l'option save image est cochée
            for class_name in self.sorted_classes:
                for x1, y1, x2, y2, confidence, class_id, x, y, w, h in self.class_data[class_name]:

                    label = f"{class_name} ({confidence:.2f})"
                    
                    self.plot_one_box([x1, y1, x2, y2], image,label)

        if self.checkbox_image.isChecked():
            cv2.imwrite(self.output, image)  # Créer l'image sur le disque

    def plot_one_box(self, boxe , img, label=None):
        # Plots one bounding box on image img
        font_scale = 0.6
        font_thickness = 1
        color = (0, 0, 255)
        c1, c2 = (int(boxe[0]), int(boxe[1])), (int(boxe[2]), int(boxe[3]))
        cv2.rectangle(img, c1, c2, color, 1, lineType=cv2.LINE_AA)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Vérifier si le rectangle rempli dépasse l'image d'origine
        if c2[0] + t_size[0] > img.shape[1]:
            c2 = (img.shape[1] - t_size[0], c2[1])
        if c1[1] - t_size[1] - 3 < 0:
            c1 = (c1[0], t_size[1] + 3)
        
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, [225, 255, 255], font_thickness, lineType=cv2.LINE_AA)
    
    # Récupére les dico qui est constitués d'informations sur les classes détéctées
    def handleResults(self, class_data):
        self.class_data = class_data

    def readtable(self):
        # Définir le chemin d'accès au fichier XML
        self.file_path = os.path.join(os.getcwd(), "Data_table", "data.xml")

        # Initialisation des variables
        self.property_option = ""  # Option de propriété
        i = 0  # Variable de compteur

        if self.file_path:
            doc = QDomDocument()

            # Charger le fichier XML existant
            file = QFile(self.file_path)
            if file.open(QIODevice.ReadOnly):
                if doc.setContent(file):
                    file.close()
                else:
                    print("Failed to load XML file:", file.errorString())
                    return
            else:
                print("Failed to open XML file:", file.errorString())
                return

            # Lire les données XML
            root_element = doc.documentElement()
            data_elements = root_element.elementsByTagName("Row")

            for row in range(data_elements.count()):
                data_element = data_elements.at(row)
                keep_element = data_element.firstChildElement("Keep")
                property_element = data_element.firstChildElement("Property")
                value_element = data_element.firstChildElement("Value")

                if property_element and value_element and keep_element:
                    property_text = property_element.text()
                    value_text = value_element.text()
                    keep_text = keep_element.text()

                    if keep_text == "Checked":  # Sélectionne dans les éléments qui ont été sélectionnés
                        if i == 0:  # Gérer l'espacement entre les élements
                            # Ajoute "--"" devant chaque paramètre (-- nor )
                            property_text = "--" + property_text
                            i = 1
                        else:
                            property_text = " --" + property_text

                        if value_text != "None":
                            # Chaine de caractères qui constitue l'ensemble des options choisie par l'utilisateur
                            self.property_option += property_text + " " + value_text
                        else:
                            self.property_option += property_text

        else:
            print("File path is not valid:", self.file_path)

    def runSegmentation(self):
        # Commande des options avancée de la segmentation
        if self.checkbox_advance.isChecked():
            if self.option != "":
                command = [
                    "python",
                    self.program,
                    "--input",
                    self.input,
                    "--output",
                    self.output,
                    "--weights",
                    self.weight
                ]
                options = self.option.split()  # Divise la chaîne des options en une liste

                command.extend(options)  # Ajoute les options à la commande

            else:
                command = [
                    "python",
                    self.program,
                    "--input",
                    self.input,
                    "--output",
                    self.output,
                    "--weights",
                    self.weight
                ]
        else:  # Commande des options par défault de la segmentation
            if self.combobox_option.currentText() == "Segnet":
                command = [
                    "python",
                    "Program/Program_Segnet/applySegnetRGB.py",
                    "--input",
                    self.input,
                    "--output",
                    self.output,
                    "--weights",
                    self.weight,
                    "--nor"

                ]

            elif self.combobox_option.currentText() == "Rednet":
                command = [
                    "python",
                    "Program/Program_Rednet/applyRedNetRGB.py",
                    "--input",
                    self.input,
                    "--output",
                    self.output,
                    "--weights",
                    self.weight,
                    "--model",
                    "rednet4B",
                    "--threshold",
                    "0.25"
                ]

        print(command)

        self.thread1 = Thread_Segmentation(command)
        # Connect le signal a la barre de progression
        self.thread1.progressChanged.connect(
            self.progress_window.progressbar_updated)

        loop = QEventLoop()
        # Connect le signal finish à l'arrêt de la boucle
        self.thread1.finished.connect(loop.quit)

        self.thread1.start()
        loop.exec_()

    def create_trImage(self):
        img = cv2.imread(self.output)

        # ajoute le canal alpha
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        # selectionne la partie de l'image non rouge
        filtre = img[:, :, 2] < 128  # 2 le rouge BGR

        rgba[filtre, 3] = 0  # Met en transparent la partie de l'image non rouge

        cv2.imwrite(self.output, rgba)  # ecris l'image sur le disque

    def superposition_picture(self):

        # Chargement des images
        image_normale = cv2.imread(self.input)
        image_transparente = cv2.imread(self.output)

        # Vérification de la transparence de l'image transparente
        if image_transparente.shape[2] == 4:
            # Extraction du canal alpha de l'image transparente
            alpha_channel = image_transparente[:, :, 3]
            alpha_channel = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2BGR)

            # Masquage de l'image normale avec le canal alpha
            image_normale = cv2.bitwise_and(image_normale, alpha_channel)

        # Superposition des deux images
        result = cv2.add(image_transparente[:, :, :3], image_normale)

        cv2.imwrite(self.output, result)  # ecris l'image sur le disque

    def runYolo_segmentation(self):

        if not self.checkbox_advance.isChecked():  # Programme par défault de la segmentation de Yolo 

            self.thread3 = Thread_yolo_seg_default(self.model_d, self.input,self.output, self.confThreshold, self.semantic_seg)

            self.thread3.start()

            loop = QEventLoop()
            # Connect le signal finished au quit du QEventLoop
            self.thread3.finished.connect(loop.quit)
            loop.exec_()

            # Met à jour la barre de progression à la fin du thread
            self.progress_window.progress_bar.setValue(100)
        else: 
            self.thread_= Thread_yolo_seg(
                self.size_, self.model_d, self.input, self.output, self.confThreshold, self.semantic_seg)
            self.thread_.progressChanged.connect(
                self.progress_window.progressbar_updated)

            self.thread_.start()

            loop = QEventLoop()
            # Connecter le signal finished au quit du QEventLoop
            self.thread_.finished.connect(loop.quit)
            loop.exec_()