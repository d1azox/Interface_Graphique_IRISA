from PyQt5.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout,QApplication
from PyQt5.QtCore import pyqtSignal 


class ProgressDialog(QDialog):
    # Signal émis lorsque la valeur de progression est mise à jour
    progress_updated = pyqtSignal(int)
    # Signal émis lorsque la valeur de la barre de progression est mise à jour
    progressbar_updated = pyqtSignal(int)

    def __init__(self, total_iterations, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Progression")
        self.setFixedSize(400, 200)

        self.current_iteration = 0
        self.total_iterations = total_iterations

        # Étiquette pour afficher l'état de progression
        self.label = QLabel("0/{}".format(total_iterations), self)
        # Barre de progression pour visualiser la progression
        self.progress_bar = QProgressBar(self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)

        # Connecter le signal à la méthode update_progress
        self.progress_updated.connect(self.update_progress)
        # Connecter le signal à la méthode update_progressbar
        self.progressbar_updated.connect(self.update_progressbar)

    def update_progress(self, value):
        self.current_iteration = value
        # Met à jour l'étiquette de progression
        self.label.setText(
            "{}/{}".format(self.current_iteration, self.total_iterations))
        QApplication.processEvents()  # Update l'interface utilisateur

    def update_progressbar(self, value):
        self.current_iteration = value
        # Met à jour la barre de progression
        self.progress_bar.setValue(self.current_iteration)
        QApplication.processEvents()  # Update l'interface utilisateur