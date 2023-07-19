from PyQt5.QtCore import pyqtSignal ,QThread
from subprocess import Popen, PIPE
import subprocess


class Thread_Segmentation(QThread): #Thread pour le réseau Segnet / Rednet de la segmentation
    progressChanged = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, command):
        super().__init__()
        self.command = command
        self.progress = 0
        self.total_steps = 0
        self.current_step = 0

    def run(self):
        try: 
            with Popen(self.command, stdout=PIPE, bufsize=1, universal_newlines=True) as p: #Execute la commande 
                for line in p.stdout: #Asynchrone pour récup les sortie de programme python pour update la barre de progression
                    if line.startswith("NB"): 
                        total_steps = int(line.split()[-1])
                    elif line.startswith("I"):
                        current_step = int(line.split()[-1])
                        progress = int(current_step / total_steps * 100)
                        self.progressChanged.emit(progress)

                self.finished.emit()

        except subprocess.CalledProcessError as e:
            print("Subprocess error:", e)