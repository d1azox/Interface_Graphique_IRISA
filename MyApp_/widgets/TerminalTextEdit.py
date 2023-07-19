from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import QObject,  pyqtSignal
from PyQt5.QtGui import  QFont, QTextCursor
import sys

class EmittingStream(QObject): #Emet un signal quand la méthode write est écris

    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

class TerminalTextEdit(QTextEdit): #Zone de texte du terminal
    def __init__(self, parent=None):
        super().__init__(parent)
        self.emit_stream = EmittingStream(textWritten=self.normalOutputWritten) #Connect le signal a la méthode
        self.setReadOnly(True)
        self.document().setDefaultFont(QFont("Courier New", 10)) #Mise en forme du texte
        self.hide() #Cache le terminal par défault

    def __del__(self):
            # Restaurer sys.stdout
            sys.stdout = sys.__stdout__

    def normalOutputWritten(self, text): #Permet d'afficher du texte en QTextEdit
        cursor = self.textCursor()  # Récupérer le curseur du widget QTextEdit
        cursor.movePosition(QTextCursor.End)  # Déplacer le curseur à la fin du texte existant
        cursor.insertText(text)  # Insérer le texte fourni à la position du curseur
        self.setTextCursor(cursor)  # Mettre à jour le curseur du widget QTextEdit
        self.ensureCursorVisible()  # S'assurer que le curseur est visible dans le widget QTextEdit

    def openTerminalWindow(self): #Gestion du terminal de l'interface
        if self.isVisible():
            self.hide()
        else:
            self.show()