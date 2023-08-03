from PyQt5.QtWidgets import QTabWidget, QApplication

from windows.MainWindow import MainWindow

class CloseableTabWidget(QTabWidget):
    

    def __init__(self, app_instance, parent=None):
        super().__init__(parent)
        self.app_instance = app_instance

        # Connection du signal pour fermer l'onglet
        self.tabCloseRequested.connect(self.close_tab)
        self.tab_counter = 1

    def close_tab(self, index):
        # Index du widget
        widget = self.widget(index)

        # Supprime l'index du tableau et le widget
        self.removeTab(index)
        widget.deleteLater()

        # Renomme les onglets suivants pour maintenir la numérotation cohérente
        for i in range(index, self.count()):
            new_tab_name = f"Window {i + 1}"
            self.setTabText(i, new_tab_name)

        # Mettre à jour le compteur d'onglets si l'onglet supprimé a le même numéro que le compteur
        if index + 1 == self.tab_counter:
            self.tab_counter -= 1

        # Sélectionner l'onglet suivant s'il en reste
        if self.count() > 0:
            next_index = min(index, self.count() - 1)
            self.setCurrentIndex(next_index)

            # Si y'a plus d'onglet restant on exit l'application
        if self.count() == 0:
            QApplication.quit()

    def add_new_tab(self):
        # Créez une nouvelle instance de votre interface
        new_interface = MainWindow(self.app_instance)

        # Vérifiez s'il y a des onglets supprimés dont le numéro peut être réutilisé
        if self.tab_counter <= self.count():
            tab_number = self.tab_counter
        else:
            tab_number = self.count() + 1

        # Ajoutez le nouvel onglet avec le numéro approprié
        self.addTab(new_interface, f"Window {tab_number}")

        # Sélectionnez le nouvel onglet
        self.setCurrentIndex(self.count() - 1)
        self.setTabsClosable(True)
        self.setMovable(True)

        # Incrémentez le compteur d'onglets
        self.tab_counter += 1