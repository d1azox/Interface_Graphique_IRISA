from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QAction
from PyQt5.QtGui import  QIcon
from widgets.CloseableTabWidget import CloseableTabWidget
import os
import sys
import webbrowser
ci_build_and_not_headless = False
try:
    from cv2.version import ci_build, headless
    ci_and_not_headless = ci_build and not headless
except:
    pass
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_FONTDIR")

class MyApp(QApplication):
    

    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.main_window = QMainWindow()

        self.main_window.resize(1280, 1024)  # Taille de la fênetre
        self.main_window.setWindowTitle("Interface")

        self._createActions()
        self._createToolBars()
        self._createMenuBar()

        # Instance de CloseableTabWidget pour la gestions des onglets
        self.tab_widget = CloseableTabWidget(self)

        # Ajoutez un premier onglet
        self.tab_widget.add_new_tab()

        # Connecter les actions aux outils
        self.createConnectActions()

        # Ajouter le CloseableTabWidget à la fenêtre 
        self.main_window.setCentralWidget(self.tab_widget)
        self.main_window.show()

    def _createMenuBar(self):

        menuBar = QMenuBar(self.main_window)
        self.main_window.setMenuBar(menuBar)

        # File menu
        fileMenu = menuBar.addMenu("&File")
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.opendirAction)
        fileMenu.addAction(self.saveAction)

        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)

        # Edit menu
        editMenu = menuBar.addMenu("&Edit")
        editMenu.addAction(self.copyAction)
        editMenu.addAction(self.pasteAction)
        editMenu.addAction(self.cutAction)
        editMenu.addSeparator()
        editMenu.addAction(self.deleteAction)
        editMenu.addAction(self.zoominAction)
        editMenu.addAction(self.zoomoutAction)
        editMenu.addAction(self.cropAction)
        editMenu.addAction(self.cropzoomAction)
        editMenu.addAction(self.addsceneAction)
        editMenu.addAction(self.runAction)

        # Help menu
        self.helpMenu = menuBar.addMenu("&Help")

        # Création d'un sous-menu pour le menu "Help"
        self.helpMenu.addAction(self.helpContentAction)
        self.helpMenu.addAction(self.aboutAction)

    def _createToolBars(self):
        # File tool
        fileToolBar = self.main_window.addToolBar("File")
        fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.opendirAction)
        fileToolBar.addAction(self.saveAction)

        fileToolBar.setMovable(False)

        # Edit tool
        editToolBar = self.main_window.addToolBar("Edit")
        editToolBar.addAction(self.copyAction)
        editToolBar.addAction(self.pasteAction)
        editToolBar.addAction(self.cutAction)
        editToolBar.addAction(self.deleteAction)
        editToolBar.addAction(self.zoominAction)
        editToolBar.addAction(self.zoomoutAction)
        editToolBar.addAction(self.cropAction)
        editToolBar.addAction(self.cropzoomAction)
        editToolBar.addAction(self.addsceneAction)
        editToolBar.addAction(self.terminalAction)
        editToolBar.addAction(self.runAction)

    def _createActions(self):

        # Action File
        self.newAction = QAction(QIcon("Icon/new-document.png"), "&New", self)
        self.openAction = QAction(QIcon("Icon/folder.png"), "&Open...", self)
        self.opendirAction = QAction(
            QIcon("Icon/folderdir.png"), "&Open dir", self)
        self.saveAction = QAction(QIcon("Icon/diskette.png"), "&Save", self)
        self.exitAction = QAction("&Exit", self)

        # Action Edit
        self.copyAction = QAction(QIcon("Icon/copy.png"), "&Copy", self)
        self.pasteAction = QAction(QIcon("Icon/paste.png"), "&Paste", self)
        self.cutAction = QAction(QIcon("Icon/scissors.png"), "C&ut", self)
        self.deleteAction = QAction(QIcon("Icon/x.png"), "Delete", self)
        self.cropAction = QAction(QIcon("Icon/crop.png"), "Crop", self)
        self.cropzoomAction = QAction(
            QIcon("Icon/crop_zoom.png"), "CropZoom", self)
        self.openTree_viewAction = QAction(
            QIcon("Icon/folder.png"), "&Open...", self)
        self.renameAction = QAction("&Rename...", self)
        self.addsceneAction = QAction(
            QIcon("Icon/benchmarking.png"), "&Add_scene...", self)
        self.terminalAction = QAction(
            QIcon("Icon/terminal.png"), "&Terminal...", self)

        self.zoominAction = QAction(
            QIcon("Icon/magnifying-glass-m.png"), "Zoom +", self)
        self.zoomoutAction = QAction(
            QIcon("Icon/magnifying-glass.png"), "Zoom -", self)
        self.runAction = QAction(QIcon("Icon/run.png"), " Run -", self)

        # Action Help
        self.helpContentAction = QAction("&Help Content", self)
        self.aboutAction = QAction("&About", self)

        # Shortcut File
        self.newAction.setShortcut("Ctrl+N")
        self.saveAction.setShortcut("Ctrl+S")
        self.openAction.setShortcut("Ctrl+O")
        self.opendirAction.setShortcut("Ctrl+D")

        # Shortcut Edit
        self.zoominAction.setShortcut("Ctrl++")
        self.zoomoutAction.setShortcut("Ctrl+-")
        self.copyAction.setShortcut("Ctrl+C")
        self.pasteAction.setShortcut("Ctrl+V")
        self.cropAction.setShortcut("Shift+C")
        self.cropzoomAction.setShortcut("Shift+Z")

        self.cutAction.setShortcut("Ctrl+x")
        self.deleteAction.setShortcut("Ctrl+w")
        self.runAction.setShortcut("Ctrl+f")
        self.terminalAction.setShortcut("Ctrl+t")
        self.addsceneAction.setShortcut("Ctrl+p")
        self.renameAction.setShortcut("Ctrl+r")

    def createConnectActions(self):
        # Connect File
        self.newAction.triggered.connect(self.handle_tab_action)
        self.saveAction.triggered.connect(self.handle_tab_action)
        self.openAction.triggered.connect(self.handle_tab_action)
        self.exitAction.triggered.connect(self.handle_tab_action)

        # Connect Edit
        self.opendirAction.triggered.connect(self.handle_tab_action)
        self.zoominAction.triggered.connect(self.handle_tab_action)
        self.zoomoutAction.triggered.connect(self.handle_tab_action)
        self.copyAction.triggered.connect(self.handle_tab_action)
        self.pasteAction.triggered.connect(self.handle_tab_action)
        self.cutAction.triggered.connect(self.handle_tab_action)
        self.deleteAction.triggered.connect(self.handle_tab_action)
        self.cropAction.triggered.connect(self.handle_tab_action)
        self.cropzoomAction.triggered.connect(self.handle_tab_action)
        self.openTree_viewAction.triggered.connect(self.handle_tab_action)
        self.renameAction.triggered.connect(self.handle_tab_action)
        self.addsceneAction.triggered.connect(self.handle_tab_action)
        self.terminalAction.triggered.connect(self.handle_tab_action)
        self.runAction.triggered.connect(self.handle_tab_action)

        # Connect Help
        self.helpContentAction.triggered.connect(self.handle_tab_action)
        self.aboutAction.triggered.connect(self.handle_tab_action)

    def handle_tab_action(self):

        # Récupére les méthodes dans l'interface
        active_interface = self.tab_widget.currentWidget()
        action = self.sender()

        action_mapping = {
            self.newAction: self.tab_widget.add_new_tab,
            self.saveAction: active_interface.saveImage,
            self.openAction: active_interface.open_tool,
            self.exitAction: self.exit,
            self.opendirAction: active_interface.tree_view.loadDir,
            self.zoominAction: active_interface.apply_zoom_in_action_to_all_scenes,
            self.zoomoutAction: active_interface.apply_zoom_out_action_to_all_scenes,
            self.copyAction: active_interface.tree_view.copySelectedItem,
            self.pasteAction: active_interface.tree_view.pasteClipboardData,
            self.cutAction: active_interface.tree_view.cut,
            self.deleteAction: active_interface.tree_view.delete,
            self.cropAction: active_interface.apply_crop_action_to_all_scenes,
            self.cropzoomAction: active_interface.apply_crop_zoom_action_to_all_scenes,
            self.openTree_viewAction: active_interface.tree_view.openTree_view,
            self.renameAction: active_interface.tree_view.renameSelectedItem,
            self.addsceneAction: active_interface.addScene,
            self.terminalAction: active_interface.textEdit.openTerminalWindow,
            self.runAction: active_interface.run,
            self.helpContentAction: self.openHelpContent,
            self.aboutAction: self.openGitHubWebPage
        }

        if action in action_mapping:
            action_mapping[action]()

    def openHelpContent(self):
        # Ouvrir le fichier PDF du manuel d'utilisation
        webbrowser.open("Help/Manuelle_utilisation.pdf")
    def openGitHubWebPage(self):
        # Ouvrir la page web de la documentation
        webbrowser.open("https://github.com/d1azox/Interface_Graphique_IRISA")

# Lance l'application
if __name__ == '__main__':
    import sys
    app = MyApp(sys.argv)
    sys.exit(app.exec_())