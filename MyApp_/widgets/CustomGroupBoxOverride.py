from widgets.CustomGroupBox import CustomGroupBox

class CustomGroupBoxOverride(CustomGroupBox): #Conteneur pour la scène principale (enfant)

    def __init__(self, app_instance, parent=None):
        super().__init__(app_instance,parent)
        self.show() #Affiche par défault

        self.parent_=parent

    def on_close_button_clicked(self): #Méthode pour le bouton 
        self.graphics_view.scene_.clear()
        self.label.clear()
        self.parent_.list_widget.clear()
        self.parent_.list_widget.setVisible(False)
        self.parent_.graphicsView.current_scene_path = None