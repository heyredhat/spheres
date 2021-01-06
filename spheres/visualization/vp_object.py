import vpython as vp

class VObject:
    """
    Parent class for visual objects (using vpython), streamlining the automatic
    updating of visuals when attributes are changed, mouse interaction, and keeping track
    of sets of visual objects which might be toggled on or off.

    """
    def __init__(self, scene=None):
        super().__setattr__("auto_refresh_attrs", {})
        super().__setattr__("toggles", {})
        self.scene = scene if scene else vp.scene
        
        self.vchildren = []
        self.toggles = {}
        self.refreshments = []
        self.refreshing = False

        self.mousedown_callbacks = {}
        self.mousemove_callbacks = []
        self.mouseup_callbacks = []
        self.scene.bind("mousedown", self.mousedown)
        self.scene.bind("mousemove", self.mousemove)
        self.scene.bind("mouseup", self.mouseup)

    def __setattr__(self, name, value):
        if name in self.auto_refresh_attrs:
            super().__setattr__(name, value)
            self.auto_refresh_attrs[name]()
        elif name in self.toggles:
            self.toggle(name, value)
        super().__setattr__(name, value)

    def toggle(self, name, value=None):
        super().__setattr__(name, (True if not hasattr(self, name) or not getattr(self, name) else False) if value == None else value)
        if getattr(self, name) and not self.toggles[name]["exists"]:
            self.toggles["vchildren"] = self.toggles[name]["create"]()
            self.vchildren.extend(self.toggles["vchildren"])
            self.toggles[name]["exists"] = True
        elif not getattr(self, name) and self.toggles[name]["exists"]:
            self.destroy_vchildren(self.toggles["vchildren"])
            self.toggles["vchildren"] = []
            self.toggles[name]["exists"] = False
    
    def add_toggle(self, name, create):
        self.toggles[name] = {"exists": False, "create": create, "vchildren": []}

    def refresh(self):
        self.refreshing = True
        [refreshment() for refreshment in self.refreshments]
        self.refreshing = False

    def mousedown(self):
        pick = self.scene.mouse.pick
        if pick in self.vchildren:
            if pick in self.mousedown_callbacks:
                self.mousedown_callbacks[pick]()

    def mousemove(self):
        [mousemove_callback() for mousemove_callback in self.mousemove_callbacks]
            
    def mouseup(self):
        [mouseup_callback() for mouseup_callback in self.mouseup_callbacks]
            
    def destroy_vchildren(self, vchildren):
        if type(vchildren) != list:
            vchildren = [vchildren]
        for vchild in vchildren:
            self.vchildren.remove(vchild)
            vchild.visible = False
            del vchild

    def destroy(self):
        for vchild in self.vchildren:
            vchild.visible = False
            del vchild
        self.scene.bind("mousedown", self.mousedown)
        self.scene.bind("mousemove", self.mousemove)
        self.scene.bind("mouseup", self.mouseup)


