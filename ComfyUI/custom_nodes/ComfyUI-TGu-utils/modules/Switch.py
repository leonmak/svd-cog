class MPNSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "active": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF", "forceInput": False}),
                    "model1": ("MODEL",),
                    "positive1": ("CONDITIONING",),
                    "negative1": ("CONDITIONING",),
                    "model2": ("MODEL",),
                    "positive2": ("CONDITIONING",),
                    "negative2": ("CONDITIONING",),
                    },
                "hidden": {"uid": "UNIQUE_ID"},
                }
    

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "doit"

    CATEGORY = "TGu_util"

    def doit(self, active, model1, model2, positive1, positive2, negative1, negative2, uid):
        if active:
            return (model2, positive2, negative2)
        else:
            return (model1, positive1, negative1)
        
class MPNReroute:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    },
                "hidden": {"uid": "UNIQUE_ID"},
                }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "doit"

    CATEGORY = "TGu_util"
    def doit(self, model, positive, negative, uid):
        return (model, positive, negative)
    
class PNSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "active": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF", "forceInput": False}),
                    "positive1": ("CONDITIONING",),
                    "negative1": ("CONDITIONING",),
                    "positive2": ("CONDITIONING",),
                    "negative2": ("CONDITIONING",),
                    },
                "hidden": {"uid": "UNIQUE_ID"},
                }
    

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "doit"

    CATEGORY = "TGu_util"

    def doit(self, active, positive1, positive2, negative1, negative2, uid):
        if active:
            return (positive2, negative2)
        else:
            return (positive1, negative1)