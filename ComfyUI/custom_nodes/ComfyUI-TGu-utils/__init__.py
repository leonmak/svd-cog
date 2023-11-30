import folder_paths
import os
import sys

comfy_path = os.path.dirname(folder_paths.__file__)
modules_path = os.path.join(os.path.dirname(__file__), "modules")

sys.path.append(modules_path)

from Switch import *

NODE_CLASS_MAPPINGS = {
    "MPNSwitch": MPNSwitch,
    "MPNReroute": MPNReroute,
    "PNSwitch": PNSwitch,
}
        
NODE_DISPLAY_NAME_MAPPINGS = {
    "MPNSwitch": "MPN Switch",
    "MPNReroute": "MPN Reroute",
    "PNSwitch": "PN Switch",
}