# Script written by Sietse Dijt for Vlijm group RUG
# 2024-01-07

class Gating:
    active:bool  = None
    delay:float = None # s
    width:float  = None # s

    def __init__(self, active=None, delay=None, width=None):
        self.active = active
        self.delay = delay
        self.width = width

class ImageSettings:
    # Uninitialized settings are None, initialization is done by other classes (data members directly accessed)

    microscope:str  = None
    objective:str   = None

    config_name:str = None

    pixel_sizes:dict = None # Like ImagePlus scales

    stage_x, stage_y, stage_z = None, None, None # m (float)
    autofocus_on:bool = None

    offset_x, offset_y, offset_z = None, None, None # m (float)
    rotation, roll, tilt = None, None, None # degrees (float)

    bidirectional_scan, fast_scan = None, None # bools

    dwelltime:float = None # s
    linesteps, line_repetitions, line_interleave = None, None, None # list(#steps:int), int, bool
    pixelsteps = None
    
    channels:list = None # See add_channel function
    
    scan_axes = None # e.g. XY, order of scan
    pinhole:float = None # AU
    split_3D = None # probably fraction

    # TODO: RESCUE, DyMIN
    def __init__(self):
        self.channels = []

    def channel_names(self):
        return [ch['name'] for ch in self.channels]
    
    def add_channel(self, name:str, detector:str, spectral_min:int, spectral_max:int, lasers:dict, gating:Gating, linesteps:list, pixelsteps:list):
        self.channels.append({
            'name': name,
            'detector': detector,
            'spectral_min': spectral_min,
            'spectral_max': spectral_max,
            'lasers': lasers, # dict(name: power%), names: ['Exc 405', 'Exc 488', 'Exc 561', 'Exc 640', 'STED 775']
            'gating': gating,
            'linesteps': linesteps,     # list of step nrs (int) in which the channel is active 
            'pixelsteps': pixelsteps,   # list of step nrs (int) in which the channel is active 
        })

    # Also functions to compare ImageSetting's with eachother
    # And to show an overview of the image settings