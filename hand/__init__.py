# HandModule (Digital Hand)
# Learns to write/draw symbols on a digital canvas
# Runs as an independent process, communicates via sockets/IPC

from .core import HandModule
from .drawing import DigitalCanvas, DigitalPen
from .utils import xavier_init, LRScheduler, check_similarity
from .comms import AsyncModuleServer, AsyncModuleClient, make_message

