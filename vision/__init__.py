# VisionModule (Eye)
# Recognizes symbols from images using Delta Rule/Rescorla-Wagner learning
# Runs as an independent process, communicates via sockets/IPC

from .core import VisionModule
from .retina import DigitalRetina
from .utils import xavier_init, LRScheduler
from .comms import AsyncModuleServer, AsyncModuleClient, make_message

