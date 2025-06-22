# EarModule (Digital Ear)
# Listens to real audio samples, extracts features, learns audio-symbol associations
# Runs as an independent process, communicates via sockets/IPC

from .core import EarModule
from .cochlea import DigitalCochlea
from .utils import xavier_init, LRScheduler
from .comms import AsyncModuleServer, AsyncModuleClient, make_message

