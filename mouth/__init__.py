# MouthModule (Digital Mouth)
# Synthesizes audio waveforms for each symbol using learnable parameters
# Runs as an independent process, communicates via sockets/IPC

from .core import MouthModule
from .articulator import DigitalArticulator
from .utils import xavier_init, LRScheduler
from .comms import AsyncModuleServer, AsyncModuleClient, make_message

