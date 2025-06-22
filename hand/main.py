import sys
import os
import signal
import logging
import argparse
from .core import HandModule
from .comms import AsyncModuleServer, make_message
from symbols import SYMBOLS

HOST = '127.0.0.1'
PORT = 5002

# Set up logging to both terminal and hand.log (overwrite on each run)
logger = logging.getLogger("HandModule")
logger.setLevel(logging.INFO)
log_path = os.path.join(os.path.dirname(__file__), '../hand.log')
file_handler = logging.FileHandler(log_path, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.handlers = [file_handler, stream_handler]

def signal_handler(sig, frame):
    logger.info('Exiting HandModule...')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def main():
    parser = argparse.ArgumentParser(description="HandModule: Production-grade symbol drawing with adaptive learning and more.")
    parser.add_argument('--lr', type=float, default=0.5, help='Initial learning rate')
    args = parser.parse_args()
    hm = HandModule(lr=args.lr)
    # For socket/IPC use, call hm.run()
    # For standalone training (example):
    # X, y = ... # Load your dataset here as list/array of images and target indices
    # scheduler = LRScheduler(args.lr) if args.scheduler else None
    # hm.train_with_scheduler(X, y, ...)
    #
    # For now, just run the server loop (migrate run logic here if needed)
    print("[HandModule] Ready. Implement run logic or server loop here.")

if __name__ == '__main__':
    main()
