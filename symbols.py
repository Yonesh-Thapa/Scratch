import string
import os

# Canonical symbol set: 0-9, A-Z, a-z, and common symbols
ALL_SYMBOLS = list(string.digits + string.ascii_uppercase + string.ascii_lowercase + "!@#$%^&*()-_=+[]{};:'\",.<>/?|\\`~")

# Optionally, filter for available data (images/audio) if needed
# By default, export ALL_SYMBOLS for use in all modules and test scripts
SYMBOLS = ALL_SYMBOLS
