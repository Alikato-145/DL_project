import os
import sys

# Set environment variable to disable signal handlers
os.environ["ULTRALYTICS_SKIP_SIGNALS"] = "TRUE"
# Alternative environment variables that might work
os.environ["YOLO_VERBOSE"] = "FALSE"

# Patch the signal module before ultralytics tries to use it
import signal
original_signal = signal.signal

def patched_signal(signalnum, handler):
    try:
        return original_signal(signalnum, handler)
    except ValueError:
        # If setting the signal fails, just return the previous handler
        return None

# Apply the patch
signal.signal = patched_signal

# Now it's safe to import ultralytics
import ultralytics