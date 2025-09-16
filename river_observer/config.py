import logging
import pathlib
import yaml
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# A dictionary to hold the configuration data
_config = {}
# A threading.Lock to ensure thread-safe access to the configuration data
_config_lock = threading.Lock()
_config_logger = logging.getLogger("YAMLConfig")


class ConfigFileEventHandler(FileSystemEventHandler):
    """
    Handles file system events for the configuration file.
    This class is part of the watchdog library setup.
    """

    def __init__(self, config_file: pathlib.Path):
        super().__init__()
        self.config_file = config_file

    def on_modified(self, event):
        """
        Called when the configuration file is modified.
        """
        if event.src_path == self.config_file:
            _config_logger.info(f"Configuration file modified. Reloading...")
            load_config(self.config_file)

def load_config(config_file: pathlib.Path):
    """
    Loads the configuration from the YAML file.
    This function is thread-safe.
    """
    global _config
    with _config_lock:
        _config_logger.info("Loading configuration from \"%s\"...", config_file)
        try:
            with open(config_file, "r") as f:
                _config = yaml.safe_load(f)
            _config_logger.info("Configuration loaded successfully.")
        except FileNotFoundError:
            _config_logger.warning("\"%s\" not found.", config_file)
            _config = {}
        except yaml.YAMLError as e:
            _config_logger.exception("Error loading YAML file.", exc_info=e)

def get_config():
    """
    Public function for other scripts to get the current configuration.
    Returns a deep copy of the configuration dictionary to prevent
    external scripts from modifying the internal state directly.
    """
    import copy

    with _config_lock:
        # Return a copy to prevent external modification
        return copy.deepcopy(_config)

def start_watcher(config_file = pathlib.Path().joinpath("config.yaml")):
    """
    Starts the watchdog observer to watch for changes to the config file.
    This runs in a separate thread to not block the main application.
    """
    # Initial load of the configuration
    load_config(config_file)

    event_handler = ConfigFileEventHandler(config_file)
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    _config_logger.info("Watching for changes to \"%s\"...", config_file)

    return observer
