import logging
import pathlib
import traceback
import yaml
import threading


# A dictionary to hold the configuration data
_config = {}
# A threading.Lock to ensure thread-safe access to the configuration data
_config_lock = threading.Lock()

def load_config(config_file: pathlib.Path):
    """
    Loads the configuration from the YAML file.
    This function is thread-safe.
    """
    global _config
    with _config_lock:
        print("Loading configuration from \"%s\"..." % config_file)
        try:
            with open(config_file, "r") as f:
                _config = yaml.safe_load(f)
            print("Configuration loaded successfully.")
        except FileNotFoundError:
            print("Warning: \"%s\" not found." % config_file)
            _config = {}
        except yaml.YAMLError as e:
            print("Error loading YAML file.")
            traceback.print_exc()

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
