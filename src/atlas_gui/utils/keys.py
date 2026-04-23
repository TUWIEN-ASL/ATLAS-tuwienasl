from PyQt5.QtCore import Qt

# Optional: Add valid key names explicitly if you want stricter control
VALID_KEYS = {name[4:]: getattr(Qt, name) for name in dir(Qt) if name.startswith("Key_")}

def key_string_to_qt_enum(key_str):
    """
    Convert a string representation of a key to the corresponding Qt key enum.

    Args:
        key_str (str): A string like 'Space', 'A', 'Enter', etc.

    Returns:
        Qt.Key: The corresponding Qt enum value (e.g., Qt.Key_Space).

    Raises:
        ValueError: If the string does not match any valid Qt key name.
    """
    key_str = str(key_str).strip().capitalize()
    if key_str in VALID_KEYS:
        return VALID_KEYS[key_str]
    raise ValueError(f"Invalid Qt key name: '{key_str}'")

def load_key_bindings_from_config(config_dict, default_keys=None):
    """
    Load and validate key bindings from a configuration dictionary.

    Args:
        config_dict (dict): A mapping of action names to key names 
                            (e.g., {'play': 'Space', 'next': 'Right'}).
        default_keys (dict, optional): A fallback dictionary to use if 
                                       a key is invalid or missing.

    Returns:
        dict: A mapping from action names to Qt key enums.

    Raises:
        ValueError: If a key is invalid and no default is provided.
    """
    bindings = {}
    for action, key_name in config_dict.items():
        try:
            bindings[action] = key_string_to_qt_enum(key_name)
        except ValueError as e:
            if default_keys and action in default_keys:
                print(f"Warning: {e}, using default key '{default_keys[action]}'")
                bindings[action] = key_string_to_qt_enum(default_keys[action])
            else:
                raise
    return bindings


def load_action_map_from_config(action_map_dict):
    """
    Convert an action map from key strings to Qt key enums.

    Args:
        action_map_dict (dict): A mapping from key strings to action names
                                (e.g., {'1': 'Approach', 'G': 'Grasp'}).

    Returns:
        dict: A mapping from Qt key enums to action names 
              (e.g., {Qt.Key_1: 'Approach', Qt.Key_G: 'Grasp'}).
    """
    qt_action_map = {}
    for key_str, action in action_map_dict.items():
        qt_key = key_string_to_qt_enum(key_str)
        qt_action_map[qt_key] = action
    return qt_action_map