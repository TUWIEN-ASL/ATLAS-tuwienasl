import yaml
from atlas_gui.datasets.dataset import DatasetBase
from atlas_gui.datasets.reassemble import Reassemble
from atlas_gui.datasets.rlds import RLDS
from atlas_gui.datasets.rosbag_ds import Rosbag
from atlas_gui.datasets.frames import Frames
from atlas_gui.datasets.video import Video

def load_config(path="config.yaml"):
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the configuration file. Defaults to 'config.yaml'.

    Returns:
        dict: Parsed YAML configuration as a Python dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def get_nested(data_dict, key_path):
    """
    Access a nested dictionary using a slash-separated key path.

    Args:
        data_dict (dict): The dictionary to traverse.
        key_path (str): Slash-separated path to the nested key (e.g., 'robot_state/joint_efforts').

    Returns:
        Any: The value found at the nested key path.
    """
    keys = key_path.split('/')
    for key in keys:
        data_dict = data_dict[key]
    return data_dict

def has_nested_key(data, path):
    """
    Check whether a nested key path exists in a dictionary.

    Args:
        data (dict): The dictionary to search.
        path (str): Slash-separated path representing the nested key.

    Returns:
        bool: True if the nested key exists, False otherwise.
    """
    keys = path.split('/')
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return False
        data = data[key]
    return True

def get_nested_np(data_dict, key_path):
    """
    Access a nested dictionary key and convert the result to a NumPy array.

    Specifically used for RLDS datasets, where TensorFlow tensors need to be converted.

    Args:
        data_dict (dict): The dictionary to access.
        key_path (str): Slash-separated path to the nested key.

    Returns:
        np.ndarray: The NumPy array corresponding to the nested value.
    """
    keys = key_path.split('/')
    for key in keys:
        data_dict = data_dict[key]
    return data_dict.numpy()

def create_dataset(dataset_type: str, config) -> DatasetBase:
    """
    Factory method to instantiate the appropriate dataset class.

    Args:
        dataset_type (str): Type of dataset to load, either 'reassemble' or 'rlds'.
        config (dict): Configuration dictionary to pass to the dataset.

    Returns:
        DatasetBase: An instance of either Reassemble or RLDS dataset.

    Raises:
        ValueError: If the dataset_type is unsupported.
    """ 
    dataset_type = dataset_type.lower()
    
    if dataset_type == "reassemble":
        return Reassemble(config)
    elif dataset_type == "rlds":
        return RLDS(config)
    elif dataset_type == "frames":
        return Frames(config)
    elif dataset_type == "video":
        return Video(config)
    elif dataset_type == "rosbag":
        return Rosbag(config)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")