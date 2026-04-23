from atlas_gui.datasets.dataset import DatasetBase
import tensorflow_datasets as tfds
import os
import json
import numpy as np
from itertools import islice

class RLDS(DatasetBase):
    """
    Dataset handler for RLDS (Reinforcement Learning Datasets) format datasets.

    This class loads RLDS datasets via TensorFlow Datasets, and supports segment access, annotation I/O,
    and automatic structuring of episode information. Each segment corresponds to an entire episode.
    """
    def __init__(self, config, split="train"):
        """
        Initialize the RLDS dataset.

        Args:
            config (dict): Configuration dictionary. Must contain keys:
                           - 'dataset_name': str, used to identify the dataset
                           - 'annotation_dir': str, path to store annotation JSONs
                           - 'fps': int or float, used to compute episode duration
                           Optional keys:
                           - 'download': bool, if True use tfds.load() to auto-download
                           - 'data_dir': str, directory for downloaded datasets
            split (str): Dataset split to load, e.g., 'train'. Defaults to 'train'.
        """
        super().__init__()
        self.config = config
        self.dataset_name = self.config['dataset_name']
        self.split = split
        self.dataset = None
        self.download_mode = self.config.get('download', False)
        self.data_dir = os.path.expanduser(self.config.get('data_dir', '~/tensorflow_datasets'))

        self._iterator = None
        self._current_episode = None
        self.current_segment_idx = 0

        self.annotation_dir = self.config['annotation_dir']
        os.makedirs(self.annotation_dir, exist_ok=True)

    def load_data(self, file_path=None):
        """
        Load the RLDS dataset.

        Args:
            file_path (str, optional): Path to the TFDS-formatted dataset directory.
                                       Required if download=False, ignored if download=True.
        """
        if self.download_mode:
            # Auto-download mode: use tfds.load()
            print(f"Loading dataset '{self.dataset_name}' from tfds (data_dir={self.data_dir})...")
            self.file_path = self.data_dir
            self.dataset = tfds.load(
                self.dataset_name,
                split=self.split,
                data_dir=self.data_dir,
                shuffle_files=False
            )
        else:
            # Local mode: use builder_from_directory()
            if file_path is None:
                raise ValueError("file_path required when download=False")
            self.file_path = file_path
            builder = tfds.builder_from_directory(builder_dir=file_path)
            self.dataset = builder.as_dataset(split='train', shuffle_files=False)

        self._iterator = iter(self.dataset)
        self.load_segments_info(file_path=self.file_path)
        self._iterator = iter(self.dataset)

    def get_segment(self, segment_idx):
        """
        Return a specific segment (episode) from the dataset, stacked into NumPy arrays.

        Args:
            segment_idx (int): Index of the segment to load.

        Returns:
            dict: A dictionary containing stacked step-wise data (e.g., observations, actions).
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        # Sequential forward step
        if (segment_idx == self.current_segment_idx + 1 and segment_idx != 0) or \
       (segment_idx == 0 and self.current_segment_idx == -1):
            try:
                self._current_episode = next(self._iterator)
                self.current_segment_idx = segment_idx
            except StopIteration:
                raise IndexError(f"Segment index {segment_idx} out of range.")
        else:
            try:
                self._iterator = iter(self.dataset)
                self._current_episode = next(islice(self._iterator, segment_idx, segment_idx + 1))
                self.current_segment_idx = segment_idx
            except StopIteration:
                raise IndexError(f"Segment index {segment_idx} out of range.")

        steps = list(self._current_episode['steps'])

        def recursively_stack(keys, depth=0):
            result = {}
            for key in keys:
                try:
                    sample = steps[0][key]
                    # Nested dict → recurse
                    if isinstance(sample, dict):
                        nested_keys = sample.keys()
                        result[key] = {}
                        for nk in nested_keys:
                            try:
                                result[key][nk] = np.stack([step[key][nk].numpy() for step in steps])
                            except Exception as e:
                                print(f"{'  ' * depth}Warning: could not stack '{key}/{nk}': {e}")
                    else:
                        result[key] = np.stack([step[key].numpy() for step in steps])
                except Exception as e:
                    print(f"{'  ' * depth}Warning: could not stack '{key}': {e}")
            return result

        stacked = recursively_stack(steps[0].keys())

        self._current_episode = {"steps": stacked}
        return self._current_episode

    def load_segments_info(self, file_path=None):
        """
        Load metadata for each episode in the dataset.

        Creates a dictionary where each segment entry includes:
        - index
        - start time
        - end time (based on FPS and number of steps)
        - language instruction (text)
        - unique ID (derived from episode metadata or index)
        """
        if self.dataset is None:
            raise ValueError("Call load_data() before loading segments info.")

        self.segments_info = {}

        for idx, episode in enumerate(self.dataset):
            steps = episode["steps"]
            num_steps = len(steps)

            # Extract text from first step using configured text_keys
            text = ""
            if self.config.get('text_keys'):
                try:
                    sample_step = next(iter(steps))
                    text_key = self.config['text_keys'][0]
                    text_value = sample_step[text_key]
                    if hasattr(text_value, 'numpy'):
                        text_value = text_value.numpy()
                    if isinstance(text_value, bytes):
                        text = text_value.decode("utf-8")
                    else:
                        text = str(text_value)
                except Exception:
                    text = ""

            # Extract a unique ID - try episode_metadata first, fall back to index
            unique_id = f"{self.dataset_name}_episode_{idx}"
            try:
                if "episode_metadata" in episode:
                    metadata = episode["episode_metadata"]
                    if "recording_folderpath" in metadata:
                        folder_path = metadata["recording_folderpath"].numpy().decode("utf-8")
                        unique_id = os.path.basename(os.path.dirname(os.path.dirname(folder_path)))
                    elif "episode_id" in metadata:
                        unique_id = str(metadata["episode_id"].numpy())
                    elif "file_path" in metadata:
                        unique_id = os.path.basename(metadata["file_path"].numpy().decode("utf-8"))
            except Exception:
                pass  # Keep default unique_id

            segment_info = {
                "index": idx,
                "start": 0.0,
                "end": (num_steps - 1) / self.config['fps'],
                "text": text,
                "uid": unique_id
            }
            self.segments_info[str(idx)] = segment_info


    def get_max_timestamp(self):
        """
        Return the end timestamp of the current episode.

        Returns:
            float: The maximum timestamp for the current segment.
        """
        return self.segments_info[str(self.current_segment_idx)]['end']


    def write_annot_data(self, segment_idx, annots):
        """
        Write annotation data to a per-dataset JSON file.

        Annotations are stored using the UID of the segment as the key.

        Args:
            segment_idx (int): Index of the segment being annotated.
            annots (dict): Dictionary of annotations (must be JSON serializable).
        """
        annotations_path = os.path.join(self.annotation_dir, f"{self.dataset_name}_annotations.json")
        # Load existing annotations if the file exists
        if os.path.exists(annotations_path):
            with open(annotations_path, "r") as f:
                all_annotations = json.load(f)
        else:
            all_annotations = {}

        # Convert numpy values to Python types (e.g., float64 → float)
        def convert(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        all_annotations[self.segments_info[str(segment_idx)]['uid']] = convert(annots)

        # Save back to JSON
        with open(annotations_path, "w") as f:
            json.dump(all_annotations, f, indent=2)

    def load_annot_data(self, segment_idx):
        """
        Load annotation data for the given segment index, using UID-based lookup.

        Args:
            segment_idx (int): Index of the segment to load annotations for.

        Returns:
            dict: Annotation data for the given segment, or an empty dict if none found.
        """
        uid = self.segments_info[str(segment_idx)]['uid']
        annotations_path = os.path.join(self.annotation_dir, f"{self.config['dataset_name']}_annotations.json")

        if not os.path.exists(annotations_path):
            return {}

        with open(annotations_path, 'r') as f:
            all_annotations = json.load(f)

        annot_data = all_annotations.get(uid, {})
        return annot_data
