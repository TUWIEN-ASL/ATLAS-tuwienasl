from atlas_gui.datasets.dataset import DatasetBase
import threading
import h5py
import numpy as np
import copy
import os
import json
from atlas_gui.utils.reassemble_helpers import load_h5_time_interval, load_segments_info as helper_load_segments

class Reassemble(DatasetBase):
    """
    Dataset class for handling REASSEMBLE HDF5-formatted data.

    This class supports loading, caching, and annotating segments of data stored
    in REASSEMBLE format. It uses a moving window cache to efficiently load
    previous, current, and next segments, and provides thread-safe file access.
    """
     
    def __init__(self, config):
        """
        Initialize the Reassemble dataset handler.

        Args:
            config (dict): Configuration dictionary.
                           - annotation_storage: "h5" (default, in-place) or "json" (external file)
                           - annotation_dir: directory for JSON files (required if annotation_storage="json")
                           - annotation_group: group name in H5 or identifier for JSON
        """
        super().__init__()
        self.config = config
        self.timestamps = None
        self.segments_info = None
        self.segments = []
        self.current_segment_idx = 0
        self.annot_group = self.config.get('annotation_group', 'low_level')

        # Annotation storage mode: "h5" (in-place) or "json" (external)
        self.annotation_storage = self.config.get('annotation_storage', 'h5').lower()
        self.annotation_dir = self.config.get('annotation_dir', 'annotations/reassemble/')
        self.dataset_name = self.config.get('dataset_name', 'reassemble')

        if self.annotation_storage == 'json':
            os.makedirs(self.annotation_dir, exist_ok=True)

        # Moving window cache
        self.cache = {"prev": None, "current": None, "next": None}

        # Thread safety
        self.lock = threading.RLock()            # Lock for cache updates
        self.file_lock = threading.RLock()       # Lock for file access
        self.thread = None

        self.is_busy = False

    def load_data(self, file_path):
        """
        Load the HDF5 file and initialize segment metadata and cache.

        Args:
            file_path (str): Path to the H5 file.
        """
        if self.file_path == file_path:
            return  # No need to reload if the file path is the same

        self.file_path = file_path
        self.current_segment_idx = 0

        # Reset cache
        self.cache = {"prev": None, "current": None, "next": None}

        self.load_segments_info(file_path)

        # Load initial cache
        self._load_initial_cache()
        
    def _load_initial_cache(self):
        """
        Load the first and second segments into the cache, if they exist.
        Happens when loading the data for the first time.
        """
        if not self.file_path or not self.segments_info:
            return

        if "0" in self.segments_info:
            self.cache["current"] = self._load_segment(0)
            self.current_segment_idx = 0
        
        if "1" in self.segments_info:
            self.cache["next"] = self._load_segment(1)

    def load_segments_info(self, file_path):
        """
        Load segment metadata.

        Args:
            file_path (str): Path to the H5 file.
        """
        self.segments_info = helper_load_segments(file_path)
    
    def set_segments_info(self, segments_info):
        """
        Manually set the segment metadata.

        Args:
            segments_info (dict): Dictionary of segment info.
        """
        self.segments_info = segments_info

    def _load_segment(self, segment_idx):
        """
        Load data for a specific segment from the H5 file.

        Args:
            segment_idx (int): Index of the segment.

        Returns:
            dict: Loaded segment data.
        """
        if str(segment_idx) not in self.segments_info:
            return None
        segment = self.segments_info[str(segment_idx)]
        with self.file_lock:
            data = load_h5_time_interval(self.file_path, segment["start"], segment["end"])
        return data

    def _preload(self, preload_idx):
        """
        Preload a segment asynchronously into the appropriate cache slot.

        Args:
            preload_idx (int): Index of the segment to preload.
        """
        with self.lock:
            if str(preload_idx) not in self.segments_info:
                return
            preload_data = self._load_segment(preload_idx)
        
            if preload_idx > self.current_segment_idx:
                self.cache["next"] = preload_data  # Preloading forward
            elif preload_idx < self.current_segment_idx:
                self.cache["prev"] = preload_data  # Preloading backward

    def _preload_both_directions(self, current_idx):
        """
        Preload both previous and next segments synchronously.
        
        Happens usually when jumping to a specific segment. In that case, previous cache
        cannot be reused, therefore we reload previous and next segment for the new
        current segment. 

        Args:
            current_idx (int): Index of the current segment.
        """
        # Get all available segment indices
        available_segments = [int(k) for k in self.segments_info.keys()]
        
        # Preload previous segment if it exists
        prev_idx = current_idx - 1
        if prev_idx in available_segments:
            if str(prev_idx) in self.segments_info:
                self.cache["prev"] = self._load_segment(prev_idx)
        else:
            self.cache["prev"] = None
        
        # Preload next segment if it exists
        next_idx = current_idx + 1
        if next_idx in available_segments:
            if str(next_idx) in self.segments_info:
                self.cache["next"] = self._load_segment(next_idx)
        else:
            self.cache["next"] = None

    def get_segment(self, segment_idx):
        """
        Retrieve a segment by index using the cache and support for sequential navigation.

        Args:
            segment_idx (int): Index of the segment to retrieve.

        Returns:
            dict: Segment data.
        """
        with self.lock:
            # Check if we're moving sequentially (step of 1)
            is_sequential = abs(segment_idx - self.current_segment_idx) == 1
            
            if segment_idx == self.current_segment_idx:
                # Same segment, no need to update cache
                pass
            elif is_sequential and segment_idx == self.current_segment_idx - 1:  # backwards
                self.cache["next"] = self.cache["current"]
                self.cache["current"] = self.cache["prev"]
                # Preload new previous segment
                preload_idx = segment_idx - 1
            elif is_sequential and segment_idx == self.current_segment_idx + 1:  # forwards
                self.cache["prev"] = self.cache["current"]
                self.cache["current"] = self.cache["next"]
                # Preload new next segment
                preload_idx = segment_idx + 1
            else:
                # Jump to non-sequential segment - need to reload everything
                self.cache["current"] = self._load_segment(segment_idx)
                
                # For jumps, we'll preload both directions in background
                self.current_segment_idx = segment_idx
                
                # Stop any existing preload thread
                if self.thread and self.thread.is_alive():
                    self.thread.join()
                
                # Start thread to preload both prev and next segments
                self.thread = threading.Thread(target=self._preload_both_directions, args=(segment_idx,))
                self.thread.start()
                
                return self.cache['current']
            
            # Update current segment index
            self.current_segment_idx = segment_idx
            
            # For sequential movement, preload in the appropriate direction
            if is_sequential:
                if self.thread and self.thread.is_alive():
                    self.thread.join()
                
                # Only preload if the target segment exists
                if str(preload_idx) in self.segments_info:
                    self.thread = threading.Thread(target=self._preload, args=(preload_idx,))
                    self.thread.start()
            
            return self.cache['current']
    
    def get_max_timestamp(self):
        """
        Get the maximum timestamp from current segment after offsetting to start at zero.

        Returns:
            float: Maximum relative timestamp.
        """
        all_timestamps = []
        for timestamps in self.cache['current']['timestamps'].values():
            all_timestamps.extend(timestamps)
        timestamp_offset = min(all_timestamps)
        for key in self.cache['current']['timestamps']:
            self.cache['current']['timestamps'][key] = np.array(self.cache['current']['timestamps'][key]) - timestamp_offset

        max_timestamp = 1 * (max(all_timestamps) - timestamp_offset)    # * 1000
        return max_timestamp

    def write_annot_data(self, segment_idx, annots):
        """
        Write annotations either to H5 file (in-place) or external JSON file.

        Args:
            segment_idx (int): Index of the segment to annotate.
            annots (dict): Dictionary of annotations to save.
        """
        if self.annotation_storage == 'json':
            self._write_annot_json(segment_idx, annots)
        else:
            self._write_annot_h5(segment_idx, annots)

    def _write_annot_h5(self, segment_idx, annots):
        """Write annotations directly to the H5 file."""
        with self.file_lock:
            with h5py.File(self.file_path, "a") as f:
                segment_group = f[f"segments_info/{segment_idx}"]
                if self.annot_group in segment_group:
                    del segment_group[self.annot_group]
                low_level_group = segment_group.create_group(self.annot_group)

                for i, (_, ann) in enumerate(annots.items()):
                    ann_group = low_level_group.create_group(f"{i}")
                    ann_group.create_dataset("end", data=np.array(ann["end"], dtype="float64"))
                    ann_group.create_dataset("start", data=np.array(ann["start"], dtype="float64"))
                    ann_group.create_dataset("success", data=np.array(ann["success"], dtype="bool"))
                    ann_group.create_dataset("text", data=np.bytes_(ann["label"]))

    def _get_json_annotation_path(self):
        """Get the JSON annotation file path based on current H5 file."""
        if self.file_path:
            # Use the H5 filename (without extension) for the JSON file
            h5_basename = os.path.splitext(os.path.basename(self.file_path))[0]
            return os.path.join(self.annotation_dir, f"{h5_basename}_annotations.json")
        else:
            # Fallback to dataset_name if no file loaded
            return os.path.join(self.annotation_dir, f"{self.dataset_name}_annotations.json")

    def _write_annot_json(self, segment_idx, annots):
        """Write annotations to an external JSON file."""
        annotations_path = self._get_json_annotation_path()

        if os.path.exists(annotations_path):
            with open(annotations_path, "r") as f:
                all_annotations = json.load(f)
        else:
            all_annotations = {}

        def convert(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        # Use segment index as key (or could use a UID if available)
        uid = f"segment_{segment_idx}"
        all_annotations[uid] = convert(annots)

        with open(annotations_path, "w") as f:
            json.dump(all_annotations, f, indent=2)

    def load_annot_data(self, segment_idx):
        """
        Load annotations either from H5 file or external JSON file.

        Args:
            segment_idx (int): Index of the segment.

        Returns:
            dict: Dictionary of annotation data.
        """
        if self.annotation_storage == 'json':
            return self._load_annot_json(segment_idx)
        else:
            return self._load_annot_h5(segment_idx)

    def _load_annot_h5(self, segment_idx):
        """Load annotations from the H5 file, adjusting for segment start."""
        if not self.file_path:
            return {}
        segment = self.segments_info[str(segment_idx)]
        segment_start = segment['start']
        annots_ll = copy.deepcopy(segment.get(self.annot_group, None))
        if annots_ll is None:
            return {}

        for ann_id, ann in annots_ll.items():
            ann['start'] -= segment_start
            ann['end'] -= segment_start

        return annots_ll

    def _load_annot_json(self, segment_idx):
        """Load annotations from an external JSON file."""
        uid = f"segment_{segment_idx}"
        annotations_path = self._get_json_annotation_path()

        if not os.path.exists(annotations_path):
            return {}

        with open(annotations_path, 'r') as f:
            all_annotations = json.load(f)
        
        annots_ll = all_annotations.get(uid, {})
        if annots_ll:
            segment = self.segments_info[str(segment_idx)]
            segment_start = segment['start']
            
            annots_ll = copy.deepcopy(annots_ll)
            for ann_id, ann in annots_ll.items():
                ann['start'] -= segment_start
                ann['end'] -= segment_start

        return annots_ll