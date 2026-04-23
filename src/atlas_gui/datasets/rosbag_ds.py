"""
RosbagDataset: A dataset handler for ROS bag files without requiring ROS installation.

This module uses the `rosbags` library to read ROS1 (.bag) and ROS2 (.db3) bag files.

Install dependencies:
    pip install rosbags numpy

For image decompression support:
    pip install opencv-python
"""

from atlas_gui.datasets.dataset import DatasetBase
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict

# rosbags imports - pure Python, no ROS required
from rosbags.rosbag1 import Reader as Reader1
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys import Stores, get_typestore

# Try to import deserialize functions (older rosbags versions)
try:
    from rosbags.serde import deserialize_cdr, deserialize_ros1
    _USE_OLD_SERDE_API = True
except ImportError:
    _USE_OLD_SERDE_API = False


class Rosbag(DatasetBase):
    """
    Dataset handler for ROS bag files (both ROS1 .bag and ROS2 .db3 formats).

    This class loads rosbag files using the pure-Python `rosbags` library,
    requiring no ROS installation. Each bag file in the folder is treated as a separate segment.

    Config structure (YAML):
        dataset_type: rosbag
        dataset_name: my_rosbag_dataset
        fps: 30
        annotation_dir: annotations/rosbag/
        annotation_group: low_level

        low_level_keys:
          - /robot_state/joint_positions
          - /robot_state/joint_velocities
          - /robot_state/ee_pose

        camera_keys:
          - /camera/color/image_raw
          - /camera/depth/image_raw

        color_format: "BGR"

        default_graphs:
          - /robot_state/joint_positions

        action_map:
          1: Approach
          2: Grasp
          ...
        
        # Streaming mode (default: True) - loads camera frames on-demand
        # Set to False to load everything into memory
        stream_mode: True
        
        # Frame cache size for streaming mode (default: 30)
        frame_cache_size: 30
    """

    def __init__(self, config: Dict[str, Any], split: str = "train"):
        """
        Initialize the Rosbag dataset.

        Args:
            config (dict): Configuration dictionary matching the YAML structure above.
            split (str): Dataset split identifier. Defaults to 'train'.
        """
        super().__init__()
        self.config = config
        self.dataset_name = config.get('dataset_name', 'rosbag_dataset')
        self.split = split

        self.annotation_dir = config.get('annotation_dir', './annotations/rosbag/')
        os.makedirs(self.annotation_dir, exist_ok=True)
        self.annotation_group = config.get('annotation_group', 'annotations')

        self.fps = config.get('fps', 30.0)
        
        # Topic configuration - matching your config structure
        self.camera_keys = config.get('camera_keys', [])
        self.low_level_keys = config.get('low_level_keys', [])
        
        self.color_format = config.get('color_format', 'BGR')
        self.default_graphs = config.get('default_graphs', [])
        self.action_map = config.get('action_map', {})

        # Internal state
        self.bag_files: List[Path] = []
        self.current_segment_idx = 0
        self._current_segment_data = None
        self.segments_info = {}

        # Type store for message deserialization
        self._typestore = None
        
        # ROS version (auto-detected)
        self._ros_version = None
        
        # Topic name mapping (cleaned name -> original topic)
        self._topic_mapping = {}
        
        # Streaming mode state
        self._message_index = None  # Maps topic -> list of (timestamp, offset/position)
        self._current_reader = None
        self._stream_mode = config.get('stream_mode', True)  # Default to streaming
        
        # Frame cache for streaming mode (LRU-style)
        self._frame_cache = {}  # (topic, frame_idx) -> image
        self._frame_cache_order = []  # Track access order
        self._frame_cache_size = config.get('frame_cache_size', 30)  # Cache last N frames

    def _detect_ros_version(self, file_path: Union[str, Path]) -> int:
        """
        Auto-detect ROS version based on file extension or structure.

        Args:
            file_path: Path to the bag file or directory.

        Returns:
            int: 1 for ROS1, 2 for ROS2
        """
        path = Path(file_path)

        if path.is_file():
            if path.suffix == '.bag':
                return 1
            elif path.suffix == '.db3':
                return 2
        elif path.is_dir():
            # ROS2 bags are directories containing .db3 files
            if any(path.glob('*.db3')):
                return 2
            if any(path.glob('*.bag')):
                return 1

        # Default to ROS1
        return 1

    def _get_reader(self, bag_path: Path):
        """
        Get the appropriate reader for the bag file.

        Args:
            bag_path: Path to the bag file.

        Returns:
            Reader context manager (Reader1 or Reader2)
        """
        ros_version = self._ros_version or self._detect_ros_version(bag_path)

        if ros_version == 1:
            return Reader1(bag_path)
        else:
            return Reader2(bag_path)

    def _get_typestore(self, ros_version: int):
        """Get or create the typestore for message deserialization."""
        if self._typestore is None:
            if ros_version == 1:
                self._typestore = get_typestore(Stores.ROS1_NOETIC)
            else:
                self._typestore = get_typestore(Stores.ROS2_HUMBLE)
        return self._typestore

    def _deserialize_message(self, rawdata: bytes, msgtype: str, ros_version: int):
        """
        Deserialize a raw message.

        Args:
            rawdata: Raw message bytes.
            msgtype: Message type string (e.g., 'sensor_msgs/msg/Image').
            ros_version: ROS version (1 or 2).

        Returns:
            Deserialized message object.
        """
        typestore = self._get_typestore(ros_version)
        
        # Old rosbags API (< 0.9.12)
        if _USE_OLD_SERDE_API:
            if ros_version == 1:
                return deserialize_ros1(rawdata, msgtype, typestore)
            else:
                return deserialize_cdr(rawdata, msgtype, typestore)
        
        # New rosbags API (>= 0.9.12) - use typestore methods
        if ros_version == 1:
            return typestore.deserialize_ros1(rawdata, msgtype)
        else:
            return typestore.deserialize_cdr(rawdata, msgtype)

    def _clean_topic_name(self, topic: str) -> str:
        """
        Convert topic name to a clean key format matching config style.
        
        e.g., '/robot_state/joint_positions' -> 'robot_state/joint_positions'
        """
        return topic.lstrip('/')

    def _topic_matches_key(self, topic: str, key: str) -> bool:
        """
        Check if a topic matches a config key.
        
        Handles both with and without leading slash.
        """
        clean_topic = self._clean_topic_name(topic)
        clean_key = self._clean_topic_name(key)

        # print(clean_topic)
        # print(clean_key)

        return clean_topic == clean_key or topic == key or clean_key in clean_topic

    def _decode_image(self, msg, encoding: Optional[str] = None) -> np.ndarray:
        """
        Decode an image message to a numpy array.

        Args:
            msg: Image message (sensor_msgs/Image or sensor_msgs/CompressedImage).
            encoding: Image encoding override.

        Returns:
            np.ndarray: Decoded image as HWC numpy array.
        """
        # Check if it's a CompressedImage
        if hasattr(msg, 'format'):
            # CompressedImage - need opencv or similar to decode
            try:
                import cv2
                np_arr = np.frombuffer(msg.data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    # Handle color format conversion
                    if self.color_format.upper() == 'RGB':
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img
            except ImportError:
                print("Warning: opencv-python not installed. Cannot decode compressed images.")
            return np.array(msg.data)

        # Regular Image message
        enc = encoding or getattr(msg, 'encoding', 'rgb8')
        height = msg.height
        width = msg.width
        data = np.frombuffer(msg.data, dtype=np.uint8)

        # Handle different encodings
        if enc in ['rgb8', 'bgr8']:
            img = data.reshape((height, width, 3))
            # Convert based on config color_format
            if enc == 'bgr8' and self.color_format.upper() == 'RGB':
                img = img[..., ::-1]  # BGR to RGB
            elif enc == 'rgb8' and self.color_format.upper() == 'BGR':
                img = img[..., ::-1]  # RGB to BGR
        elif enc in ['rgba8', 'bgra8']:
            img = data.reshape((height, width, 4))
            if enc == 'bgra8' and self.color_format.upper() == 'RGB':
                img = img[..., [2, 1, 0, 3]]  # BGRA to RGBA
            elif enc == 'rgba8' and self.color_format.upper() == 'BGR':
                img = img[..., [2, 1, 0, 3]]  # RGBA to BGRA
        elif enc == 'mono8':
            img = data.reshape((height, width))
        elif enc == '16UC1':
            data = np.frombuffer(msg.data, dtype=np.uint16)
            img = data.reshape((height, width))
        elif enc == '32FC1':
            data = np.frombuffer(msg.data, dtype=np.float32)
            img = data.reshape((height, width))
        else:
            # Try to guess based on step size
            channels = msg.step // width if width > 0 else 1
            if channels == 1:
                img = data.reshape((height, width))
            elif channels in [3, 4]:
                img = data.reshape((height, width, channels))
            else:
                img = data

        return img

    def _extract_pose(self, msg) -> np.ndarray:
        """
        Extract pose data as a single stacked vector:
        [x, y, z, qx, qy, qz, qw]
        """

        # Handle PoseStamped, Pose, Transform, etc.
        if hasattr(msg, 'pose'):
            pose = msg.pose
        elif hasattr(msg, 'transform'):
            pose = msg.transform
        else:
            pose = msg

        # Position / translation
        if hasattr(pose, 'position'):
            p = pose.position
            pos = np.array([p.x, p.y, p.z])
        elif hasattr(pose, 'translation'):
            t = pose.translation
            pos = np.array([t.x, t.y, t.z])
        else:
            pos = np.zeros(3)

        # Orientation / rotation
        if hasattr(pose, 'orientation'):
            q = pose.orientation
            ori = np.array([q.x, q.y, q.z, q.w])
        elif hasattr(pose, 'rotation'):
            r = pose.rotation
            ori = np.array([r.x, r.y, r.z, r.w])
        else:
            ori = np.array([0.0, 0.0, 0.0, 1.0])

        return np.concatenate([pos, ori])

    def _extract_numeric_data(self, msg) -> np.ndarray:
        """
        Extract numeric data and return a single stacked numpy array.
        """

        # JointState
        if hasattr(msg, 'position') and hasattr(msg, 'velocity') and hasattr(msg, 'effort'):
            parts = []
            if hasattr(msg.position, '__len__') and len(msg.position) > 0:
                parts.append(np.array(msg.position))
            if hasattr(msg.velocity, '__len__') and len(msg.velocity) > 0:
                parts.append(np.array(msg.velocity))
            if hasattr(msg.effort, '__len__') and len(msg.effort) > 0:
                parts.append(np.array(msg.effort))

            return np.concatenate(parts) if parts else np.array([])

        # WrenchStamped / Wrench → [fx, fy, fz, tx, ty, tz]
        if hasattr(msg, 'wrench'):
            w = msg.wrench
            return np.array([
                w.force.x, w.force.y, w.force.z,
                w.torque.x, w.torque.y, w.torque.z
            ])
        elif hasattr(msg, 'force') and hasattr(msg, 'torque') and hasattr(msg.force, 'x'):
            return np.array([
                msg.force.x, msg.force.y, msg.force.z,
                msg.torque.x, msg.torque.y, msg.torque.z
            ])

        # Pose / Transform
        if hasattr(msg, 'pose') or hasattr(msg, 'transform'):
            return self._extract_pose(msg)
        if hasattr(msg, 'position') and hasattr(msg, 'orientation') and hasattr(msg.position, 'x'):
            return self._extract_pose(msg)

        # Twist → [vx, vy, vz, wx, wy, wz]
        if hasattr(msg, 'twist'):
            t = msg.twist
            return np.array([
                t.linear.x, t.linear.y, t.linear.z,
                t.angular.x, t.angular.y, t.angular.z
            ])
        elif hasattr(msg, 'linear') and hasattr(msg, 'angular') and hasattr(msg.linear, 'x'):
            return np.array([
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z
            ])

        # Float64MultiArray or similar
        if hasattr(msg, 'data'):
            data = msg.data
            if isinstance(data, (bytes, bytearray)):
                return np.frombuffer(data, dtype=np.float64)
            elif hasattr(data, '__len__') and not isinstance(data, str):
                return np.array(list(data))

        # Fallback: flatten all numeric fields
        values = []
        for attr in dir(msg):
            if attr.startswith('_'):
                continue
            try:
                val = getattr(msg, attr)
                if isinstance(val, (int, float)):
                    values.append(val)
                elif isinstance(val, np.ndarray):
                    values.append(val.ravel())
                elif hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
                    arr = np.array(list(val))
                    if arr.size > 0:
                        values.append(arr.ravel())
            except Exception:
                pass

        if values:
            return np.concatenate(
                [v if isinstance(v, np.ndarray) else np.array([v]) for v in values]
            )

        return np.array([])

    def load_data(self, file_path: str):
        """
        Load rosbag data from a folder containing bag files.
        Each bag file in the folder is treated as a separate segment.

        Args:
            file_path: Path to a directory containing bag files (.bag or .db3).
        """
        self.file_path = file_path
        path = Path(file_path)

        if not path.is_dir():
            raise ValueError(f"Expected a directory path, got: {file_path}")

        # Collect all bag files from the directory
        self.bag_files = []
        
        # Find all .bag files (ROS1)
        bag_files = sorted(list(path.glob('*.bag')))
        
        # Find all .db3 files (ROS2)
        db3_files = sorted(list(path.glob('*.db3')))
        
        # Check for ROS2 bag directories (subdirectories with metadata.yaml)
        ros2_dirs = sorted([d for d in path.iterdir() 
                           if d.is_dir() and (d / 'metadata.yaml').exists()])
        
        # Combine all bag files
        self.bag_files = bag_files + db3_files + ros2_dirs

        if not self.bag_files:
            raise ValueError(f"No bag files found in directory: {file_path}")

        print(f"Found {len(self.bag_files)} bag file(s) in {file_path}")
        print(f"Each bag will be treated as a separate segment")

        # Auto-detect ROS version from first bag
        self._ros_version = self._detect_ros_version(self.bag_files[0])

        # Load segment info - each bag is now a segment
        self.load_segments_info(file_path)
        self.current_segment_idx = 0

    def load_segments_info(self, file_path: str = None):
        """
        Load metadata for each bag file (segment) in the dataset.
        Each bag file is treated as one segment.

        Populates segments_info with:
            - index
            - start time
            - end time
            - duration
            - text (empty string, for compatibility)
            - uid (unique identifier from filename)
            - topics (list of available topics)
        """
        if not self.bag_files:
            raise ValueError("No bag files loaded. Call load_data() first.")

        self.segments_info = {}

        for idx, bag_path in enumerate(self.bag_files):
            ros_version = self._ros_version or self._detect_ros_version(bag_path)

            with self._get_reader(bag_path) as reader:
                # Get time bounds
                start_time = reader.start_time / 1e9  # Convert to seconds
                end_time = reader.end_time / 1e9

                # Get available topics
                topics = list(reader.topics.keys()) if hasattr(reader, 'topics') else []
                
                # Build camera topic resolution map
                resolved_cameras = {}
                for cam in self.camera_keys:
                    full = self._resolve_camera_topic(cam, topics)
                    if full:
                        resolved_cameras[cam] = full

                # Generate unique ID from filename
                uid = bag_path.stem if bag_path.is_file() else bag_path.name

                segment_info = {
                    "index": idx,
                    "start": 0.0,  # Relative start within segment
                    "end": end_time - start_time,
                    "duration": end_time - start_time,
                    "absolute_start": start_time,
                    "absolute_end": end_time,
                    "text": "",  # For compatibility with other dataset classes
                    "uid": uid,
                    "path": str(bag_path),
                    "topics": topics,
                    "message_count": reader.message_count if hasattr(reader, 'message_count') else 0,
                    "resolved_cameras": resolved_cameras
                }

                self.segments_info[str(idx)] = segment_info
                print(f"Segment {idx}: {uid} - Duration: {segment_info['duration']:.2f}s")

        self.data = list(range(len(self.segments_info)))

    def _stack_data(self, data_list: List[Any]) -> Any:
        """
        Stack a list of data items into arrays.
        """
        if not data_list:
            return np.array([])
            
        try:
            if isinstance(data_list[0], dict):
                # Stack dict values separately
                stacked_dict = {}
                for key in data_list[0].keys():
                    values = [d[key] for d in data_list if key in d]
                    try:
                        stacked_dict[key] = np.stack(values)
                    except Exception:
                        stacked_dict[key] = values
                return stacked_dict
            elif isinstance(data_list[0], np.ndarray):
                return np.stack(data_list)
            elif isinstance(data_list[0], str):
                return data_list  # Keep as list for strings
            else:
                return np.array(data_list)
        except Exception as e:
            print(f"Warning: Could not stack data: {e}")
            return data_list

    def _set_nested(self, d: Dict, keys: List[str], value: Any):
        """
        Set a value in a nested dict using a list of keys.
        
        Example: _set_nested(d, ['cam1', 'image_raw', 'compressed'], data)
        Results in: d['cam1']['image_raw']['compressed'] = data
        """
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    def _get_nested(self, d: Dict, keys: List[str], default=None) -> Any:
        """
        Get a value from a nested dict using a list of keys.
        """
        for key in keys:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return default
        return d

    def _topic_to_keys(self, topic: str) -> List[str]:
        """
        Convert a topic string to a list of keys.
        
        Example: '/cam1/image_raw/compressed' -> ['cam1', 'image_raw', 'compressed']
        """
        return [k for k in topic.strip('/').split('/') if k]

    def _build_nested_dict(self, flat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a flat dict with topic keys to a nested dict.
        
        Example:
            {'/cam1/image_raw/compressed': data, '/cam1/image_raw/compressed_timestamps': ts}
        Becomes:
            {'cam1': {'image_raw': {'compressed': data, 'compressed_timestamps': ts}}}
        """
        nested = {}
        for topic, value in flat_data.items():
            keys = self._topic_to_keys(topic)
            if keys:
                self._set_nested(nested, keys, value)
        return nested

    def _build_message_index(self, segment_idx: int) -> Dict[str, List[tuple]]:
        """
        Build an index of message positions for camera topics only.
        
        Returns:
            Dict mapping topic -> list of (relative_timestamp, frame_index)
        """
        bag_path = self.bag_files[segment_idx]
        
        index = defaultdict(list)
        msg_count = defaultdict(int)
        
        with self._get_reader(bag_path) as reader:
            base_time = reader.start_time
            
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                
                # Only index camera topics
                is_camera = any(self._topic_matches_key(topic, ck) for ck in self.camera_keys)
                if not is_camera:
                    continue
                
                rel_time = (timestamp - base_time) / 1e9
                index[topic].append((rel_time, msg_count[topic]))
                msg_count[topic] += 1
        
        return dict(index)

    def _resolve_camera_topic(self, cam_key: str, available_topics: List[str]) -> Optional[str]:
        """
        Resolve a short camera key like 'cam1' to the full topic path.
        """
        cam_key = cam_key.strip('/')

        for topic in available_topics:
            if topic.strip('/').startswith(cam_key + '/'):
                return topic
        return None


    def _final_topic_name(self, topic: str) -> str:
        """
        Returns the last element of a topic path.
        '/a/b/c' -> 'c'
        """
        return topic.strip('/').split('/')[-1]

    def get_segment(self, segment_idx: int) -> Dict[str, Any]:
        """
        Load and return a specific segment (bag file).
        
        In stream_mode (default): Only loads low_level_keys data fully.
        Camera data is indexed but loaded on-demand via get_frame_by_index().
        
        With stream_mode=False: Loads everything into memory (original behavior).
        
        The returned structure:
        
        - Camera data: FLAT structure with camera key directly containing image array
          Example: {'cam1': np.array (N, H, W, C)}
        - Low-level data: NESTED structure based on topic paths
          Example: {'robot_state': {'joint_positions': np.array (N, joints)}}
        - Timestamps: Top-level dict with all timestamps
          Example: {'timestamps': {'cam1': np.array (N,), 'joint_positions': np.array (N,)}}

        Args:
            segment_idx: Index of the segment to load.

        Returns:
            dict: Dictionary containing camera data (flat) and low-level data (nested).
        """
        if not self.bag_files:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        if segment_idx < 0 or segment_idx >= len(self.bag_files):
            raise IndexError(f"Segment index {segment_idx} out of range (0-{len(self.bag_files)-1})")

        bag_path = self.bag_files[segment_idx]
        ros_version = self._ros_version or self._detect_ros_version(bag_path)
        
        self.current_segment_idx = segment_idx
        
        # Clear frame cache when switching segments
        self._frame_cache = {}
        self._frame_cache_order = []

        if self._stream_mode:
            return self._get_segment_streaming(segment_idx, bag_path, ros_version)
        else:
            return self._get_segment_full(segment_idx, bag_path, ros_version)

    def _get_segment_streaming(self, segment_idx: int, bag_path: Path, ros_version: int) -> Dict[str, Any]:
        """
        Load segment in streaming mode - only loads low_level data, indexes cameras.
        """
        # Build message index for camera topics only
        self._message_index = self._build_message_index(segment_idx)
        
        # Only load low_level_keys data
        topic_data = defaultdict(list)
        timestamps = defaultdict(list)
        skipped_types = set()

        with self._get_reader(bag_path) as reader:
            base_time = reader.start_time

            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                
                # Skip topics not in low_level_keys
                is_low_level = any(self._topic_matches_key(topic, lk) for lk in self.low_level_keys)
                if not is_low_level:
                    continue

                print(topic, timestamp)

                rel_time = (timestamp - base_time) / 1e9

                try:
                    msg = self._deserialize_message(rawdata, connection.msgtype, ros_version)
                    data = self._extract_numeric_data(msg)
                    topic_data[topic].append(data)
                    timestamps[topic].append(rel_time)

                except KeyError:
                    if connection.msgtype not in skipped_types:
                        skipped_types.add(connection.msgtype)
                except Exception as e:
                    if topic not in skipped_types:
                        print(f"Warning: Could not process message on {topic}: {e}")
                        skipped_types.add(topic)

        if skipped_types:
            print(f"Skipped unknown/unsupported message types: {skipped_types}")

        segment_flat_data = {}
        timestamps_dict = {}
        frame_count_dict = {}

        # ---- low-level topics ----
        for topic, data_list in topic_data.items():
            if not data_list:
                continue
            segment_flat_data[topic] = self._stack_data(data_list)
            timestamps_dict[topic] = np.array(timestamps[topic])

        # ---- camera topics (indexed only) ----
        for topic, idx_info in self._message_index.items():
            # Use camera key from config for flat storage
            cam_key = None
            for ck in self.camera_keys:
                if self._topic_matches_key(topic, ck):
                    cam_key = ck.strip('/')
                    break
            
            if cam_key:
                timestamps_dict[cam_key] = np.array([t for t, _ in idx_info])
                frame_count_dict[cam_key] = len(idx_info)

        # ---- build nested structure for low-level data only ----
        nested_data = self._build_nested_dict(segment_flat_data)

        # ---- attach metadata at top level ----
        nested_data["timestamps"] = timestamps_dict
        nested_data["frame_count"] = frame_count_dict

        self._current_segment_data = nested_data
        return self._current_segment_data

    def _get_segment_full(self, segment_idx: int, bag_path: Path, ros_version: int) -> Dict[str, Any]:
        """
        Load segment fully into memory - only topics in camera_keys and low_level_keys.

        Camera data is stored with FLAT keys (e.g., 'cam1' directly contains the image array).
        Low-level data uses NESTED structure based on topic paths.
        
        Stores timestamps in a top-level dict:
            self._current_segment_data['timestamps'][<key>] = np.array([...])
        """
        camera_data = {}  # Flat structure for cameras
        low_level_data = defaultdict(list)  # For nested structure
        timestamps = defaultdict(list)  # temporary storage per topic
        skipped_types = set()

        with self._get_reader(bag_path) as reader:
            base_time = reader.start_time

            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic

                # Only process topics from config
                is_camera = any(self._topic_matches_key(topic, ck) for ck in self.camera_keys)
                is_low_level = any(self._topic_matches_key(topic, lk) for lk in self.low_level_keys)

                if not is_camera and not is_low_level:
                    continue

                rel_time = (timestamp - base_time) / 1e9

                try:
                    msg = self._deserialize_message(rawdata, connection.msgtype, ros_version)

                    if is_camera:
                        img = self._decode_image(msg)
                        # Use camera key from config for flat storage
                        cam_key = None
                        for ck in self.camera_keys:
                            if self._topic_matches_key(topic, ck):
                                # Use the config key directly (without leading slash)
                                cam_key = ck.strip('/')
                                break
                        
                        if cam_key:
                            if cam_key not in camera_data:
                                camera_data[cam_key] = []
                            camera_data[cam_key].append(img)
                            timestamps[cam_key].append(rel_time)
                    else:
                        data = self._extract_numeric_data(msg)
                        low_level_data[topic].append(data)
                        # Use the last key of the topic for timestamps
                        last_key = self._topic_to_keys(topic)[-1]
                        timestamps[last_key].append(rel_time)

                except KeyError:
                    if connection.msgtype not in skipped_types:
                        skipped_types.add(connection.msgtype)
                except Exception as e:
                    if topic not in skipped_types:
                        print(f"Warning: Could not process message on {topic}: {e}")
                        skipped_types.add(topic)

        if skipped_types:
            print(f"Skipped unknown/unsupported message types: {skipped_types}")

        # Stack camera data (flat structure)
        for cam_key, img_list in camera_data.items():
            camera_data[cam_key] = self._stack_data(img_list)

        # Stack low-level data into arrays
        stacked_low_level = {}
        for topic, data_list in low_level_data.items():
            if not data_list:
                continue
            self._topic_mapping[topic] = topic
            stacked_low_level[topic] = self._stack_data(data_list)

        # Build nested dict for low-level data only
        nested_data = self._build_nested_dict(stacked_low_level)
        
        # Add camera data at top level (flat)
        nested_data.update(camera_data)

        # Convert timestamps to top-level dict
        timestamps_dict = {key: np.array(ts) for key, ts in timestamps.items()}
        nested_data['timestamps'] = timestamps_dict

        self._current_segment_data = nested_data

        print(self._current_segment_data.keys())
        print(self._current_segment_data["timestamps"].keys())

        return self._current_segment_data


    def get_max_timestamp(self) -> float:
        """
        Return the maximum timestamp for the current segment.

        Returns:
            float: Maximum timestamp in seconds.
        """
        if str(self.current_segment_idx) not in self.segments_info:
            raise ValueError("No segment loaded.")

        return self.segments_info[str(self.current_segment_idx)]['end']

    def get_frame_by_index(self, frame_idx: int, camera_key: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get a specific frame by index (streaming mode).
        
        This method loads only the requested frame from disk, not all frames.
        
        Args:
            frame_idx: Index of the frame to retrieve.
            camera_key: Camera topic key. If None, uses first camera_key from config.
            
        Returns:
            Image as numpy array, or None if not found.
        """
        if self._message_index is None:
            # Not in streaming mode or segment not loaded
            return self.get_camera_frame(camera_key or self.camera_keys[0], frame_idx)
        
        # Normalize camera key
        if camera_key:
            camera_key = camera_key.strip('/')
        else:
            camera_key = self.camera_keys[0].strip('/') if self.camera_keys else None
            
        if camera_key is None:
            return None
        
        # Find the camera topic
        target_topic = None
        for topic in self._message_index:
            if self._topic_matches_key(topic, camera_key):
                target_topic = topic
                break
        
        if target_topic is None:
            return None
            
        # Check frame index is valid
        if frame_idx < 0 or frame_idx >= len(self._message_index.get(target_topic, [])):
            return None
        
        # Check cache first
        cache_key = (camera_key, frame_idx)
        if cache_key in self._frame_cache:
            # Move to end of access order (most recent)
            if cache_key in self._frame_cache_order:
                self._frame_cache_order.remove(cache_key)
            self._frame_cache_order.append(cache_key)
            return self._frame_cache[cache_key]
        
        # Read the specific frame from the bag
        bag_path = self.bag_files[self.current_segment_idx]
        ros_version = self._ros_version or self._detect_ros_version(bag_path)
        
        current_frame = 0
        result = None
        with self._get_reader(bag_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic != target_topic:
                    continue
                    
                if current_frame == frame_idx:
                    try:
                        msg = self._deserialize_message(rawdata, connection.msgtype, ros_version)
                        result = self._decode_image(msg)
                    except Exception as e:
                        print(f"Warning: Could not decode frame {frame_idx}: {e}")
                    break
                        
                current_frame += 1
        
        # Cache the result
        if result is not None:
            self._frame_cache[cache_key] = result
            self._frame_cache_order.append(cache_key)
            
            # Evict old entries if cache is full
            while len(self._frame_cache_order) > self._frame_cache_size:
                old_key = self._frame_cache_order.pop(0)
                self._frame_cache.pop(old_key, None)
        
        return result

    def get_num_frames(self, camera_key: Optional[str] = None) -> int:
        """
        Get the number of frames for a camera topic.
        
        Args:
            camera_key: Camera topic key. If None, uses first camera_key from config.
            
        Returns:
            Number of frames, or 0 if not found.
        """
        if self._current_segment_data is None:
            return 0
        
        key = camera_key.strip('/') if camera_key else (self.camera_keys[0].strip('/') if self.camera_keys else None)
        if key is None:
            return 0
        
        # Check for frame_count in flat structure
        count = self._current_segment_data.get("frame_count", {}).get(key)
        if count is not None:
            return count
        
        # Check if images are loaded directly (flat structure)
        images = self._current_segment_data.get(key)
        if images is not None and isinstance(images, np.ndarray):
            return len(images)
        
        # Check message index
        if self._message_index:
            for topic in self._message_index:
                if self._topic_matches_key(topic, key):
                    return len(self._message_index[topic])
        
        return 0

    def write_annot_data(self, segment_idx: int, annots: Dict[str, Any]):
        """
        Write annotation data to a per-dataset JSON file.

        Annotations are stored using the UID of the segment as the key.

        Args:
            segment_idx: Index of the segment being annotated.
            annots: Dictionary of annotations (must be JSON serializable).
        """
        annotations_path = os.path.join(
            self.annotation_dir,
            f"{self.dataset_name}_annotations.json"
        )

        # Load existing annotations
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                all_annotations = json.load(f)
        else:
            all_annotations = {}

        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        uid = self.segments_info[str(segment_idx)]['uid']
        all_annotations[uid] = convert(annots)

        with open(annotations_path, 'w') as f:
            json.dump(all_annotations, f, indent=2)

    def load_annot_data(self, segment_idx: int) -> Dict[str, Any]:
        """
        Load annotation data for the given segment.

        Args:
            segment_idx: Index of the segment.

        Returns:
            dict: Annotation data, or empty dict if none found.
        """
        uid = self.segments_info[str(segment_idx)]['uid']
        annotations_path = os.path.join(
            self.annotation_dir,
            f"{self.dataset_name}_annotations.json"
        )

        if not os.path.exists(annotations_path):
            return {}

        with open(annotations_path, 'r') as f:
            all_annotations = json.load(f)

        return all_annotations.get(uid, {})

    def get_topics(self, segment_idx: Optional[int] = None) -> List[str]:
        """
        Get list of available topics for a segment.

        Args:
            segment_idx: Segment index. If None, uses current segment.

        Returns:
            List of topic names.
        """
        if segment_idx is None:
            segment_idx = self.current_segment_idx

        if str(segment_idx) not in self.segments_info:
            return []

        return self.segments_info[str(segment_idx)].get('topics', [])

    def get_camera_frame(
        self,
        camera_key: str,
        frame_idx: int,
        segment_idx: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Get a specific frame from a camera topic.

        Args:
            camera_key: Camera key from config (e.g., 'cam1' or '/cam1/image_raw').
            frame_idx: Frame index.
            segment_idx: Segment index. If None, uses current segment.

        Returns:
            Image as numpy array, or None if not found.
        """
        if segment_idx is not None and segment_idx != self.current_segment_idx:
            self.get_segment(segment_idx)

        if self._current_segment_data is None:
            return None
        
        # Normalize camera key (remove leading slash)
        key = camera_key.strip('/')
        
        # Check if images are loaded directly (flat structure)
        images = self._current_segment_data.get(key)
        if images is not None and isinstance(images, np.ndarray) and frame_idx < len(images):
            return images[frame_idx]
        
        # In streaming mode, images may not be loaded - use get_frame_by_index
        if self._stream_mode and self._message_index is not None:
            return self.get_frame_by_index(frame_idx, camera_key)
            
        return None

    def get_frame_at_timestamp(
        self,
        timestamp: float,
        camera_key: Optional[str] = None,
        segment_idx: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Get the image closest to the given timestamp.

        Args:
            timestamp: Target timestamp in seconds.
            camera_key: Specific camera key. If None, uses first camera_key from config.
            segment_idx: Segment index. If None, uses current segment.

        Returns:
            Image as numpy array, or None if not found.
        """
        if segment_idx is not None and segment_idx != self.current_segment_idx:
            self.get_segment(segment_idx)

        if self._current_segment_data is None:
            return None
        
        # Determine which camera key to use and normalize it
        key = camera_key.strip('/') if camera_key else (self.camera_keys[0].strip('/') if self.camera_keys else None)
        if key is None:
            return None
        
        # Get timestamps from flat structure
        ts = self._current_segment_data.get("timestamps", {}).get(key)
        
        if ts is None:
            return None

        # Find closest frame index
        idx = int(np.argmin(np.abs(ts - timestamp)))
        
        # Check if images are loaded (flat structure)
        images = self._current_segment_data.get(key)
        if images is not None and isinstance(images, np.ndarray) and idx < len(images):
            return images[idx]
        
        # Use streaming mode
        if self._stream_mode and self._message_index is not None:
            return self.get_frame_by_index(idx, key)
            
        return None


# Convenience function for quick loading
def load_rosbag_dataset(file_path: str, config: Dict[str, Any]) -> Rosbag:
    """
    Convenience function to quickly load a rosbag dataset from a folder.
    Each bag file in the folder will be treated as a separate segment.

    Args:
        file_path: Path to directory containing bag files.
        config: Configuration dictionary.

    Returns:
        Loaded Rosbag instance.
    """
    dataset = Rosbag(config)
    dataset.load_data(file_path)
    return dataset