from atlas_gui.datasets.dataset import DatasetBase
import os
import json
import numpy as np
import cv2


class Video(DatasetBase):
    """
    Dataset handler for video files (.mp4, .avi, .mkv).

    Supports three structures (auto-detected):
    
    - Single file: a direct file path or folder with one video (1 segment, 1 camera)
    - Folder of videos: each video file = one segment (N segments, 1 camera)
    - Multi-camera: folder of subfolders, each containing multiple video files
      (each subfolder = one segment, each video = one camera, basename = camera key)
    """

    VALID_EXTENSIONS = {'.mp4', '.avi', '.mkv'}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_name = config['dataset_name']
        self.annotation_dir = config['annotation_dir']
        self.current_segment_idx = 0
        self.structure = None  # 'single_file', 'folder', or 'multicam'
        self.segments = {}     # idx -> {'cameras': {cam_key: video_path}}

        os.makedirs(self.annotation_dir, exist_ok=True)

    def _is_video(self, filename):
        return os.path.splitext(filename)[1].lower() in self.VALID_EXTENSIONS

    def _decode_video(self, video_path):
        """Decode an entire video file into a numpy array of frames."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"Could not read any frames from {video_path}")
        return np.stack(frames)

    def _get_video_frame_count(self, video_path):
        """Get number of frames without decoding the entire video."""
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count

    def load_data(self, file_path):
        self.file_path = file_path
        self._detect_structure(file_path)
        self.load_segments_info(file_path)

    def _detect_structure(self, path):
        """Auto-detect the dataset structure and populate self.segments."""
        # Direct file path
        if os.path.isfile(path):
            self.structure = 'single_file'
            cam_key = self.config['camera_keys'][0] if self.config.get('camera_keys') else 'camera'
            if not self.config.get('camera_keys'):
                self.config['camera_keys'] = [cam_key]
            self.segments = {0: {'cameras': {cam_key: path}}}
            return

        # Directory
        entries = sorted(os.listdir(path))
        video_files = [e for e in entries
                       if os.path.isfile(os.path.join(path, e)) and self._is_video(e)]
        subdirs = [e for e in entries
                   if os.path.isdir(os.path.join(path, e))]

        if video_files and not subdirs:
            # Folder of videos: each video = one segment
            self.structure = 'folder'
            cam_key = self.config['camera_keys'][0] if self.config.get('camera_keys') else 'camera'
            if not self.config.get('camera_keys'):
                self.config['camera_keys'] = [cam_key]
            self.segments = {
                i: {'cameras': {cam_key: os.path.join(path, vf)}}
                for i, vf in enumerate(video_files)
            }

        elif subdirs:
            # Multi-camera: subfolders with video files
            self.structure = 'multicam'
            self.segments = {}

            # Auto-detect camera keys from first subfolder if not configured
            first_sub = os.path.join(path, subdirs[0])
            sub_videos = sorted([
                e for e in os.listdir(first_sub)
                if os.path.isfile(os.path.join(first_sub, e)) and self._is_video(e)
            ])

            if not self.config.get('camera_keys'):
                self.config['camera_keys'] = [
                    os.path.splitext(v)[0] for v in sub_videos
                ]

            for i, d in enumerate(subdirs):
                sub_path = os.path.join(path, d)
                cameras = {}
                for vf in sorted(os.listdir(sub_path)):
                    if os.path.isfile(os.path.join(sub_path, vf)) and self._is_video(vf):
                        cam_name = os.path.splitext(vf)[0]
                        cameras[cam_name] = os.path.join(sub_path, vf)
                self.segments[i] = {'cameras': cameras}
        else:
            raise ValueError(
                f"Could not detect dataset structure in '{path}'. "
                "Expected video files or subfolders containing videos."
            )

    def load_segments_info(self, file_path):
        self.segments_info = {}
        for idx, seg_data in self.segments.items():
            first_cam_key = list(seg_data['cameras'].keys())[0]
            video_path = seg_data['cameras'][first_cam_key]
            n_frames = self._get_video_frame_count(video_path)
            duration = (n_frames - 1) / self.config['fps'] if n_frames > 1 else 0.0

            if self.structure == 'multicam':
                text = os.path.basename(os.path.dirname(video_path))
            else:
                text = os.path.splitext(os.path.basename(video_path))[0]

            self.segments_info[str(idx)] = {
                'index': idx,
                'start': 0.0,
                'end': duration,
                'text': text,
                'uid': f"segment_{idx}",
            }

    def get_segment(self, segment_idx):
        self.current_segment_idx = segment_idx
        seg_data = self.segments[segment_idx]
        result = {}

        for cam_key in self.config['camera_keys']:
            if cam_key in seg_data['cameras']:
                result[cam_key] = self._decode_video(seg_data['cameras'][cam_key])

        self.data = result
        return result

    def get_max_timestamp(self):
        return self.segments_info[str(self.current_segment_idx)]['end']

    def write_annot_data(self, segment_idx, annots):
        annotations_path = os.path.join(
            self.annotation_dir, f"{self.dataset_name}_annotations.json"
        )
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

        uid = self.segments_info[str(segment_idx)]['uid']
        all_annotations[uid] = convert(annots)

        with open(annotations_path, "w") as f:
            json.dump(all_annotations, f, indent=2)

    def load_annot_data(self, segment_idx):
        uid = self.segments_info[str(segment_idx)]['uid']
        annotations_path = os.path.join(
            self.annotation_dir, f"{self.dataset_name}_annotations.json"
        )
        if not os.path.exists(annotations_path):
            return {}
        with open(annotations_path, 'r') as f:
            all_annotations = json.load(f)
        return all_annotations.get(uid, {})
