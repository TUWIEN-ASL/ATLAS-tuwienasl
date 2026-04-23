from atlas_gui.datasets.dataset import DatasetBase
import os
import json
import numpy as np
import cv2


class Frames(DatasetBase):
    """
    Dataset handler for image frame sequences stored in folders.

    Supports three folder structures (auto-detected):
    
    - Flat: images directly in the selected folder (single segment, single camera)
    - Subfolder: subfolders each containing images (each subfolder = one segment, single camera)
    - Multi-camera: subfolders containing camera subfolders with images
      (each top-level subfolder = one segment, camera subfolders = cameras)
    """

    VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_name = config['dataset_name']
        self.annotation_dir = config['annotation_dir']
        self.current_segment_idx = 0
        self.structure = None  # 'flat', 'subfolder', or 'multicam'
        self.segments = {}     # idx -> segment directory path

        os.makedirs(self.annotation_dir, exist_ok=True)

    def _is_image(self, filename):
        return os.path.splitext(filename)[1].lower() in self.VALID_EXTENSIONS

    def _list_images(self, folder_path):
        """Return sorted list of image file paths in a folder."""
        return sorted([
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and self._is_image(f)
        ])

    def load_data(self, file_path):
        self.file_path = file_path
        self._detect_structure(file_path)
        self.load_segments_info(file_path)

    def _detect_structure(self, root):
        """Auto-detect the folder structure and populate self.segments."""
        entries = sorted(os.listdir(root))
        subdirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
        images = [e for e in entries
                  if os.path.isfile(os.path.join(root, e)) and self._is_image(e)]

        if images and not subdirs:
            # Flat: images directly in root folder
            self.structure = 'flat'
            self.segments = {0: root}
            if not self.config.get('camera_keys'):
                self.config['camera_keys'] = ['camera']

        elif subdirs:
            first_sub = os.path.join(root, subdirs[0])
            sub_entries = os.listdir(first_sub)
            sub_subdirs = [e for e in sub_entries
                           if os.path.isdir(os.path.join(first_sub, e))]
            sub_images = [e for e in sub_entries
                          if os.path.isfile(os.path.join(first_sub, e))
                          and self._is_image(e)]

            if sub_images and not sub_subdirs:
                # Subfolder: each subfolder is a segment with images directly inside
                self.structure = 'subfolder'
                self.segments = {
                    i: os.path.join(root, d) for i, d in enumerate(subdirs)
                }
                if not self.config.get('camera_keys'):
                    self.config['camera_keys'] = ['camera']

            elif sub_subdirs:
                # Multi-camera: subfolders contain camera subfolders
                self.structure = 'multicam'
                self.segments = {
                    i: os.path.join(root, d) for i, d in enumerate(subdirs)
                }
                if not self.config.get('camera_keys'):
                    self.config['camera_keys'] = sorted(sub_subdirs)
        else:
            raise ValueError(
                f"Could not detect folder structure in '{root}'. "
                "Expected images or subfolders containing images."
            )

    def load_segments_info(self, file_path):
        self.segments_info = {}
        for idx, segment_path in self.segments.items():
            if self.structure == 'multicam':
                first_cam = self.config['camera_keys'][0]
                cam_path = os.path.join(segment_path, first_cam)
                n_frames = len(self._list_images(cam_path))
            else:
                n_frames = len(self._list_images(segment_path))

            duration = (n_frames - 1) / self.config['fps'] if n_frames > 1 else 0.0

            self.segments_info[str(idx)] = {
                'index': idx,
                'start': 0.0,
                'end': duration,
                'text': os.path.basename(segment_path) if self.structure != 'flat'
                        else os.path.basename(file_path),
                'uid': f"segment_{idx}",
            }

    def get_segment(self, segment_idx):
        self.current_segment_idx = segment_idx
        segment_path = self.segments[segment_idx]
        result = {}

        if self.structure in ('flat', 'subfolder'):
            cam_key = self.config['camera_keys'][0]
            image_paths = self._list_images(segment_path)
            frames = [cv2.imread(p) for p in image_paths]
            result[cam_key] = np.stack(frames)
        elif self.structure == 'multicam':
            for cam_key in self.config['camera_keys']:
                cam_path = os.path.join(segment_path, cam_key)
                image_paths = self._list_images(cam_path)
                frames = [cv2.imread(p) for p in image_paths]
                result[cam_key] = np.stack(frames)

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
