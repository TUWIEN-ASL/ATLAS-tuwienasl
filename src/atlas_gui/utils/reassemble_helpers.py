import h5py
import cv2
import numpy as np
import tempfile

def merge_dict_keys(data_dict):
    """
    Reorganize robot state data by merging related keys into structured arrays.
    
    This function combines separate x, y, z components into single vector arrays
    and organizes joint data into logical groups.
    
    Args:
        data_dict: Dictionary containing robot state data with individual components
        
    Returns:
        Reorganized dictionary with merged components
    """
    # Define which keys to merge and what their new names should be
    merge_map = {
        # "F_ext_base": ["F_ext_base_force_x", "F_ext_base_force_y", "F_ext_base_force_z"],
        # "F_ext_base_torque": ["F_ext_base_torque_x", "F_ext_base_torque_y", "F_ext_base_torque_z"],
        "compensated_base_force": ["compensated_base_force_x", "compensated_base_force_y", "compensated_base_force_z"],
        "compensated_base_torque": ["compensated_base_torque_x", "compensated_base_torque_y", "compensated_base_torque_z"],
        "measured_force": ["measured_force_x", "measured_force_y", "measured_force_z"],
        "measured_torque": ["measured_torque_x", "measured_torque_y", "measured_torque_z"]
    }
    
    # Linear + angular components for pose and velocity (combined into one)
    pose_keys = ["pose_x", "pose_y", "pose_z", "pose_qw", "pose_qx", "pose_qy", "pose_qz"]
    vel_keys = ["vel_x", "vel_y", "vel_z", "vel_qx", "vel_qy", "vel_qz"]

    # Joint positions, velocities, and efforts (combined by joints)
    joint_position_keys = [f"pos_joint{i}" for i in range(1, 10)]  # Joint positions
    joint_velocity_keys = [f"vel_joint{i}" for i in range(1, 10)]  # Joint velocities
    joint_effort_keys    = [f"eff_joint{i}" for i in range(1, 10)]  # Joint efforts

    new_dict = {}
    
    # Merge keys from merge_map (force, torque, etc.)
    for new_key, old_keys in merge_map.items():
        merged_array = np.stack([data_dict[key] for key in old_keys], axis=-1)
        new_dict[new_key] = merged_array
        
        # Remove old keys after merging
        for key in old_keys:
            data_dict.pop(key, None)

    # Handle pose (linear and angular combined into one)
    pose_array = np.stack([data_dict[key] for key in pose_keys], axis=-1)
    new_dict["pose"] = pose_array
    
    # Remove old keys after merging
    for key in pose_keys:
        data_dict.pop(key, None)

    # Handle velocity (linear and angular combined into one)
    vel_array = np.stack([data_dict[key] for key in vel_keys], axis=-1)
    new_dict["velocity"] = vel_array
    
    # Remove old keys after merging
    for key in vel_keys:
        data_dict.pop(key, None)

    # Handle main joint positions (first 7 joints)
    joint_position_array = np.stack([data_dict[key] for key in joint_position_keys[:7]], axis=-1)
    new_dict["joint_positions"] = joint_position_array

    # Handle gripper position (last 2 joints)
    gripper_position_array = np.stack([data_dict[key] for key in joint_position_keys[7:]], axis=-1)
    new_dict["gripper_positions"] = gripper_position_array

    # Remove old keys after merging
    for key in joint_position_keys:
        data_dict.pop(key, None)

    # Handle main joint velocities (first 7 joints)
    joint_velocity_array = np.stack([data_dict[key] for key in joint_velocity_keys[:7]], axis=-1)
    new_dict["joint_velocities"] = joint_velocity_array

    # Handle gripper velocities (last 2 joints)
    gripper_velocity_array = np.stack([data_dict[key] for key in joint_velocity_keys[7:]], axis=-1)
    new_dict["gripper_velocities"] = gripper_velocity_array

    # Remove old keys after merging
    for key in joint_velocity_keys:
        data_dict.pop(key, None)

    # Handle main joint efforts (first 7 joints)
    joint_effort_array = np.stack([data_dict[key] for key in joint_effort_keys[:7]], axis=-1)
    new_dict["joint_efforts"] = joint_effort_array

    # Handle gripper efforts (last 2 joints)
    gripper_effort_array = np.stack([data_dict[key] for key in joint_effort_keys[7:]], axis=-1)
    new_dict["gripper_efforts"] = gripper_effort_array

    # Remove old keys after merging
    for key in joint_effort_keys:
        data_dict.pop(key, None)

    # Add other untouched keys back to the new dictionary
    new_dict.update(data_dict)
    
    return new_dict
    
def load_segments_info(file_path):
    """
    Load only the segments_info from an h5 file.
    
    Args:
        file_path (str): Path to the h5 file
        
    Returns:
        The segments_info data from the h5 file
    """
    def recursively_convert_to_dict(obj):
        if isinstance(obj, h5py.Group):
            return {key: recursively_convert_to_dict(obj[key]) for key in obj.keys()}
        elif isinstance(obj, h5py.Dataset):
            return obj[()]  # Convert dataset to numpy array or other types
        else:
            raise TypeError("Unknown object type")

    # Open the HDF5 file
    with h5py.File(file_path, 'r') as h5_file:
        if 'segments_info' not in h5_file:
            raise KeyError("No segments_info found in the h5 file")
            
        return recursively_convert_to_dict(h5_file['segments_info'])
    
def mp4_blob_to_numpy_interval(binary_blob: bytes, frame_indices: list) -> np.ndarray:
    """
    Convert MP4 binary blob to numpy array, loading only specified frames
    using OpenCV for better performance.
    
    Args:
        binary_blob (bytes): MP4 encoded video as bytes
        frame_indices (list): List of frame indices to extract
    Returns:
        np.ndarray: Array of decoded frames at specified indices
    """
    
    # Sort indices for sequential access
    sorted_indices = sorted(frame_indices)
    
    # Create a temporary file to write the video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as temp_file:
        temp_file.write(binary_blob)
        temp_file.flush()
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(temp_file.name)
        
        frames = []
        current_frame = 0
        last_needed = max(sorted_indices)
        
        for idx in sorted_indices:
            # Skip frames until we reach the desired index
            if idx > current_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                current_frame = idx
            
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB (OpenCV uses BGR by default)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                current_frame += 1
            else:
                break
                
            # Stop if we've processed the last needed frame
            if current_frame > last_needed:
                break
                
        cap.release()
        
    return np.array(frames)



def load_h5_time_interval(file_path, start_time, end_time, skip_no_timestamps=True):
    """
    Load data from h5 file for a specified time interval, maintaining original dictionary structure.
    
    Args:
        file_path (str): Path to the h5 file
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        skip_no_timestamps (bool): If True, skip loading data without corresponding timestamps.
                                 If False, load the entire data for keys without timestamps.
        
    Returns:
        dict: Dictionary containing the data within the specified time interval
    """
    def recursively_convert_to_dict(obj, timestamps_dict=None, time_indices=None):
        if isinstance(obj, h5py.Group):
            result = {}
            for key in obj.keys():
                # Skip data without timestamps if specified
                if skip_no_timestamps and timestamps_dict is not None:
                    # if obj.name == '/' and key not in timestamps_dict.keys(): #key not in ['timestamps']:  # For root level
                    if key not in [*timestamps_dict.keys(), 'robot_state']:
                        continue
                converted = recursively_convert_to_dict(obj[key], timestamps_dict, time_indices)
                if converted is not None:  # Only add non-None results
                    result[key] = converted
            return result
        elif isinstance(obj, h5py.Dataset):
            data = obj[()]
            
            # Check if this data has corresponding timestamps
            # data_key = obj.name[1:]  # Remove leading '/'
            data_key = obj.name.split("/")[-1]
            if timestamps_dict is not None and data_key not in timestamps_dict and skip_no_timestamps:
                if obj.parent.name != '/timestamps':  # Don't skip timestamp arrays themselves
                    return None
            
            # Handle video data
            if isinstance(data, np.void):
                if timestamps_dict is not None and data_key in timestamps_dict:
                    indices = time_indices[data_key]
                    return mp4_blob_to_numpy_interval(data, indices)
                return data if not skip_no_timestamps else None
            # Handle numerical data
            else:
                if timestamps_dict is not None and data_key in timestamps_dict:
                    indices = time_indices[data_key]
                    return data[indices]
                return data if not skip_no_timestamps else None
        else:
            raise TypeError(f"Unknown object type: {type(obj)}")

    def get_time_indices(timestamps, start_time, end_time):
        return np.where((timestamps >= start_time) & (timestamps <= end_time))[0]
    
    with h5py.File(file_path, 'r') as h5_file:
        # First, get all timestamps and calculate indices
        timestamps_dict = {}
        time_indices = {}
        
        for key in h5_file['timestamps'].keys():
            timestamps = h5_file['timestamps'][key][()]
            indices = get_time_indices(timestamps, start_time, end_time)
            if len(indices) > 0:
                timestamps_dict[key] = timestamps[indices]
                time_indices[key] = indices
        
        # Now convert the whole file with time filtering
        data = recursively_convert_to_dict(h5_file, timestamps_dict, time_indices)
        
        # Filter timestamps
        data['timestamps'] = timestamps_dict
        
        return data
    

def save_data_to_h5(file_path, data):
    """
    Save modified data back to the original HDF5 file.

    Args:
        file_path (str): Path to the original HDF5 file.
        data (dict): Modified data dictionary (including additional keys).
    """
    def recursively_save(group, data_dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                # Recursively handle groups (sub-dictionaries)
                if key not in group:
                    subgroup = group.create_group(key)
                else:
                    subgroup = group[key]
                recursively_save(subgroup, value)
            else:
                # If dataset exists, delete & overwrite it
                if key in group:
                    del group[key]
                # Create new dataset
                group.create_dataset(key, data=value)

    with h5py.File(file_path, 'r+') as h5_file:
        recursively_save(h5_file, data)

