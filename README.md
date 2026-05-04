# ATLAS: An Annotation Tool for Long-horizon Robotic Action Segmentation


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
[![GitHub Stars](https://img.shields.io/github/stars/TUWIEN-ASL/ATLAS-tuwienasl.svg)](https://github.com/TUWIEN-ASL/ATLAS-tuwienasl/stargazers)
[![Documentation Status](https://readthedocs.org/projects/atlas-tuwienasl/badge/?version=latest)](https://atlas-tuwienasl.readthedocs.io/en/latest/)
[![GitHub Issues](https://img.shields.io/github/issues/TUWIEN-ASL/ATLAS-tuwienasl.svg)](https://github.com/TUWIEN-ASL/ATLAS-tuwienasl/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/TUWIEN-ASL/ATLAS-tuwienasl.svg)](https://github.com/TUWIEN-ASL/ATLAS-tuwienasl/pulls)
[![arXiv](https://img.shields.io/badge/arXiv-2604.26637-b31b1b.svg)](https://arxiv.org/abs/2604.26637)


## 📋 Abstract
Annotating long-horizon robotic demonstrations with precise temporal action boundaries is crucial for training and evaluating action segmentation and manipulation policy learning methods. Existing annotation tools, however, are often limited: they are designed primarily for vision-only data, do not natively support synchronized visualization of robot-specific time-series signals (e.g., gripper state or force/torque), or require substantial effort to adapt to different dataset formats. In this paper, we introduce ATLAS, an annotation tool tailored for long-horizon robotic action segmentation. ATLAS provides time-synchronized visualization of multi-modal robotic data, including multi-view video and proprioceptive signals, and supports annotation of action boundaries, action labels, and task outcomes. The tool natively handles widely used robotics dataset formats such as ROS bags and the Reinforcement Learning Dataset (RLDS) format, and provides direct support for specific datasets such as REASSEMBLE. ATLAS can be easily extended to new formats via a modular dataset abstraction layer.  Its keyboard-centric interface minimizes annotation effort and improves efficiency. In experiments on a contact-rich assembly task, ATLAS reduced the average per-action annotation time by at least 6% compared to ELAN, while the inclusion of time-series data improved temporal alignment with expert annotations by more than 2.8% and decreased boundary error fivefold compared to vision-only annotation tools.

### ✨ Key Features
- **Segment-Based Video Playback and Annotation**: Supports per-segment (episode) visualization and annotation of multi-camera video data.
- **Annotation Editing and Management**: Extending already existing datasets with an easy and interactive annotation process, visualized in a clear manner. 
- **Flexible Dataset Integration**: Supports ```REASSEMBLE``` and ```RLDS``` dataset types, as well as generic images, videos and ROS bags.
- **Configurable Interface**: The GUI settings easily configurable through the config files (shortcut keys, dataset specifics, etc.)


## 🚀 Getting Started

List of all the prerequisites required to use the project:

```
Python 3.10+
conda (recommended)
```


### Installation for loading data
Step-by-step guide on how to install the GUI.
```bash
# Clone the repository
git clone git@github.com:TUWIEN-ASL/ATLAS-tuwienasl.git

# Navigate to the project directory
cd ATLAS-tuwienasl

# Create and activate a conda environment
conda create -n gui python=3.10
conda activate gui

# Install ATLAS
pip install -e .

# To include optional docs dependencies
pip install -e ".[docs]"
```


## Usage

>**IMPORTANT**  
>Before starting with the annotation, be aware that the annotation using ```REASSEMBLE``` format is done **in-place**, meaning that your original data will get overwritten by your newly annotated data.  


Start the annotation tool by running the following:
```bash
python -m atlas_gui.gui --config config/rlds.yaml
```

(You can look into the ```config``` folder to see the default REASSEMBLE/RLDS config files.)

It will open the GUI, where you should select the data to load.

The overlay of the GUI, from top to bottom, is as follows:
- **Video feeds**
- **Data selector** - Select the data to show in the plots
- **Plots** - Selected plots are plotted here
- **Time slider** - You can manually move across the video with this slider, above it is also the text of the action in the current segment
- **Annotations List** - You can see all loaded as well as your own annotations here, with the ability to adjust their values (start/end times, success) just by clicking on them
- **Buttons** - There is also information about the current segment number, as well as the time which you might find useful if you need to manually fix some annotations

Regarding the buttons, you can see their keyboard shortcuts in the parenthesis, e.g. Start Action **(S)**.

Default shortcuts are:
- **S** - Start/Stop annotating action
- **Q** - Load previous segment
- **E** - Load next segment
- **A** - Go backwards 100ms
- **D** - Go forwards 100ms
- **Z** - Go backwards 10ms
- **C** - Go forwards 10ms
- **F** - Jump to the end of the segment
- **Space** - Play/Pause
- **Backspace** - Delete last (most recent) annotation

If you want different button configuration, change your config file accordingly.

When you annotate a segment, you are presented with a dialog box where you should type in the name of the action and set its success status (True by default). In order to keep your hands on keyboard, after typing in the name you can press Tab to go to the success field, and use Space to check/uncheck it. After setting everything up correctly, press Enter to close the dialog box. Made a mistake? Edit the annotation manually from the annotations list by clicking on it, or press **Backspace** to delete last annotation and try again.

While there is a separate button for saving the annotations you made, they will be automatically saved anyway when you move to the next or the previous segment. 

For more information about the configuration files, please see the [documentation](https://atlas-tuwienasl.readthedocs.io/).



## Using existing RLDS datasets

In order to use existing RLDS datasets, you need to download them first. If you are looking to get a dataset which is a part of the Open X-Embodiment, you may use the following command to download it:

```bash
gsutil -m cp -r gs://gresearch/robotics/<dataset_name> ~/tensorflow_datasets/
```
Replace `<dataset_name>` in the command above with any of the supported names listed in the following dropdown.
<details> <summary><b>Click to view all supported dataset names</b></summary>
     
     'fractal20220817_data'
     'kuka'
     'bridge'
     'taco_play'
     'jaco_play'
     'berkeley_cable_routing'
     'roboturk'
     'nyu_door_opening_surprising_effectiveness'
     'viola'
     'berkeley_autolab_ur5'
     'toto'
     'language_table'
     'columbia_cairlab_pusht_real'
     'stanford_kuka_multimodal_dataset_converted_externally_to_rlds'
     'nyu_rot_dataset_converted_externally_to_rlds'
     'stanford_hydra_dataset_converted_externally_to_rlds'
     'austin_buds_dataset_converted_externally_to_rlds'
     'nyu_franka_play_dataset_converted_externally_to_rlds'
     'maniskill_dataset_converted_externally_to_rlds'
     'furniture_bench_dataset_converted_externally_to_rlds'
     'cmu_franka_exploration_dataset_converted_externally_to_rlds'
     'ucsd_kitchen_dataset_converted_externally_to_rlds'
     'ucsd_pick_and_place_dataset_converted_externally_to_rlds'
     'austin_sailor_dataset_converted_externally_to_rlds'
     'austin_sirius_dataset_converted_externally_to_rlds'
     'bc_z'
     'usc_cloth_sim_converted_externally_to_rlds'
     'utokyo_pr2_opening_fridge_converted_externally_to_rlds'
     'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds'
     'utokyo_saytap_converted_externally_to_rlds'
     'utokyo_xarm_pick_and_place_converted_externally_to_rlds'
     'utokyo_xarm_bimanual_converted_externally_to_rlds'
     'robo_net'
     'berkeley_mvp_converted_externally_to_rlds'
     'berkeley_rpt_converted_externally_to_rlds'
     'kaist_nonprehensile_converted_externally_to_rlds'
     'stanford_mask_vit_converted_externally_to_rlds'
     'tokyo_u_lsmo_converted_externally_to_rlds'
     'dlr_sara_pour_converted_externally_to_rlds'
     'dlr_sara_grid_clamp_converted_externally_to_rlds'
     'dlr_edan_shared_control_converted_externally_to_rlds'
     'asu_table_top_converted_externally_to_rlds'
     'stanford_robocook_converted_externally_to_rlds'
     'eth_agent_affordances'
     'imperialcollege_sawyer_wrist_cam'
     'iamlab_cmu_pickup_insert_converted_externally_to_rlds'
     'qut_dexterous_manipulation'
     'uiuc_d3field'
     'utaustin_mutex'
     'berkeley_fanuc_manipulation'
     'cmu_playing_with_food'
     'cmu_play_fusion'
     'cmu_stretch'
     'berkeley_gnm_recon'
     'berkeley_gnm_cory_hall'
     'berkeley_gnm_sac_son'
     'robot_vqa'
     'droid'
     'conq_hose_manipulation'
     'dobbe'
     'fmb'
     'io_ai_tech'
     'mimic_play'
     'aloha_mobile'
     'robo_set'
     'tidybot'
     'vima_converted_externally_to_rlds'
     'spoc'
     'plex_robosuite'
</details>

This command will download the ```<dataset_name>``` dataset into the ```tensorflow_datasets``` folder in your home directory.


## 📖 Citation
```
@inproceedings{stanovcic2026atlas,
  title     = {{ATLAS: An Annotation Tool for Long-horizon Robotic Action Segmentation}},
  author    = {Stanovcic, Sergej and Sliwowski, Daniel and Lee, Dongheui},
  booktitle = {2026 IEEE International Conference on Advanced Robotics and its Social Impact (ARSO)},
  year      = {2026}
}
```
## 📝 License
This project is released under the MIT License. See [LICENSE.md](LICENSE.md) for details.

