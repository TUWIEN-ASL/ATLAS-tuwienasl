Configuration Guide
===================

This page explains how to configure the ATLAS GUI application through the main configuration file.

Configuration Parameters
------------------------

Dataset Settings
~~~~~~~~~~~~~~~~

**dataset_type**
   Currently there are only 2 types of dataset types supported: ``REASSEMBLE`` and ``RLDS``.
   
   * ``REASSEMBLE`` - For datasets using the `REASSEMBLE <https://tuwien-asl.github.io/REASSEMBLE_page/>`_ format 
   * ``RLDS`` - For `RLDS <https://github.com/google-research/rlds>`_ (Reinforcement Learning Datasets) format 

**dataset_name**
   Name of your dataset, will be used during logging of annotations. This helps identify which dataset was used when reviewing annotation logs.

**fps**
   Frequency at which the data was recorded and/or at which the data will be synced to. This ensures proper temporal alignment of data streams.

**annotation_dir**
   Directory where the JSON annotation files will be saved.
   
   .. note::
      This applies only to datasets which save data externally, not to datasets like ``REASSEMBLE`` which will overwrite the original data.

Data Configuration
~~~~~~~~~~~~~~~~~~

**low_level_keys**
   List of proprioceptive robot data keys. You should list all available sensor keys here that you want to access during annotation.
   
   .. important::
      You should include full dictionary path to your desired keys.

   Example:
   
   .. code-block:: yaml
   
      low_level_keys:
        - steps/observation/gripper_position
        - steps/observation/cartesian_position
        - steps/observation/joint_position

**camera_keys**
   List the camera keys that you wish to display in the GUI interface.
   
   Example:
   
   .. code-block:: yaml
   
      camera_keys:
        - steps/observation/wrist_image_left
        - steps/observation/exterior_image_1_left
        - steps/observation/exterior_image_2_left

**color_format**
   Image color format, either ``"RGB"`` or ``"BGR"``. Specifies the format of the camera data to ensure proper color display.

**text_keys**
   Which text descriptions to load for a given segment.
   
   .. note::
      This key applies only to RLDS-like datasets that contain textual descriptions.

Display Settings
~~~~~~~~~~~~~~~~

**default_graphs**
   Proprioceptive data that you want to have displayed by default each time you load a segment.
   This allows you to automatically show the most relevant sensor data without manual selection.
   
   Example:
   
   .. code-block:: yaml
   
      default_graphs:
        - steps/observation/cartesian_position
        - steps/observation/gripper_position

Keyboard Configuration
~~~~~~~~~~~~~~~~~~~~~~

**keys**
   Keyboard shortcuts for annotation actions. Define custom key bindings for common annotation operations.
   
   Example:
   
   .. code-block:: yaml
   
      keys:
         play: Space
         previous_segment: Q
         next_segment: E
         toggle_annotation: S
         delete_last_annotation: Backspace
         save_segment_annotation: P
         fast_forward: D
         rewind: A
         fast_forward_small: C
         rewind_small: Z
         jump_to_end: F


**action_map**
   Keyboard number shortcuts for inputting the name of the segmented part during annotation. This allows quick selection of annotation categories using number keys.
   
   Example:
   
   .. code-block:: yaml
   
      action_map:
         1: Approach
         2: Grasp
         3: Lift
         4: Release
         5: Align
         6: Push
         7: Pull
         8: Nudge
         9: Twist

Complete Configuration Example
------------------------------

Here's a complete example configuration file:

.. code-block:: yaml

   # Dataset configuration
   dataset_type: "rlds"
   dataset_name: "robot_manipulation_v1"
   fps: 30
   annotation_dir: "/path/to/annotations"
   
   # Data keys
   low_level_keys:
     - steps/observation/gripper_position
     - steps/observation/cartesian_position
     - steps/observation/joint_position

   camera_keys:
     - steps/observation/wrist_image_left
     - steps/observation/exterior_image_1_left
     - steps/observation/exterior_image_2_left
  
   color_format: "RGB"
      
   text_keys:
     - language_instruction
   
   # Display settings
   default_graphs:
     - steps/observation/cartesian_position
     - steps/observation/gripper_position 
   
   # Keyboard shortcuts
   keys:
      play: Space
      previous_segment: Q
      next_segment: E
      toggle_annotation: S
      delete_last_annotation: Backspace
      save_segment_annotation: P
      fast_forward: D
      rewind: A
      fast_forward_small: C
      rewind_small: Z
      jump_to_end: F

   ff_value_big: 0.1
   ff_value_small: 0.01
   
   action_map:
      1: Approach
      2: Grasp
      3: Lift
      4: Release
      5: Align
      6: Push
      7: Pull
      8: Nudge
      9: Twist

See Also
--------

* :doc:`modules` - API Documentation
* :doc:`atlas_gui` - Package Overview