Configuration Guide
===================

This page explains how to configure the ATLAS GUI application through the main configuration file.

Configuration Parameters
------------------------

Dataset Settings
~~~~~~~~~~~~~~~~

**dataset_type**
   The type of dataset to load. Supported formats are:

   * ``REASSEMBLE`` - For datasets using the `REASSEMBLE <https://tuwien-asl.github.io/REASSEMBLE_page/>`_ format
   * ``RLDS`` - For `RLDS <https://github.com/google-research/rlds>`_ (Reinforcement Learning Datasets) format
   * ``rosbag`` - For ROS bag files (``.bag``) with synchronized image and proprioceptive topics
   * ``video`` - For a folder of generic video files (``.mp4``, ``.avi``, etc.)
   * ``frames`` - For a folder of image sequences (``.png``, ``.jpg``)

**dataset_name**
   Name of your dataset, will be used during logging of annotations. This helps identify which dataset was used when reviewing annotation logs.

**fps**
   Frequency at which the data was recorded and/or at which the data will be synced to. This ensures proper temporal alignment of data streams.

**annotation_dir**
   Directory where the JSON annotation files will be saved.

   .. note::
      This applies only when annotations are saved externally as JSON files.
      ``REASSEMBLE`` datasets configured with ``annotation_storage: h5`` write
      annotations directly back into the source HDF5 file and do not use this
      directory.

Data Configuration
~~~~~~~~~~~~~~~~~~

The data-related keys behave slightly differently depending on the chosen
``dataset_type``. Select the relevant tab below. For complete, ready-to-use
configuration files for each format, see :ref:`complete-configuration-example`.

.. tab-set::

   .. tab-item:: REASSEMBLE

      **low_level_keys**
         List of proprioceptive robot data keys. You should list all available sensor keys here that you want to access during annotation.

         .. important::
            You should include the full dictionary path to your desired keys.

         Example:

         .. code-block:: yaml

            low_level_keys:
              - steps/observation/gripper_position
              - steps/observation/cartesian_position
              - steps/observation/joint_position

      **camera_keys**
         List the camera keys that you wish to display in the GUI interface.

         .. important::
            You should include the full dictionary path to your desired keys.

         Example:

         .. code-block:: yaml

            camera_keys:
              - steps/observation/wrist_image_left
              - steps/observation/exterior_image_1_left
              - steps/observation/exterior_image_2_left

      **color_format**
         Image color format, either ``"RGB"`` or ``"BGR"``. Specifies the format of the camera data to ensure proper color display.

      **annotation_storage**
         How annotations are persisted. Either ``"h5"`` (default) or ``"json"``.

         * ``h5`` — annotations are written directly back into the source HDF5 file under the group specified by ``annotation_group``.
         * ``json`` — annotations are written to an external JSON file inside ``annotation_dir``, leaving the source data untouched.

         .. code-block:: yaml

            annotation_storage: json

      **annotation_group**
         Name of the HDF5 group under which annotations are stored when ``annotation_storage: h5``. Defaults to ``low_level``.

         .. code-block:: yaml

            annotation_group: low_level

   .. tab-item:: RLDS

      **low_level_keys**
         List of proprioceptive robot data keys. You should list all available sensor keys here that you want to access during annotation.

         .. important::
            You should include the full dictionary path to your desired keys.

         Example:

         .. code-block:: yaml

            low_level_keys:
              - steps/observation/gripper_position
              - steps/observation/cartesian_position
              - steps/observation/joint_position

      **camera_keys**
         List the camera keys that you wish to display in the GUI interface.

         .. important::
            You should include the full dictionary path to your desired keys.

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

         Example:

         .. code-block:: yaml

            text_keys:
              - language_instruction

   .. tab-item:: rosbag

      **low_level_keys**
         List of proprioceptive ROS topics whose messages will be made available during annotation.

         Example:

         .. code-block:: yaml

            low_level_keys:
              - franka_state_controller/O_T_EE
              - joint_states
              - ft_sensor/ft_compensated_base

      **camera_keys**
         List of ROS image topics that you wish to display in the GUI interface.

         Example:

         .. code-block:: yaml

            camera_keys:
              - cam1
              - cam2

      **color_format**
         Image color format, either ``"RGB"`` or ``"BGR"``. Specifies the format of the camera data to ensure proper color display.

      **annotation_group**
         Name of the dictionary entry under which annotations will be saved within the bag's metadata. Defaults to ``annotations``.

         Example:

         .. code-block:: yaml

            annotation_group: high_level

      **stream_mode**
         If ``True`` (default), camera frames are loaded on-demand from the bag to keep memory usage low. Set to ``False`` to load all frames into memory at startup — playback will be faster but RAM usage will be much higher.

         .. code-block:: yaml

            stream_mode: true

      **frame_cache_size**
         Number of recently accessed frames to keep in memory while in streaming mode. Larger values smooth out playback at the cost of more RAM. Defaults to ``30``.

         .. code-block:: yaml

            frame_cache_size: 60

   .. tab-item:: video

      **low_level_keys**
         Leave empty — generic video files do not contain proprioceptive data.

         .. code-block:: yaml

            low_level_keys:

      **camera_keys**
         Leave empty to auto-detect camera streams from the folder structure, or provide a list of explicit camera names if you have a known multi-camera layout.

         .. code-block:: yaml

            camera_keys:
              # - camera           # uncomment for a single named camera

      **color_format**
         Image color format, either ``"RGB"`` or ``"BGR"``. Specifies the format of the camera data to ensure proper color display.

   .. tab-item:: frames

      **low_level_keys**
         Leave empty — image folders do not contain proprioceptive data.

         .. code-block:: yaml

            low_level_keys:

      **camera_keys**
         Leave empty to auto-detect camera streams from the folder structure, or provide a list of explicit camera names if you have a known multi-camera layout.

         .. code-block:: yaml

            camera_keys:
              # - camera           # uncomment for a single named camera
              # - left_cam         # uncomment and add entries for a known multi-camera setup
              # - right_cam

      **color_format**
         Image color format, either ``"RGB"`` or ``"BGR"``. Specifies the format of the camera data to ensure proper color display.

Display Settings
~~~~~~~~~~~~~~~~

**default_graphs**
   Proprioceptive data that you want to have displayed by default each time you load a segment.
   This allows you to automatically show the most relevant sensor data without manual selection.

   .. note::
      This setting has no effect for the ``video`` and ``frames`` dataset types,
      since those formats do not carry proprioceptive data. Leave it empty in
      that case.

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


**ff_value_big**
   Step size in seconds for the ``fast_forward`` / ``rewind`` keys (large jumps).

   Example:

   .. code-block:: yaml

      ff_value_big: 0.1   # 100 ms per press

**ff_value_small**
   Step size in seconds for the ``fast_forward_small`` / ``rewind_small`` keys (fine-grained jumps).

   Example:

   .. code-block:: yaml

      ff_value_small: 0.01   # 10 ms per press

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

.. _complete-configuration-example:

Complete Configuration Example
------------------------------

Below is a complete example configuration file for each supported dataset format.
Select the relevant tab to see the full template you can drop into your own setup.

.. tip::
   Each of the example files shown here is also shipped in the
   :gh-tree:`config/ <config>` folder of the repository, so you can copy one
   and edit it directly instead of starting from scratch.

.. tab-set::

   .. tab-item:: REASSEMBLE

      :gh-blob:`View on GitHub <config/reassemble.yaml>`

      .. literalinclude:: ../config/reassemble.yaml
         :language: yaml
         :caption: config/reassemble.yaml

   .. tab-item:: RLDS

      :gh-blob:`View on GitHub <config/rlds.yaml>`

      .. literalinclude:: ../config/rlds.yaml
         :language: yaml
         :caption: config/rlds.yaml

   .. tab-item:: rosbag

      :gh-blob:`View on GitHub <config/rosbag.yaml>`

      .. literalinclude:: ../config/rosbag.yaml
         :language: yaml
         :caption: config/rosbag.yaml

   .. tab-item:: video

      :gh-blob:`View on GitHub <config/video.yaml>`

      .. literalinclude:: ../config/video.yaml
         :language: yaml
         :caption: config/video.yaml

   .. tab-item:: frames

      :gh-blob:`View on GitHub <config/frames.yaml>`

      .. literalinclude:: ../config/frames.yaml
         :language: yaml
         :caption: config/frames.yaml

See Also
--------

* :doc:`modules` - API Documentation
* :doc:`atlas_gui` - Package Overview