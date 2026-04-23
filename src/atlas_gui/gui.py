import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QFileDialog,
                           QLineEdit, QScrollArea, QSlider, QComboBox,
                           QMenu, QDoubleSpinBox, QCheckBox, QGridLayout,
                           QMessageBox, QStatusBar, QInputDialog ,QSizePolicy,
                           QSpacerItem, QDialog, QSplitter)
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer, QElapsedTimer
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QFont, QKeyEvent
import matplotlib.pyplot as plt
from copy import deepcopy
import pyqtgraph as pg

from atlas_gui.utils.config import load_config, get_nested, has_nested_key, create_dataset
from atlas_gui.datasets.reassemble import Reassemble
from atlas_gui.utils.keys import load_key_bindings_from_config, load_action_map_from_config
import argparse

SUCCESS_KEY = Qt.Key_Space



class ActionDialog(QDialog):
    """
    A custom Qt dialog for selecting an action label and success flag during annotation.

    This dialog is used in ATLAS to collect a user-defined label and 
    a success boolean for a low-level action during video annotation.

    Features:
    - Displays a key-action map to guide the user (e.g., 1: Approach, 2: Grasp, etc.).
    - Allows manual input of an action name.
    - Lets the user toggle a success flag via a checkbox or Space key.
    - Returns the chosen action name and success state when accepted.

    Args:
        parent (QWidget, optional): Parent widget. Defaults to None.
        action_map (dict, optional): A mapping from Qt key codes to action label strings.

    Key Bindings:
        - Number keys (1–9): Set corresponding action label based on `action_map`.
        - Space key: Toggle success flag checkbox.
    """
    def __init__(self, parent=None, action_map=None):
        super().__init__(parent)
        self.action_map = action_map
        self.setWindowTitle("Enter Action Name and Success Flag")

        self.layout = QVBoxLayout(self)

        # Predefined action mappings
        # self.action_map = load_action_map_from_config(config["action_map"])
        self.success_key = SUCCESS_KEY

        # Create instruction text from action_map
        action_text = "<b>Press a number to select an action:</b><br>" + "<br>".join(
            [f"{Qt.Key(key) - Qt.Key_0}: {name}" for key, name in self.action_map.items()]
        )

        # Add a label to display key bindings
        self.info_label = QLabel(action_text, self)
        self.info_label.setWordWrap(True)  # Ensure proper wrapping
        self.layout.addWidget(self.info_label)

        # Action name field
        self.action_name_label = QLabel("Action Name:", self)
        self.action_name_input = QLineEdit(self)
        self.layout.addWidget(self.action_name_label)
        self.layout.addWidget(self.action_name_input)

        # Success flag checkbox (default to checked)
        self.success_flag_label = QLabel("Success Flag (check for True):", self)
        self.success_flag_checkbox = QCheckBox(self)
        self.success_flag_checkbox.setChecked(True)
        self.layout.addWidget(self.success_flag_label)
        self.layout.addWidget(self.success_flag_checkbox)

        # OK button
        self.ok_button = QPushButton("OK (Enter)", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

        # Remove focus from text box so key presses trigger keyPressEvent
        self.setFocus()

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handle key press events to enable quick interaction.

        - If a key in `action_map` is pressed, its corresponding action label is populated.
        - If the space key is pressed, the success checkbox is toggled.
        - Other keys are passed to the base class implementation.

        Args:
            event (QKeyEvent): The Qt key event.
        """
        key = event.key()
        if key in self.action_map:
            self.action_name_input.setText(self.action_map[key])  # Set the action name
        elif key == self.success_key:  # Space key
            self.success_flag_checkbox.setChecked(not self.success_flag_checkbox.isChecked())  # Toggle checkbox
        else:
            super().keyPressEvent(event)  # Handle other key events normally

    def get_input(self):
        """
        Retrieve the user input from the dialog.

        Returns:
            Tuple[str, bool]: A tuple containing:
                - action_name: The selected or manually entered action name.
                - success: Whether the action is marked as successful.
        """
        action_name = self.action_name_input.text().strip()
        success = self.success_flag_checkbox.isChecked()  

        # Default action if none was selected
        if not action_name:
            action_name = "Transition"

        return action_name, success


class EditableAnnotationWidget(QWidget):
    """
    A widget that allows editing an individual annotation.

    Provides UI elements for editing the label, start time, end time, and success flag,
    as well as buttons to save, cancel, or delete the annotation.

    Args:
        annotation_id (int): Unique identifier for the annotation.
        annotation (dict): Annotation data containing 'label', 'start', 'end', 'success', and 'color'.
        parent (QWidget, optional): Parent widget.
    """
    def __init__(self, annotation_id, annotation, parent=None):
        super().__init__(parent)
        self.annotation_id = annotation_id
        self.annotation = annotation
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create input fields
        self.label_input = QLineEdit(annotation['label'])

        self.start_time = QDoubleSpinBox()
        self.start_time.setDecimals(3)
        self.start_time.setRange(0, 999999)
        self.start_time.setValue(annotation['start'])

        self.end_time = QDoubleSpinBox()
        self.end_time.setDecimals(3)
        self.end_time.setRange(0, 999999)
        self.end_time.setValue(annotation['end'])

        self.success_checkbox = QCheckBox("Success")
        self.success_checkbox.setChecked(bool(annotation.get("success", False)))
        
        # Add fields to layout
        layout.addWidget(QLabel("Label:"))
        layout.addWidget(self.label_input)
        layout.addWidget(QLabel("Start:"))
        layout.addWidget(self.start_time)
        layout.addWidget(QLabel("End:"))
        layout.addWidget(self.end_time)
        layout.addWidget(self.success_checkbox)
        
        # Add save/cancel buttons
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_changes)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancel_changes)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self.delete_annotation)
        
        layout.addWidget(save_btn)
        layout.addWidget(cancel_btn)
        layout.addWidget(delete_btn)
        
        # Set widget color
        self.setStyleSheet(f"background-color: rgba({int(annotation['color'][0]*255)}, "
                          f"{int(annotation['color'][1]*255)}, "
                          f"{int(annotation['color'][2]*255)}, 0.3)")
    
    def save_changes(self):
        """
        Save the modified annotation values and propagate the update
        to the parent `SegmentedVideoAnnotator` widget.
        """
        # Find the VideoAnnotator instance by traversing up the widget hierarchy
        widget = self
        while widget is not None:
            if isinstance(widget, SegmentedVideoAnnotator):
                widget.update_annotation(self.annotation_id, {
                    'label': self.label_input.text(),
                    'start': self.start_time.value(),
                    'end': self.end_time.value(),
                    'success': self.success_checkbox.isChecked(), #self.annotation['success'],
                    'color': self.annotation['color'],
                })
                break
            widget = widget.parent()
    
    def cancel_changes(self):
        """
        Cancel editing and refresh the timeline to discard changes.
        """
        widget = self
        while widget is not None:
            if isinstance(widget, SegmentedVideoAnnotator):
                widget.update_timeline()
                break
            widget = widget.parent()
    
    def delete_annotation(self):
        """
        Delete the current annotation via the parent `SegmentedVideoAnnotator` widget.
        """
        widget = self
        while widget is not None:
            if isinstance(widget, SegmentedVideoAnnotator):
                widget.delete_annotation(self.annotation_id)
                break
            widget = widget.parent()


class SegmentedVideoAnnotator(QMainWindow):
    """
    Main GUI class for interactive annotation of segmented video data.

    Provides tools to:
    - Visualize synchronized video and sensor streams.
    - Navigate and annotate segments.
    - Interact with multiple data formats (e.g., REASSEMBLE, RLDS).
    - Save and load annotation metadata.

    Args:
        config (dict): Configuration dictionary specifying keys, dataset type,
                       camera keys, key bindings, and visual settings.
    """
    def __init__(self, config):
        """
        Initializes the main GUI window and sets up the application state and layout.

        Args:
            config (dict): Configuration for dataset, display, keys, and behavior.
        """
        super().__init__()

        self.config = config
        # Normalize optional list config values (YAML parses empty values as None)
        if self.config.get('camera_keys') is None:
            self.config['camera_keys'] = []
        if self.config.get('low_level_keys') is None:
            self.config['low_level_keys'] = []
        self.key_bindings = load_key_bindings_from_config(config["keys"])
        self.action_map = load_action_map_from_config(config["action_map"])

        self.setWindowTitle("ATLAS")
        self.setGeometry(100, 100, 1400, 1000)
        
        # Data storage
        self.data = None
        self.current_time = 0
        self.playing = False
        self.min_timestamp = 0
        self.max_timestamp = 0
        self.timestamp_offset = 0
        self.dataset = create_dataset(dataset_type=self.config['dataset_type'],
                                       config=config)
        
        # timestamps for fixed-frequency data
        self.fixed_timestamps = None

        # Segment-specific data
        self.segments_info = {}
        self.current_segment_index = 0
        self.file_path = None
        
        # Current data selection (None = use defaults, [] = user cleared all)
        self.selected_numerical_data = None
        self.plot_figures = []
        self.plot_canvases = []
        
        # Annotations storage
        self.annotations = {}
        self.current_annotation_id = 0
        self.recording_annotation = False
        self.current_annotation_start = None
        self.used_colors = set()
        self.tab10_colors = plt.get_cmap("tab10").colors
        self.color_format = self.config.get('color_format', 'BGR')  # default to BGR
        
        # Create main widget and layout
        self.setup_ui()
        # self.setChildrenFocusPolicy(QtCore.Qt.NoFocus)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Setup timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.elapsed_timer = QElapsedTimer()


        # dark_stylesheet = """
        #     QWidget {
        #         background-color: #121212;
        #         color: #ffffff;
        #     }
        #     QLabel {
        #         color: #ffffff;
        #     }
        #     QPushButton {
        #         background-color: #333;
        #         color: white;
        #         border-radius: 5px;
        #         padding: 5px;
        #     }
        #     QPushButton:hover {
        #         background-color: #444;
        #     }
        #     QSlider::groove:horizontal {
        #         background: #555;
        #         height: 8px;
        #         border-radius: 4px;
        #     }
        #     QSlider::handle:horizontal {
        #         background: #aaa;
        #         width: 14px;
        #         border-radius: 7px;
        #     }
        #     QScrollArea {
        #         background: #222;
        #     }
        # """
        # self.setStyleSheet(dark_stylesheet)

    def setup_ui(self):
        """
        Constructs the full user interface, including:
        - Video displays for multiple cameras.
        - Data selector for numeric streams.
        - Plots, timeline slider, and action label.
        - Scrollable timeline widget.
        - Control buttons and status bar.
        """
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create main vertical splitter for resizable sections
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.splitterMoved.connect(self._on_splitter_moved)

        # === TOP SECTION: Video displays ===
        video_widget = QWidget()
        self.video_layout = QHBoxLayout(video_widget)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_labels = []

        for _ in range(len(self.config['camera_keys'])):
            label = QLabel()
            label.setMinimumSize(200, 150)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid black")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.video_labels.append(label)
            self.video_layout.addWidget(label)
        main_splitter.addWidget(video_widget)

        # === MIDDLE SECTION: Selector + Plots ===
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)

        # Data selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select Data:"))

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        self.data_selector_scroll = QScrollArea()
        self.data_selector_scroll.setWidget(scroll_widget)
        self.data_selector_scroll.setWidgetResizable(True)
        self.data_selector_scroll.setMaximumHeight(100)

        self.data_selectors = []
        selector_layout.addWidget(self.data_selector_scroll)
        middle_layout.addLayout(selector_layout)

        # Plots container
        self.plots_widget = QWidget()
        self.plots_grid = QGridLayout()
        self.plots_widget.setLayout(self.plots_grid)
        self.plots_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        middle_layout.addWidget(self.plots_widget)

        main_splitter.addWidget(middle_widget)

        # === BOTTOM SECTION: Timeline, annotations, controls ===
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Action text label
        self.action_label = QLabel("Current Action: None")
        self.action_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        self.action_label.setFont(font)
        bottom_layout.addWidget(self.action_label)

        # Timeline slider
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(1000)
        self.timeline_slider.valueChanged.connect(self.slider_changed)
        bottom_layout.addWidget(self.timeline_slider)

        # Annotation overview bar (visual representation of annotated regions)
        self.annotation_bar = pg.PlotWidget()
        self.annotation_bar.setBackground('white')
        self.annotation_bar.setFixedHeight(40)
        self.annotation_bar.hideAxis('left')
        self.annotation_bar.hideAxis('bottom')
        self.annotation_bar.setMouseEnabled(x=False, y=False)
        self.annotation_bar.getViewBox().setDefaultPadding(0)
        self.annotation_bar.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.annotation_bar_vline = pg.InfiniteLine(angle=90, movable=False, pen='k')
        self.annotation_bar.addItem(self.annotation_bar_vline)
        self.annotation_bar_regions = []
        bottom_layout.addWidget(self.annotation_bar)

        # Timeline/annotations scroll area
        self.timeline_widget = QWidget()
        self.timeline_layout = QVBoxLayout(self.timeline_widget)
        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidget(self.timeline_widget)
        self.timeline_scroll.setWidgetResizable(True)
        bottom_layout.addWidget(self.timeline_scroll)

        # Controls
        self.setup_controls(bottom_layout)
        self.setup_segment_controls(bottom_layout)

        main_splitter.addWidget(bottom_widget)

        # Style splitter handles for visibility
        main_splitter.setHandleWidth(6)
        main_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #cccccc; }"
        )

        # Set initial sizes (video, middle, bottom)
        main_splitter.setSizes([250, 300, 200])

        layout.addWidget(main_splitter)

        # Set up the status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def setup_controls(self, layout):
        """
        Adds core controls to the GUI for:
        - Loading data.
        - Playback (play/pause).
        - Displaying current time.
        - Starting, ending, and deleting annotations.

        Args:
            layout (QLayout): Layout object to which the controls are added.
        """
        control_layout = QHBoxLayout()
        
        # Load data button
        load_data_btn = QPushButton("Load Data")
        load_data_btn.clicked.connect(self.load_data)
        control_layout.addWidget(load_data_btn)
        
        # Playback controls
        self.play_button = QPushButton(f"Play ({key_to_string(self.key_bindings['play'])})")
        self.play_button.clicked.connect(self.toggle_playback)
        control_layout.addWidget(self.play_button)
        
        # Current time display
        self.time_label = QLabel("Time: 0.000 s")
        control_layout.addWidget(self.time_label)
        
        # Annotation controls
        # self.annotation_input = QLineEdit()
        # self.annotation_input.setPlaceholderText("Enter action label")
        # control_layout.addWidget(self.annotation_input)
        
        # Start/End annotation button
        self.annotation_button = QPushButton(f"Start Action ({key_to_string(self.key_bindings['toggle_annotation'])})")
        self.annotation_button.clicked.connect(self.toggle_annotation)
        control_layout.addWidget(self.annotation_button)
        
        # Delete last annotation button
        delete_last_btn = QPushButton(f"Delete Last ({key_to_string(self.key_bindings['delete_last_annotation'])})")
        delete_last_btn.clicked.connect(self.delete_last_annotation)
        control_layout.addWidget(delete_last_btn)

        layout.addLayout(control_layout)

    def setup_segment_controls(self, layout):
        """
        Adds segment-level controls to the GUI for:
        - Navigating between segments (previous, next).
        - Jumping to a segment by index.
        - Saving annotations for the current segment.

        Args:
            layout (QLayout): Layout object to which the controls are added.
        """
        segment_layout = QHBoxLayout()
        
        # Add segment navigation controls
        prev_segment_btn = QPushButton(f"Previous Segment ({key_to_string(self.key_bindings['previous_segment'])})")
        prev_segment_btn.clicked.connect(self.load_previous_segment)
        next_segment_btn = QPushButton(f"Next Segment ({key_to_string(self.key_bindings['next_segment'])})")
        next_segment_btn.clicked.connect(self.load_next_segment)
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("Enter segment index")
        jump_btn = QPushButton("Jump to Segment")
        jump_btn.clicked.connect(self.jump_to_segment)
        
        # Add segment info display
        self.segment_info_label = QLabel("Segment: 0/0")
        
        # Add save segment annotations button
        save_segment_btn = QPushButton(f"Save Segment ({key_to_string(self.key_bindings['save_segment_annotation'])})")
        save_segment_btn.clicked.connect(self.save_segment_annotations)
    
        
        # Add controls to layout
        segment_layout.addWidget(prev_segment_btn)
        segment_layout.addWidget(self.segment_info_label)
        segment_layout.addWidget(next_segment_btn)
        segment_layout.addWidget(save_segment_btn)
        segment_layout.addWidget(self.jump_input)
        segment_layout.addWidget(jump_btn)
        
        layout.addLayout(segment_layout)
    
    def create_annotation_widget(self, annotation_id, annotation):
        """
        Create a QPushButton widget to visually represent an annotation in the timeline.

        Args:
            annotation_id (int): Unique identifier for the annotation.
            annotation (dict): Annotation data including 'start', 'end', 'label', 'success', and 'color'.

        Returns:
            QPushButton: A clickable and context-sensitive button widget for the annotation.
        """
        success_symbol = "✅" if annotation.get("success", False) else "❌"
        display_widget = QPushButton(
            f"{annotation['start']:.3f}s - {annotation['end']:.3f}s: {annotation['label']} | {success_symbol}"
        )
        display_widget.setStyleSheet(f"background-color: rgba({int(annotation['color'][0]*255)}, "
            f"{int(annotation['color'][1]*255)}, "
            f"{int(annotation['color'][2]*255)}, 0.3)")
        display_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        display_widget.customContextMenuRequested.connect(
            lambda pos, aid=annotation_id: self.show_annotation_menu(pos, aid))
        display_widget.clicked.connect(
            lambda checked, aid=annotation_id: self.edit_annotation(aid))
        
        # Store the annotation_id as a property of the button
        display_widget.setProperty("annotation_id", annotation_id)
        
        return display_widget

    def show_annotation_menu(self, pos, annotation_id):
        """
        Display a context menu with options to edit or delete the annotation.

        Args:
            pos (QPoint): Position of the context menu trigger.
            annotation_id (int): ID of the annotation for which the menu is shown.
        """
        menu = QMenu(self)
        edit_action = menu.addAction("Edit")
        delete_action = menu.addAction("Delete")
        
        action = menu.exec_(self.sender().mapToGlobal(pos))
        
        if action == edit_action:
            self.edit_annotation(annotation_id)
        elif action == delete_action:
            self.delete_annotation(annotation_id)


    def edit_annotation(self, annotation_id):
        """
        Replace the annotation widget in the timeline with an editable form.

        Maintains the visual order of annotations and ensures only one is editable at a time.

        Args:
            annotation_id (int): ID of the annotation to edit.
        """
        # Store the current order of annotations
        current_order = []
        
        # Collect widgets and their annotation IDs
        for i in range(self.timeline_layout.count()):
            item = self.timeline_layout.itemAt(i)
            widget = item.widget()
            if widget and not isinstance(widget, QSpacerItem):
                # Get the annotation ID from the widget property
                aid = widget.property("annotation_id")
                if aid is not None:
                    current_order.append(aid)
        
        # Clear the timeline layout
        while self.timeline_layout.count():
            item = self.timeline_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # If we couldn't get the order, fall back to sorting by start time
        if not current_order:
            current_order = [aid for aid, _ in sorted(self.annotations.items(), 
                                                    key=lambda x: x[1]['start'])]
        
        # Rebuild the timeline with widgets in the same order
        for aid in current_order:
            if aid == annotation_id:
                editable_widget = EditableAnnotationWidget(annotation_id, self.annotations[annotation_id])
                self.timeline_layout.addWidget(editable_widget)
            else:
                widget = self.create_annotation_widget(aid, self.annotations[aid])
                self.timeline_layout.addWidget(widget)
        
        # Add stretch at the end
        self.timeline_layout.addStretch()

    def update_annotation(self, annotation_id, new_annotation):
        """
        Update the internal annotation dictionary and refresh the UI.

        Args:
            annotation_id (int): ID of the annotation to update.
            new_annotation (dict): The updated annotation values.
        """
        self.annotations[annotation_id] = new_annotation
        self.update_timeline()
        self.update_plots()
        self.update_annotation_bar()

    def delete_annotation(self, annotation_id):
        """
        Delete an annotation from the internal storage and update the timeline and plots.

        Args:
            annotation_id (int): ID of the annotation to delete.
        """
        annotation = self.annotations[annotation_id]
        color_hex = '#{:02x}{:02x}{:02x}'.format(
            int(annotation['color'][0] * 255),
            int(annotation['color'][1] * 255),
            int(annotation['color'][2] * 255)
        )
        self.used_colors.discard(color_hex)
        
        del self.annotations[annotation_id]
        self.update_timeline()
        self.update_plots()
        self.update_annotation_bar()

    def show_error_message(self, message):
        """
        Display an error message in the status bar with red-colored text.

        Args:
            message (str): The message to display.
        """
        # Show an error message in the status bar with red-colored text
        self.status_bar.setStyleSheet("QStatusBar { color: red; }")
        self.status_bar.showMessage(message, 5000)  # Message stays for 5 seconds

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.data is not None:
            self.update_frame()

    def _on_splitter_moved(self):
        if self.data is not None:
            self.update_frame()

    def keyPressEvent(self, event):
        """
        Handle key press events for playback, navigation, annotation, and scrubbing.

        Args:
            event (QKeyEvent): The key event triggered by the user.
        """
        key = event.key()
        
        if key == self.key_bindings['play']:  # Toggle play/pause with Spacebar
            self.toggle_playback()

        elif key == self.key_bindings['previous_segment']:  # Move to previous segment
            self.load_previous_segment()

        elif key == self.key_bindings['next_segment']:  # Move to next segment
            self.load_next_segment()

        elif key == self.key_bindings['toggle_annotation']:  # Start/Stop annotation with Enter
            self.toggle_annotation()

        elif key == self.key_bindings['delete_last_annotation']:  # Delete last annotation with Backspace
            self.delete_last_annotation()

        elif key == self.key_bindings['save_segment_annotation']:  # Save segment annotations with 'S'
            self.save_segment_annotations()

        elif key == self.key_bindings['fast_forward']:    # Fast-forward video
            self.scrub_video(self.config['ff_value_big'])

        elif key == self.key_bindings['rewind']: # Rewind video
            self.scrub_video(-self.config['ff_value_big'])

        elif key == self.key_bindings['fast_forward_small']:
            self.scrub_video(self.config['ff_value_small'])

        elif key == self.key_bindings['rewind_small']:
            self.scrub_video(-self.config['ff_value_small'])

        elif key == self.key_bindings['jump_to_end']:  # Jump to end
            self.scrub_video(self.max_timestamp)

        else:
            super().keyPressEvent(event)  # Ensure default behavior for unhandled keys

    def _rebuild_video_labels(self):
        """Rebuild video label widgets to match current camera_keys."""
        for label in self.video_labels:
            self.video_layout.removeWidget(label)
            label.deleteLater()
        self.video_labels.clear()

        for _ in range(len(self.config['camera_keys'])):
            label = QLabel()
            label.setMinimumSize(200, 150)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid black")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.video_labels.append(label)
            self.video_layout.addWidget(label)

    def load_data(self):
        """
        Load the dataset. If download mode is enabled (RLDS), auto-downloads.
        Otherwise prompts user to select a dataset path.
        Initializes segment index, updates segment info, and prepares UI.
        """
        download_mode = self.config.get('download', False)

        if download_mode:
            # Auto-download mode: no file dialog needed
            self.file_path = self.config.get('data_dir', '~/tensorflow_datasets')
            self.status_bar.showMessage(f"Downloading/loading dataset '{self.config['dataset_name']}'...")
            QApplication.processEvents()
            self.dataset.load_data()
        else:
            file_path = self.select_dataset_path()
            if not file_path:
                return
            self.file_path = file_path
            self.dataset.load_data(file_path=file_path)

        self.setFocusPolicy(Qt.StrongFocus)

        # Rebuild video labels if camera_keys were auto-detected
        if len(self.config['camera_keys']) != len(self.video_labels):
            self._rebuild_video_labels()

        # Initialize with first segment
        self.current_segment_index = self.dataset.current_segment_idx
        self.load_current_segment()

        # Update segment info display
        self.update_segment_info()
        self.status_bar.showMessage("Dataset loaded successfully.", 5000)


    def load_current_segment(self):
        """
        Load the current segment's data and annotations. Updates visual elements,
        plots, timeline, and internal state for playback and annotation.
        """
        if not self.dataset.segments_info or not self.file_path:
            return
        
        # Reassemble stores annotations in-place in the H5 file, so segments_info
        # must be reloaded to pick up saved annotation data
        if isinstance(self.dataset, Reassemble):
            self.dataset.load_segments_info(self.file_path)
        segment = self.dataset.segments_info[str(self.current_segment_index)]

        # Update action label
        text = segment['text']
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        self.action_label.setText(f"Current Action: {segment['text']}")  # Update the action label with the action text
        
        # load data with dataloader
        self.data = self.dataset.get_segment(segment_idx=self.current_segment_index)
        self.max_timestamp = self.dataset.get_max_timestamp()
        
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(int(self.max_timestamp*1000))

        # Reset current time and annotations
        self.current_time = 0
        self.annotations = {}
        self.current_annotation_id = 0
        
        # Load existing annotations for this segment
        self.load_segment_annotations()
        
        # Update numerical data selectors
        self.update_data_selectors()
        
        # Update UI
        self.update_segment_info()
        self.update_frame()
        self.update_timeline()
        self.update_plots()



    def update_data_selectors(self):
        """
        Update the scrollable list of checkboxes for selecting which numerical
        data fields to visualize. This is based on the config-defined `low_level_keys`.
        """
        # Clear existing checkboxes
        for selector in self.data_selectors:
            selector.deleteLater()
        self.data_selectors.clear()
        
        if not self.data: # or 'robot_state' not in self.data:
            return
            
        # Create new checkboxes for each data field
        scroll_widget = self.data_selector_scroll.widget()
        scroll_layout = scroll_widget.layout()
        
        # for key in self.data['robot_state'].keys():
        if self.config['low_level_keys']:
            # Use current selection if set (even if empty), otherwise fall back to defaults
            selected = self.selected_numerical_data if self.selected_numerical_data is not None else self.config.get('default_graphs', [])
            for key in self.config['low_level_keys']:
                checkbox = QCheckBox(key)
                checkbox.stateChanged.connect(self.on_data_selection_changed)
                self.data_selectors.append(checkbox)
                if key in selected:
                    checkbox.setChecked(True)
                scroll_layout.addWidget(checkbox)

    def load_previous_segment(self):
        """
        Save annotations and load the previous data segment, if available.
        """
        if self.current_segment_index > 0:
            # Save current segment annotations
            self.save_segment_annotations()
            
            # Load previous segment
            self.current_segment_index -= 1
            self.load_current_segment()
            self.update_segment_info()
    
    def load_next_segment(self):
        """
        Save annotations and load the next data segment, if available.
        """
        if self.current_segment_index < len(self.dataset.segments_info) - 1:
            # Save current segment annotations
            self.save_segment_annotations()
            
            # Load next segment
            self.current_segment_index += 1
            self.load_current_segment()
            self.update_segment_info()
    
    def update_segment_info(self):
        """
        Update the segment info label to reflect the current segment index and total count.
        """
        self.segment_info_label.setText(
            f"Segment: {self.current_segment_index + 1}/{len(self.dataset.segments_info)}")
    
    def save_segment_annotations(self):
        """
        Save current segment's annotations to the dataset using absolute timestamps.
        Sorts annotations by start time before saving.
        """
        if not self.file_path or not self.annotations:
            return
        
        # Prepare annotations with absolute timestamps
        segment = self.dataset.segments_info[str(self.current_segment_index)]
        segment_start = segment['start']

        annots = deepcopy(self.annotations)
        for ann_id, ann in annots.items():
            ann['start'] += segment_start
            ann['end'] += segment_start

        # Sort annotations by start time
        # Convert dict to list of (id, annotation) tuples, sort, and convert back to dict
        sorted_annots = dict(sorted(annots.items(), key=lambda item: item[1]['start']))
        self.dataset.write_annot_data(segment_idx=self.current_segment_index,
                                            annots=sorted_annots)
    
    def load_segment_annotations(self):
        """
        Load annotations for the current segment and prepare internal storage.
        Assign colors and update annotation IDs accordingly.
        """
        if not self.file_path:
            return
        self.annotations = {}
        annots = self.dataset.load_annot_data(self.current_segment_index)
        for ann_id, ann in annots.items():
            self.annotations[int(ann_id)] = {
                'start': ann['start'],
                'end': ann['end'],
                'label': next(
                (v.decode('utf-8') if isinstance(v, bytes) else v
                for k in ['label', 'text'] if (v := ann.get(k)) is not None),
                ''
                ),
                'success': ann['success'] if 'success' in ann.keys() else True,
                'color': self.generate_random_color(idx=int(ann_id))
            }

            self.current_annotation_id = max(
                    self.current_annotation_id, 
                    int(ann_id) + 1
                ) 
    
    def jump_to_segment(self):
        """
        Jump to a user-specified segment index entered in the text box.
        Validates the index and loads the corresponding segment if valid.
        """
        try:
            index = int(self.jump_input.text())
            max_index = len(self.dataset.segments_info)
            if index < 0 or index >= max_index:
                QMessageBox.warning(self, "Invalid Index", f"Please enter a value between 0 and {max_index - 1}")
                return
            self.current_segment_index = index
            self.load_current_segment()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer.")
    
    def on_data_selection_changed(self):
        """
        Update the list of selected numerical data fields and refresh plots
        when checkboxes are toggled.
        """
        self.selected_numerical_data = [
            selector.text() for selector in self.data_selectors 
            if selector.isChecked()
        ]
        self.update_plots()

    def slider_changed(self):
        """
        Update the current time in seconds based on the slider position.
        Refresh the video frame accordingly.
        """
        self.current_time = self.timeline_slider.value() / 1000.0
        self.update_frame()

    def toggle_playback(self):
        """
        Start or stop video playback and update the play button text and timer.
        """
        self.playing = not self.playing
        if self.playing:
            self.play_button.setText(f"Pause ({key_to_string(self.key_bindings['play'])})")
            self.timer.start(round(1000/self.config['fps']))  # was 33, ~30 fps
        else:
            self.play_button.setText(f"Play ({key_to_string(self.key_bindings['play'])})")
            self.timer.stop()

    def toggle_annotation(self):
        """
        Start or end an annotation interval.
        On ending, shows a dialog to input action label and success flag,
        then stores the annotation and updates the UI.
        """
        if not self.recording_annotation:
            # if self.current_annotation_id >= 1:
            #     # Show an error message in the message bar if there's already an active annotation
            #     self.show_error_message("You already have an active annotation. Please end the current annotation before starting a new one.")
            #     return  # Exit the function if there's an error

            # We'll get the label when the annotation ends, so use a placeholder for now     
            # Start new annotation
            # label = self.annotation_input.text()
            label = 'Transition'
            
            if not label or not self.file_path:
                return
                
            self.recording_annotation = True
            self.current_annotation_start = self.current_time
            self.annotation_button.setText(f"End Action ({key_to_string(self.key_bindings['toggle_annotation'])})")
            self.annotation_button.setStyleSheet("background-color: #ff9999")
        else:
            # End current annotation
            if self.current_annotation_start is not None:
                # Show custom dialog
                dialog = ActionDialog(parent=self, action_map=self.action_map)
                if dialog.exec_() == QDialog.Accepted:
                    # Get input values
                    annotation_name, success = dialog.get_input()

                    # Generate a random color that hasn't been used yet
                    color = self.generate_random_color(self.current_annotation_id)

                    # Store the annotation with the success flag
                    self.annotations[self.current_annotation_id] = {
                        'start': self.current_annotation_start,
                        'end': self.current_time,
                        'success': success,
                        'label': annotation_name, 
                        'color': color
                    }

                    self.current_annotation_id += 1
                    self.recording_annotation = False
                    self.current_annotation_start = None
                    self.annotation_button.setText(f"Start Action ({key_to_string(self.key_bindings['toggle_annotation'])})")
                    self.annotation_button.setStyleSheet("")

                    # Update the UI elements
                    self.update_timeline()
                    self.update_plots()
                    self.update_annotation_bar()


    def delete_last_annotation(self):
        """
        Delete the most recent annotation and free up its color.
        Updates the timeline and plots accordingly.
        """
        if self.current_annotation_id > 0:
            self.current_annotation_id -= 1
            annotation = self.annotations.pop(self.current_annotation_id)
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(annotation['color'][0] * 255),
                int(annotation['color'][1] * 255),
                int(annotation['color'][2] * 255)
            )
            self.used_colors.discard(color_hex)
            self.update_timeline()
            self.update_plots()
            self.update_annotation_bar()

    def find_nearest_frame_index(self, timestamps, target_time):
        """
        Find the index of the timestamp closest to the target time.
        
        Args:
            timestamps (list or np.ndarray): Sorted list of timestamps.
            target_time (float): Target time in seconds.

        Returns:
            int or None: Index of the nearest timestamp, or None if unavailable.
        """
        if timestamps is None or len(timestamps) == 0:
            return None
        idx = np.searchsorted(timestamps, target_time)
        if idx >= len(timestamps):
            idx = len(timestamps) - 1
        return idx

    def update_frame(self):
        """
        Update the video display and timeline slider based on the current time.
        Refreshes frames for all camera views and updates plots.
        """
        if self.data is None:
            return
            
        # Update time display
        self.time_label.setText(f"Time: {self.current_time:.3f} s")

        # Update timeline slider WITHOUT triggering valueChanged signal
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(int(self.current_time * 1000))
        self.timeline_slider.blockSignals(False)
        
        # Update video frames
        # cameras = ['hama1', 'hama2', 'hand']
        # for i, cam in enumerate(cameras):
        #     if cam in self.data and cam in self.data['timestamps']:
        #         idx = self.find_nearest_frame_index(self.data['timestamps'][cam], 
        #                                           self.current_time)
        #         if idx is not None:
        #             self.display_frame(i, self.data[cam][idx])

        for i, cam_key in enumerate(self.config["camera_keys"]):
            frame_data = get_nested(self.data, cam_key)

            if 'timestamps' in self.data and cam_key in self.data['timestamps']:
                idx = self.find_nearest_frame_index(self.data['timestamps'][cam_key], self.current_time)
                if idx is not None:
                    self.display_frame(i, frame_data[idx])
            else:
                self.display_frame(i, frame_data[round(self.current_time * self.config['fps'])])

            # except (KeyError, TypeError):
            #     print(f"Warning: Could not resolve camera key: {cam_key}")
            #     continue
        
        # Update all plots and annotation overview bar
        self.update_plots()
        self.update_annotation_bar()

        # Increment time if playing
        if self.playing:
            self.current_time += 1 / self.config['fps'] 
            if self.current_time > self.max_timestamp:
                self.current_time = 0
        

    def display_frame(self, index, frame):
        """
        Display a given frame in the specified video label.
        
        Args:
            index (int): Index of the video label.
            frame (np.ndarray): Frame image as a numpy array (grayscale or color).
        """
        # Grayscale
        if frame.ndim == 2: 
            h, w = frame.shape
            bytes_per_line = w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        
        # Color image
        elif frame.ndim == 3:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            if ch == 3:
                # Use config to decide whether conversion is needed
                color_format = self.config.get('color_format', 'BGR')  # default to BGR

                if color_format.upper() == 'BGR':
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:  # Assume it's already RGB
                    frame_rgb = frame

                qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                return  # Unsupported format
    
        else:
            return  # Invalid frame shape
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_labels[index].size(), 
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_labels[index].setPixmap(scaled_pixmap)

    def scrub_video(self, delta):
        """
        Move the video playback time forward or backward by a given delta.

        Args:
            delta (float): Amount in seconds to move the playback position.
        """
        if self.data is None:
            return
        
        self.current_time = max(0, min(self.current_time + delta, self.max_timestamp))
        self.timeline_slider.setValue(int(self.current_time * 1000))   # * 1000
        # self.update_frame()  # Refresh the display with the new timestamp


    def update_plots(self):
        """
        Update the numerical data plots based on selected data fields.
        
        Includes:
        - Plotting of new data
        - Drawing vertical playback line
        - Displaying annotation regions
        - Handling single/multi-dimensional data
        - Adjusting to dynamic plot count and layout
        """
        if not self.data:
            return

        if not self.selected_numerical_data:
            # Nothing selected or no numerical data configured — remove all plots
            while self.plot_canvases:
                plot_widget = self.plot_canvases.pop()
                self.plot_figures.pop()
                self.plots_grid.removeWidget(plot_widget)
                plot_widget.deleteLater()
            return

        n_plots = len(self.selected_numerical_data)
        if n_plots == 0:
            # Remove all plots when nothing is selected
            while self.plot_canvases:
                plot_widget = self.plot_canvases.pop()
                self.plot_figures.pop()
                self.plots_grid.removeWidget(plot_widget)
                plot_widget.deleteLater()
            return
        
        n_rows = min(2, (n_plots + 1) // 2)
        n_cols = (n_plots + n_rows - 1) // n_rows
        
        # Clear existing plots if plot count changes
        while len(self.plot_canvases) < n_plots:
            # Create new plot widget
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('white')
            plot_widget.showGrid(x=True, y=True)
            
            # Create curves for each dimension
            curves = [
                plot_widget.plot(pen=pg.mkPen('b', width=2)),
                plot_widget.plot(pen=pg.mkPen('g', width=2)),
                plot_widget.plot(pen=pg.mkPen('r', width=2))
            ]
            
            # Create vertical line
            vertical_line = pg.InfiniteLine(angle=90, movable=False, pen='k')
            plot_widget.addItem(vertical_line)
            
            # Store plot components
            self.plot_canvases.append(plot_widget)
            self.plot_figures.append({
                'widget': plot_widget,
                'curves': curves,
                'vertical_line': vertical_line,
                'regions': [],
            })
            
            # Add to grid
            row = (len(self.plot_canvases)-1) // 2
            col = (len(self.plot_canvases)+1) % 2
            # print(row,col,len(self.plot_canvases))
            self.plots_grid.addWidget(self.plot_canvases[-1], row, col)
        
        # Remove extra plots if needed
        while len(self.plot_canvases) > n_plots:
            plot_widget = self.plot_canvases.pop()
            plot_figure = self.plot_figures.pop()
            self.plots_grid.removeWidget(plot_widget)
            plot_widget.deleteLater()
        
        # Update each plot
        for idx, (plot_widget, plot_figure) in enumerate(zip(self.plot_canvases, self.plot_figures)):
            data_key = self.selected_numerical_data[idx]
            
            # Solves the data selector bug, no more loose axis lines in wrong plots
            for curve in plot_figure['curves']:
                curve.clear()
            
            # if data_key in self.data['robot_state'] and data_key in self.data['timestamps']:
            #     numerical_data = self.data['robot_state'][data_key]
            #     timestamps = self.data['timestamps'][data_key]

            if has_nested_key(self.data, data_key): # and data_key.split('/')[-1] in self.data['timestamps']:
                numerical_data = get_nested(self.data, data_key)
                # timestamps = self.data['timestamps'][data_key.split('/')[-1]]
                timestamp_key = data_key.split('/')[-1]
                timestamps = self.data.get('timestamps', {}).get(timestamp_key, None)
                if timestamps is None:
                    n_points = len(numerical_data)
                    numerical_data = get_nested(self.data, data_key)
                    timestamps = np.arange(n_points) / self.config['fps']
                
                # Handle multi-dimensional data
                if numerical_data.ndim > 1:
                    # Plot each column
                    for col in range(min(numerical_data.shape[1], 3)):
                        plot_figure['curves'][col].setData(timestamps, numerical_data[:, col])
                    
                    # Set labels for dimensions
                    dimension_labels = ['X', 'Y', 'Z'][:numerical_data.shape[1]]
                    plot_widget.setTitle(f"{data_key} - {', '.join(dimension_labels)}")
                else:
                    # Single column data
                    plot_figure['curves'][0].setData(timestamps, numerical_data)
                    plot_widget.setTitle(data_key)
                
                # Update x-axis limits
                plot_widget.setXRange(0, self.max_timestamp)
                
                # Update vertical line position
                plot_figure['vertical_line'].setPos(self.current_time)

                # Remove old regions
                for region in plot_figure['regions']:
                    plot_widget.removeItem(region)
                plot_figure['regions'].clear()

                # Add annotations as shaded regions
                for annotation in self.annotations.values():
                    rgb = tuple(int(round(c*255)) for c in annotation['color'])
                    # color = QtGui.QColor('blue') 
                    color = QtGui.QColor(*rgb)  # Convert to QColor
                    color.setAlpha(60)  # Set transparency (0 = fully transparent, 255 = fully opaque)
                    region = pg.LinearRegionItem(values=[annotation['start'],
                                                        annotation['end']], 
                                                        brush=pg.mkBrush(color),
                                                movable=False)
                    
                    plot_widget.addItem(region)
                    plot_figure['regions'].append(region)

                # If recording annotation is active, add a red region
                if self.recording_annotation and self.current_annotation_start is not None:
                    record_color = QtGui.QColor('red')
                    record_color.setAlpha(50)  # Set transparency
                    record_region = pg.LinearRegionItem(values=[self.current_annotation_start,
                                                                self.current_time],
                                                                brush=pg.mkBrush(record_color),
                                                        movable=False)
                    plot_widget.addItem(record_region)
                    plot_figure['regions'].append(record_region)

    def update_annotation_bar(self):
        """
        Update the annotation overview bar with colored regions for each annotation,
        a playback position line, and the active recording region.
        """
        if self.max_timestamp <= 0:
            return

        # Remove old regions
        for region in self.annotation_bar_regions:
            self.annotation_bar.removeItem(region)
        self.annotation_bar_regions.clear()

        # Set ranges
        self.annotation_bar.setXRange(0, self.max_timestamp)
        self.annotation_bar.setYRange(0, 1)

        # Add annotation regions
        for annotation in self.annotations.values():
            rgb = tuple(int(round(c * 255)) for c in annotation['color'])
            color = QtGui.QColor(*rgb)
            color.setAlpha(120)
            region = pg.LinearRegionItem(
                values=[annotation['start'], annotation['end']],
                brush=pg.mkBrush(color),
                movable=False
            )
            self.annotation_bar.addItem(region)
            self.annotation_bar_regions.append(region)

        # Show recording region
        if self.recording_annotation and self.current_annotation_start is not None:
            record_color = QtGui.QColor('red')
            record_color.setAlpha(80)
            record_region = pg.LinearRegionItem(
                values=[self.current_annotation_start, self.current_time],
                brush=pg.mkBrush(record_color),
                movable=False
            )
            self.annotation_bar.addItem(record_region)
            self.annotation_bar_regions.append(record_region)

        # Update playback position line
        self.annotation_bar_vline.setPos(self.current_time)

    def update_timeline(self):
        """
        Rebuild the timeline widget with current annotations.

        Removes old widgets and creates new annotation widgets in sorted order
        based on annotation start times.
        """

        while self.timeline_layout.count():
            item = self.timeline_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for annotation_id, annotation in sorted(self.annotations.items(), 
                                             key=lambda x: x[1]['start']):
            annotation_widget = self.create_annotation_widget(annotation_id, annotation)
            self.timeline_layout.addWidget(annotation_widget)

        self.timeline_layout.addStretch()

    def generate_random_color(self, idx):
        """
        Generate a distinct color from the matplotlib 'tab10' colormap.

        Args:
            idx (int): Index used to pick a color.

        Returns:
            tuple: RGB color as a tuple of floats (0 to 1).
        """
        return self.tab10_colors[idx%10]

    def select_dataset_path(self):
        """
        Open a file or folder dialog to select the dataset path, depending on dataset type.

        Returns:
            str or None: Selected file or folder path, or None if canceled.
        """
        dataset_type = type(self.dataset).__name__.lower()

        if dataset_type == "reassemble":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select H5 Data File", "", "HDF5 Files (*.h5 *.hdf5)"
            )
        elif dataset_type == "rlds":
            file_path = QFileDialog.getExistingDirectory(
                self, "Select RLDS Dataset Folder", ""
            )
        elif dataset_type == "frames":
            file_path = QFileDialog.getExistingDirectory(
                self, "Select Frames Folder", ""
            )
        elif dataset_type == "video":
            file_path = QFileDialog.getExistingDirectory(
                self, "Select Video Folder", ""
            )
        elif dataset_type == "rosbag":
            file_path = QFileDialog.getExistingDirectory(
                self, "Select folder with rosbags", ""
            )
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Data File", ""
            )

        return file_path or None  # Returns None if user cancels


def key_to_string(key):
    """
    Convert a Qt key enum to a readable string (e.g., Qt.Key_Space → "Space").

    Args:
        key (int): Qt key enum.

    Returns:
        str: Human-readable key name.
    """
    return QKeySequence(key).toString()

if __name__ == '__main__':
    # Entry point for running the GUI as a standalone application
    parser = argparse.ArgumentParser(description="Launch ATLAS")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    app = QApplication(sys.argv)
    window = SegmentedVideoAnnotator(config=config)
    window.show()
    sys.exit(app.exec_())