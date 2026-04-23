from abc import ABC, abstractmethod

class DatasetBase(ABC):
    """
    Abstract base class for dataset handlers used in the segmentation GUI.

    This interface enforces a standard structure for different dataset types
    (e.g., REASSEMBLE, RLDS), including methods for loading data, accessing segments,
    and reading/writing annotations.
    """

    def __init__(self):
        """
        Initialize the dataset base class.

        Attributes:
            file_path (str or None): Path to the dataset source.
            data (Any): Loaded dataset object (to be set in subclass).
            current_segment_idx (int or None): Index of the currently loaded segment.
            segments_info (dict or None): Metadata about available segments.
        """
        self.file_path = None
        self.data = None
        self.current_segment_idx = None
        self.segments_info = None

    def __len__(self):
        """
        Return the number of data entries or segments in the dataset.

        Returns:
            int: Number of data elements.
        """
        return len(self.data)

    @abstractmethod
    def get_segment(self, segment_idx):
        """
        Retrieve a specific segment from the dataset.

        Args:
            segment_idx (int): Index of the segment to retrieve.

        Returns:
            Any: A dataset-specific segment. 
                 In REASSEMBLE, this is a high-level action within one H5 file.
                 In RLDS, this is an episode.
        """
        pass

    @abstractmethod
    def load_data(self, file_path):
        """
        Load dataset from the given file path.

        Args:
            file_path (str): Path to the dataset.

        This method should also prepare the dataset for iteration and load the
        first sample, if applicable.
        """
        pass

    @abstractmethod
    def write_annot_data(self, segment_idx, annots):
        """
        Write annotation data back to the dataset or external storage.

        Args:
            segment_idx (int): Index of the segment to annotate.
            annots (dict): Dictionary containing annotation data.

        Note:
            - In REASSEMBLE, this modifies the dataset file directly.
            - In RLDS, annotations are stored in external JSON files.
        """
        pass

    @abstractmethod
    def load_annot_data(self, segment_idx):
        """
        Load annotation data for a given segment.

        Args:
            segment_idx (int): Index of the segment.

        Returns:
            dict: Loaded annotation data for the segment.
        """
        pass

    @abstractmethod
    def load_segments_info(self, file_path):
        """
        Load segment metadata for the given file.

        Args:
            file_path (str): Path to the dataset file.

        This populates self.segments_info with segment metadata
        (e.g., start/end times, text descriptions).
        """
        pass

    @abstractmethod
    def get_max_timestamp(self):
        """
        Return the maximum timestamp for the current segment.

        Returns:
            float: Maximum timestamp value in seconds.
        """
        pass
