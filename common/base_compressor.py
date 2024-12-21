class BaseCompressor:
    """
    Abstract base class for compression methods.
    All specific compression methods should inherit from this class.
    """

    def __init__(self, model):
        self.model = model

    def compress(self):
        """
        Apply compression to the model.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
