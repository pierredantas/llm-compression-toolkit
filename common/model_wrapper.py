class ModelWrapper:
    """
    Wrapper class for models to provide additional functionality
    and abstraction for compression tasks.
    """

    def __init__(self, model):
        self.model = model

    def save(self, path):
        """
        Save the model to the specified path.

        Parameters:
            path (str): Path to save the model.
        """
        self.model.save_pretrained(path)

    def load(self, path):
        """
        Load the model from the specified path.

        Parameters:
            path (str): Path to load the model.
        """
        self.model.from_pretrained(path)
