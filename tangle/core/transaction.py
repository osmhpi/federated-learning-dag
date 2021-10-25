class Transaction:
    def __init__(self, parents, id=None, metadata=None):
        self.parents = parents
        self.id = id
        self.metadata = metadata if metadata is not None else {}

    def add_metadata(self, key, value):
        self.metadata[key] = value
