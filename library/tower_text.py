class TextModel:
    def __init__(self, category: int, model, processor):
        self.category = category
        self.model = model
        self.processor = processor
