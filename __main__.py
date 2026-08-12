from models.download_model import get_or_download_model
from library.tower_text import TextModel

def main():
    # picture tower part
    pass

    # text tower part
    model, processor = get_or_download_model()
    text_models = (TextModel(0, model, processor), TextModel(1, model, processor))

    # join part
    pass

if __name__ == '__main__':
    main()
