from models.download_model import get_or_download_model
from library.text_tower import TextTower
import pandas as pd
from pathlib import Path

DEFAULT_DATA_PATH = Path('train_dataset/data.csv')

def main(data_path: Path = DEFAULT_DATA_PATH):
    # picture tower part
    pass

    # text tower part
    data = pd.read_csv(DEFAULT_DATA_PATH)
    model = get_or_download_model()
    text_tower = TextTower(model)
    decs = data.loc[0, 'description']
    tensor = text_tower([decs])
    print(tensor.shape, type(tensor), tensor,  sep='\n')

    token_embeddings_output = text_tower([decs], output_value='token_embeddings')[0]
    print(token_embeddings_output.shape, type(token_embeddings_output), token_embeddings_output,  sep='\n')
    # join part
    pass

if __name__ == '__main__':
    main()
