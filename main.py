# This is a sample Python script.
from trianing.tarin import BNSentenceTransformer


if __name__ == '__main__':
    transformer = BNSentenceTransformer()

    # path = 'DATA/dataset.txt'
    path = 'DATA/hf_aibarat_dataset.txt'
    transformer.train_new(path)

