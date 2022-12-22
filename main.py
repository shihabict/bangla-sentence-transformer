# This is a sample Python script.
from trianing.tarin import BNSentenceTransformer

if __name__ == '__main__':
    transformer = BNSentenceTransformer()

    # path = 'DATA/dataset.txt'
    #path = 'DATA/hf_aibarat_dataset.txt'
    path = 'DATA/combined_data.txt'
    number_of_sentences = 'Full_data'
    #number_of_sentences = 100
    save_model_name = f'bangla_transformer_{number_of_sentences}'
    transformer.train_new(path,number_of_sentences,save_model_name)

