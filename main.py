# This is a sample Python script.
from training.tarin import BNSentenceTransformer

if __name__ == '__main__':
    transformer = BNSentenceTransformer()

    # path = 'DATA/dataset.txt'
    # path = 'DATA/hf_aibarat_dataset.txt'
    path = 'DATA/combined_data.txt'
    number_of_sentences = 'all_data'
    # number_of_sentences = 10000
    # save_model_name = f'bangla_snt_indic'
    save_model_name = f'bengal_transformer_distilroberta'
    transformer.train_st(path, number_of_sentences, save_model_name)
