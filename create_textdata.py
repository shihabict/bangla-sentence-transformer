import random
from tqdm import tqdm


def get_test_data(train_dat_path, test_dat_path):
    with open(train_dat_path) as f:
        train_data = f.readlines()

    test_data = random.sample(train_data, 2000)

    #remove test data from train data
    for line in tqdm(test_data):
        train_data.remove(line)
    with open(test_dat_path, 'w') as f:
        f.writelines(test_data)

    with open(train_data_path, 'w') as f:
        f.writelines(train_data)

    return train_data, test_data


def reduce_training_data(train_data_path, number_of_sentences):
    with open(train_data_path) as f:
        train_data = f.readlines()
    train_data = random.sample(train_data,number_of_sentences)
    with open(train_data_path.split('.txt')[0] + f'_{number_of_sentences}.txt', 'w') as f:
        f.writelines(train_data)


if __name__ == '__main__':
    train_data_path = 'DATA/combined_data.txt'
    # train_data_path = 'DATA/reformated_combine_dataset.txt'
    test_data_path = 'DATA/test_data.txt'
    get_test_data(train_data_path, test_data_path)
    # reduce_training_data(train_data_path, 1000000)
