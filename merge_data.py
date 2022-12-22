import os
import string
from multiprocessing import Pool

from tqdm import tqdm

from settings import BASE_DIR


class MergeData:

    def __init__(self):
        self.existing_data_path = BASE_DIR + '/DATA/dataset.txt'
        self.new_data_path = BASE_DIR + '/DATA/hf_aibarat_dataset.txt'
        self.output_file = BASE_DIR + '/DATA/combined_data.txt'

    def write_into_txt(self, line):
        line = line.strip()
        with open(self.output_file, 'a') as file:
            file.write(line)
            file.write('\n')

    def merge_all_data(self):
        with open(self.existing_data_path) as existing_file:
            existing_data = existing_file.read()

        with open(self.new_data_path) as new_file:
            new_data = new_file.read()


        combined_data = existing_data+ '\n' + new_data
        # combined_data = new_data

        splited_combined_data = combined_data.split('\n')

        # splited_combined_data = splited_combined_data[:1000]

        pool = Pool(processes=18)
        for _ in tqdm(pool.imap_unordered(self.write_into_txt, splited_combined_data),
                           total=len(splited_combined_data)):
            pass

if __name__ == '__main__':
    merge_data = MergeData()
    merge_data.merge_all_data()


