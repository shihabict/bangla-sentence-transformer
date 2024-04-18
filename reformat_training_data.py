from tqdm import tqdm
from p_tqdm import p_map


class TrainingDataFormatter:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def save_date(self, data):
        data = data.replace(' ### ', '\t')
        with open(self.output_path, 'a') as file:
            file.writelines(data)

    def load_data(self):
        with open(self.input_path, 'r') as file:
            lines = file.readlines()

        p_map(self.save_date, lines, **{"num_cpus": 8})
        # for line in tqdm(lines):
        #     self.save_date(line)


if __name__ == '__main__':
    input_path = 'DATA/dataset.txt'
    output_path = 'DATA/reformated_dataset.txt'
    formatter = TrainingDataFormatter(input_path, output_path)
    formatter.load_data()
