import pandas as pd
from tqdm import tqdm
from settings import BASE_DIR


class CreateNewData:
    # def __init__(self, input_path, output_path):
    #     self.input_path = BASE_DIR + '/' + input_path
    #     self.output_path = BASE_DIR + '/' + output_path

    def format_hf_data(self,file_path):
        data = pd.read_csv(file_path)
        count = 0
        for _, row in tqdm(data.iterrows()):
            with open('DATA/hf_aibarat_dataset.txt','a') as file:
                file.write(f"{row[3]} ### {row[2]}")
                file.write('\n')


    def translate_using_helsinki(self,input_sentence):
        input_sentence = input_sentence.strip()
        self.tokenizer.tgt_lang = "en"
        encoded_hi = self.tokenizer(input_sentence, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_hi)
        predicted_sentence = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predicted_sentence = ''.join(predicted_sentence)
        with open(self.output_path, "a") as write_file:
            write_file.write(f"{input_sentence} ### {predicted_sentence}")
            write_file.write('\n')
    def translate_using_alirezamsh(self, input_sentence):

        # hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
        # chinese_text = "生活就像一盒巧克力。"
        input_sentence = input_sentence.strip()
        self.tokenizer.tgt_lang = "en"
        encoded_hi = self.tokenizer(input_sentence, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_hi)
        predicted_sentence = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predicted_sentence = ''.join(predicted_sentence)
        with open(self.output_path, "a") as write_file:
            write_file.write(f"{input_sentence} ### {predicted_sentence}")
            write_file.write('\n')

        # => "Life is like a box of chocolate."

    def translate_using_cse_buet(self, input_sentence):
        input_sentence = input_sentence.strip()
        input_ids = self.tokenizer(normalize(input_sentence), return_tensors="pt").input_ids
        generated_tokens = self.model.generate(input_ids)
        decoded_tokens = self.tokenizer.batch_decode(generated_tokens)[0]
        predicted_sentece = decoded_tokens.lstrip('<pad>').rstrip('</s>')
        # print(f"{input_sentence}----------{decoded_tokens.lstrip('<pad>').rstrip('</s>')}")
        with open(self.output_path, "a") as write_file:
            write_file.write(f"{input_sentence} ###{predicted_sentece}")
            write_file.write('\n')

    def translate_using_google_trans(self, line):
        line = line.strip('\n')

        out = self.translator.translate(line)

        with open(self.output_path, "a") as write_file:
            write_file.write(f"{line} ###{out}")
            write_file.write('\n')

    def translate(self):

        with open(self.input_path) as file:
            lines = file.readlines()
        lines = lines[:100]

        # pool = Pool(processes=3)
        # for _ in tqdm(pool.imap_unordered(self.translate_using_google_trans, lines),
        #                    total=len(lines)):
        #     pass
        # pool = Pool(processes=3)
        # for _ in tqdm(pool.imap_unordered(self.translate_using_alirezamsh, lines),
        #                    total=len(lines)):
        #     pass
        for line in tqdm(lines):
            line = line.strip('\n')
            if len(line) > 0:
                # self.translate_bn_en(line)
                # self.translate_using_alirezamsh(line)
                # self.translate_using_helsinki(line)
                self.translate_using_cse_buet(line)


if __name__ == '__main__':
    input_path = 'DATA/news_articles_cleaned.txt'
    output_path = 'DATA/new_dataset_cse_buet.txt'
    input_dir = 'DATA/aibarat_transcription_data.csv'
    # new_data_creation = CreateNewData(input_path, output_path)
    new_data_creation = CreateNewData()
    # new_data_creation.translate()
    new_data_creation.format_hf_data(input_dir)