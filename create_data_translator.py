from tqdm import tqdm

input_path = 'DATA/news_articles_cleaned.txt'
with open(input_path) as file:
    lines = file.readlines()
lines = lines[:100]
from translate import Translator
translator= Translator(to_lang="en",from_lang='bn')

for line in tqdm(lines):
    line=line.strip('\n').strip()

    out = translator.translate(line)
    # print(out)
    # output: {'ta': 'வணக்கம் உலகம்'}
    # translatedText = gs.translate(line, 'en')
    # translated_text = predictor.predict(line)
    print("{}-------{}".format(line, out))
