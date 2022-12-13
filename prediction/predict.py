# from Bangla-sentence-embedding-transformer.Bangla_transformer import Bangla_sentence_transformer_small
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim


transformer = SentenceTransformer(
    'bangla_transformer_100/')

sentences = ['আপনার বয়স কত', 'আমি তোমার বয়স জানতে চাই', 'আমার ফোন ভাল আছে', 'আপনার সেলফোনটি দুর্দান্ত দেখাচ্ছে']

sentences_embeddings = transformer.encode(sentences)

for i in range(len(sentences)):
    j = 0
    while j < len(sentences):
        s1 = sentences[i]
        s2 = sentences[j]
        print(s1, ' --- ', s2, pytorch_cos_sim(sentences_embeddings[i], sentences_embeddings[j]))

        j += 1
