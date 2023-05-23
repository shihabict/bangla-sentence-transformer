# Bangla Sentence Transformer

 Sentence Transformer is a cutting-edge natural language processing (NLP) model that is capable of encoding and transforming sentences into high-dimensional embeddings. With this technology, we can unlock powerful insights and applications in various fields like text classification, information retrieval, semantic search, and more.

This model is finetune from ```stsb-xlm-r-multilingual``` 
 it's now available on Hugging Face! 🎉🎉

## Install

```
pip install -U sentence-transformers
```
## How to get sentence similarity

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim


transformer = SentenceTransformer('shihab17/bangla-sentence-transformer')

sentences = ['আমি আপেল খেতে পছন্দ করি। ', 'আমার একটি আপেল মোবাইল আছে।','এইবার কমলার ফলনা ভাল হয়নি', 'বাচ্চাটি দেখতে আপেলের মত সুন্দর','আপেলের জুস আমার অনেক প্রিয়']

sentences_embeddings = transformer.encode(sentences)

for i in range(len(sentences)):
    for j in range(i, len(sentences)):
        sen_1 = sentences[i]
        sen_2 = sentences[j]
        sim_score = float(pytorch_cos_sim(sentences_embeddings[i], sentences_embeddings[j]))
        print(sen_1, '----->', sen_2, sim_score)
```


```python
from sentence_transformers import SentenceTransformer
sentences = ['আমি আপেল খেতে পছন্দ করি। ', 'আমার একটি আপেল মোবাইল আছে।','এইবার কমলার ফলনা ভাল হয়নি', 'বাচ্চাটি দেখতে আপেলের মত সুন্দর','আপেলের জুস আমার অনেক প্রিয়']

model = SentenceTransformer('shihab17/bangla-sentence-transformer ')
embeddings = model.encode(sentences)
print(embeddings)
```

```python
from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['আমি আপেল খেতে পছন্দ করি। ', 'আমার একটি আপেল মোবাইল আছে।','এইবার কমলার ফলনা ভাল হয়নি', 'বাচ্চাটি দেখতে আপেলের মত সুন্দর','আপেলের জুস আমার অনেক প্রিয়']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('shihab17/bangla-sentence-transformer')
model = AutoModel.from_pretrained('shihab17/bangla-sentence-transformer')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
```

## Best MSE: 7.57528096437454
