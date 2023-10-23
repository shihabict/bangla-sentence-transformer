import torch
import random
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class MakeTorchData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


class ClassificationAccuracyMeasurement:
    def __init__(self, data_path, n_rows):
        self.data = pd.read_csv(data_path)
        from sklearn.utils import shuffle
        self.data = shuffle(self.data)
        # self.data.sample(frac=1)
        self.data = self.data[:n_rows]
        self.metrics_name = 'f1'

        X = self.data.cleaned_text
        y = self.data.label
        y = pd.factorize(y)[0]
        print(y)

        # Load Metrics
        self.metric = load_metric(self.metrics_name)

        test_size = 0.2

        # Split Data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X.tolist(), y, test_size=test_size)
        self.max_length = 512
        self.num_epochs = 10
        self.num_labels = 2

    # Create Metrics
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # 'micro', 'macro', etc. are for multi-label classification. If you are running a binary classification, leave it as default or specify "binary" for average
        return self.metric.compute(predictions=predictions, references=labels, average="micro")

    def train(self, model_name):
        # Call the Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

        # Encode the text
        train_encodings = tokenizer(self.X_train, truncation=True, padding=True, max_length=self.max_length)
        valid_encodings = tokenizer(self.X_test, truncation=True, padding=True, max_length=self.max_length)
        # convert our tokenized data into a torch Dataset
        train_dataset = MakeTorchData(train_encodings, self.y_train.ravel())
        valid_dataset = MakeTorchData(valid_encodings, self.y_test.ravel())

        # Call Model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels).to("cpu")

        training_args = TrainingArguments(
            output_dir='./racism_results',  # output directory
            num_train_epochs=self.num_epochs,  # total number of training epochs
            per_device_train_batch_size=32,  # batch size per device during training
            per_device_eval_batch_size=20,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
            metric_for_best_model=self.metrics_name,  # select the base metrics
            logging_steps=200,  # log & save weights each logging_steps
            save_steps=200,
            evaluation_strategy="epoch",  # evaluate each `logging_steps`
            save_strategy='epoch'
        )

        # Call the Trainer
        trainer = Trainer(
            model=model,  # the instantiated Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=valid_dataset,  # evaluation dataset
            compute_metrics=self.compute_metrics,  # the callback that computes metrics of interest
        )

        # Train the model
        trainer.train()

        # Call the summary
        res = trainer.evaluate()
        with open('REPORT/classification_report_racism.txt', 'a+') as tgt_file:
            tgt_file.write(f"{model_name} : {res['eval_f1']}\n")
        print(f"{model_name} : {res['eval_f1']}")


if __name__ == '__main__':
    # df = pd.read_csv("DATA/bbc-text.csv")[:10]
    n_rows = 2882
    classifier = ClassificationAccuracyMeasurement('DATA/sentiment_racism.csv', n_rows)
    sen_transformers = ['./bangla_transformer','./bangla_snt','sentence-transformers/all-MiniLM-L6-v2',
                        'sentence-transformers/all-mpnet-base-v2',
                        'sentence-transformers/stsb-xlm-r-multilingual']

    # sen_transformers = ['sentence-transformers/all-mpnet-base-v2',
    #                     'sentence-transformers/stsb-xlm-r-multilingual','l3cube-pune/indic-sentence-bert-nli','l3cube-pune/indic-sentence-similarity-sbert']
    for sen_trans in sen_transformers:
        print(f'_____________________________{sen_trans}________________________________________')
        classifier.train(sen_trans)


    # sbert_trainer, sbert_model = TextClassification_with_Transformer(
    #     model_name='sentence-transformers/all-mpnet-base-v2',
    #     Data=df.text,
    #     Target=df.category,
    #     test_size=0.33,
    #     max_length=512,
    #     num_labels=5,
    #     num_epochs=3,
    #     metrics_name='f1')
    # print(0)
