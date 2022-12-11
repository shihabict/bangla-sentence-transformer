from datasets import load_dataset

ai_dataset = load_dataset("ai4bharat/samanantar", 'bn').data['train'].to_pandas()
ai_dataset.to_csv('DATA/aibarat_transcription_data.csv')
print(0)
