# %%
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from datasets import load_dataset, Dataset
from transformers import MarianMTModel, MarianTokenizer, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
# %%
corpus_test_file_path = "./Task 3 - translation/EN-FR-train/joker_task3_2024_test.json"
corpus_train_file_path = "./Task 3 - translation/EN-FR-train/joker_translation_EN-FR_train_input.json"
qrels_train_file_path = "./Task 3 - translation/EN-FR-train/joker_translation_EN-FR_train_qrels.json"

with open(corpus_test_file_path, 'r') as f:
    corpus_test_data = json.load(f)

with open(corpus_train_file_path, 'r') as f:
    corpus_train_data = json.load(f)

with open(qrels_train_file_path, 'r') as f:
    qrels_train_file_data = json.load(f)

# %%
corpus_test_df = pd.DataFrame(corpus_test_data)
corpus_train_df = pd.DataFrame(corpus_train_data)
qrels_train_df = pd.DataFrame(qrels_train_file_data)
# %%
train_data = pd.merge(corpus_train_df, qrels_train_df, on="id_en")

# %%
def process_data(data):
    source_texts = data["text_en"].values
    target_texts = data["text_fr"].values
    return source_texts, target_texts


# %%
source_texts, target_texts = process_data(train_data)
train_src, valid_src, train_tgt, valid_tgt = train_test_split(source_texts, target_texts, test_size=0.2)

train_dataset = Dataset.from_dict({
    'translation': [{'en': src, 'fr': tgt} for src, tgt in zip(train_src, train_tgt)]
})
valid_dataset = Dataset.from_dict({
    'translation': [{'en': src, 'fr': tgt} for src, tgt in zip(valid_src, valid_tgt)]
})
# %%
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
# %%
def preprocess_function(examples):
    inputs = [ex['en'] for ex in examples['translation']]
    targets = [ex['fr'] for ex in examples['translation']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
valid_tokenized_dataset = valid_dataset.map(preprocess_function, batched=True)
# %%
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=valid_tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
# %%
trainer.train()
# %%
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
# %%
