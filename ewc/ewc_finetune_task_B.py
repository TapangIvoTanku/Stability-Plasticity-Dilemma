from transformers import AdamW, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments, get_scheduler, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json
import evaluate

from ewc_class import EWC


model_id = "/content/flan-t5-large-task-A-full-finetune" #base-model
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_id)
ewc_model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

# Params
MAX_SOURCE_LENGTH = 768
MAX_TARGET_LENGTH = 512
LEARNING_RATE = 2e-5
NUM_EPOCHS = 20

#load ds
ds_A = []
with open("/content/data/task_A/task_A_test_ds.jsonl", 'r') as file:
  for line in file:
    ds_A.append("Summarize: " + json.loads(line)['input'])

ds_B = load_dataset(
    "json", data_files={"train": "/content/data/task_B/task_B_train_ds.jsonl", "eval": "/content/data/task_B/task_B_test_ds.jsonl"}
)

def preprocess_function_details(sample, padding="max_length", prefix=''):
    # add prefix to the input for t5
    inputs = [prefix + item for item in sample["input"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, padding=padding, truncation=True, return_tensors="pt")

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["output"], max_length=MAX_TARGET_LENGTH, padding=padding, truncation=True, return_tensors="pt")

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
    return model_inputs


ds_B = ds_B.map(
    lambda sample: preprocess_function_details(sample, prefix="Question: "),
    batched=True,
    desc="Running tokenizer on dataset"
)

loader= DataLoader(ds_B['train'], shuffle=True, batch_size=2)

print('Calculating Fisher Information Matrix...')
ewc = EWC(ewc_model, ds_A)
print('Done! Starting training...')

optimizer = AdamW(ewc_model.parameters(), lr=LEARNING_RATE)
num_training_steps = NUM_EPOCHS * len(loader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

ds_A = ds_A.map(
    lambda sample: preprocess_function_details(sample, prefix=""), #prefix is empty since the "Summarize: " is already included
    batched=True,
    desc="Running tokenizer on dataset"
)

#metrics
rouge = evaluate.load("rouge")
bleuscore = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
def eval_model(model, ds):
    progress_bar = tqdm(range(len(ds)))
    model.eval()
    shuffled = ds.shuffle()
    b_size = 8
    start = 0
    end = b_size
    model.eval()
    predictions = []
    references = []
    while True:
        examples = shuffled[start: end]
        input_ids = tokenizer(examples['input'], return_tensors="pt", padding=True).input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids)

        predictions.extend([s for s in tokenizer.batch_decode(outputs, skip_special_tokens=True)])
        references.extend([s for s in examples['output']])
        progress_bar.update(len(examples['input']))
        start = end
        end = end + b_size
        del input_ids
        del outputs
        if end > len(ds):
            break

    rouge_score = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleuscore.compute(predictions=predictions, references=references)
    bert_score = bertscore.compute(references = references, predictions=predictions, model_type="distilbert-base-uncased")
    return {"rouge_score": rouge_score,
            "bleu_score": bleu_score,
            "bert_score": {"precision": sum(bert_score['precision'])/len(bert_score['precision']), 
                           'f1': sum(bert_score['f1'])/len(bert_score['f1'])}}

ewc_history = {'task_A': [], 'task_B': []}

progress_bar = tqdm(range(num_training_steps))
model_batch_loss = []
model_ewc_batch_loss = []
for epoch in range(NUM_EPOCHS):
    for batch in loader:
        input_ids = tokenizer(batch['input'], return_tensors="pt", padding=True).input_ids.to(device)
        labels = tokenizer(batch['output'], return_tensors="pt", padding=True).input_ids.to(device)
        outputs = ewc_model(input_ids=input_ids, labels=labels)
        ewc_loss = outputs.loss + 1000 * ewc.penalty(ewc_model)
        model_batch_loss.append(outputs.loss)
        model_ewc_batch_loss.append(ewc_loss)

        ewc_loss.backward()
        optimizer.step()

        lr_scheduler.step()
        optimizer.zero_grad()
        del input_ids
        del labels
        del outputs
        progress_bar.update(1)

    task_A_performance = eval_model(ewc_model, ds=ds_A)
    ewc_history['task_A'].append({'metrics': task_A_performance})
    
    task_B_performance = eval_model(ewc_model, ds=ds_B['eval'])
    ewc_history['task_B'].append({'metrics': task_B_performance})

