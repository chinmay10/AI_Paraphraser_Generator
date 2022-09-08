import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from transformers import BartForConditionalGeneration, BartTokenizer
import warnings
import pandas as pd
import os
from datetime import datetime
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter




logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

#data preprocessing
def string_cleaning(str1):
    if not isinstance(str1, str):
        warnings.warn(f">>> {str1} <<< is not a string.")
        str1 = str(str1)
    str1 = (
        str1.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return str1

    
#loading data
file_name=r'quora_duplicate_questions.tsv'
df = pd.read_csv(file_name, sep="\t", error_bad_lines=False)
df = df.loc[df["is_duplicate"] == 1]
df = df.rename(columns={"question1": "input_text", "question2": "target_text"})
df = df[["input_text", "target_text"]]
df["prefix"] = "paraphrase"

train_df, eval_df = train_test_split(df)



train_df = train_df[[ "input_text", "target_text"]]
eval_df = eval_df[[ "input_text", "target_text"]]

train_df = train_df.dropna()
eval_df = eval_df.dropna()

train_df["input_text"] = train_df["input_text"].apply(string_cleaning)
train_df["target_text"] = train_df["target_text"].apply(string_cleaning)

eval_df["input_text"] = eval_df["input_text"].apply(string_cleaning)
eval_df["target_text"] = eval_df["target_text"].apply(string_cleaning)

train_dataloader = DataLoader(train_df, batch_size=64,shuffle=True, num_workers=1)
test_dataloader = DataLoader(eval_df, batch_size=64,shuffle=True, num_workers=1)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)

#training
learning_rate = 5e-5
batch_size = 64
epochs = 5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(test_dataloader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    
    
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1








# input_sentence = "They were there to enjoy  with us and they were there to pray for us."

# model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
# batch = tokenizer(input_sentence, return_tensors='pt').to(device)

# torch.manual_seed(0)
# generated_ids_set = model.generate(batch['input_ids'], num_beams=30, num_return_sequences=10, early_stopping=True)
# for generated_ids in generated_ids_set: 
#   generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#   print(generated_sentence)


