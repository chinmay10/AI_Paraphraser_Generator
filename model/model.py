import torch
from transformers import BartForConditionalGeneration, BartTokenizer

input_sentence = "They were there to enjoy  with us and they were there to pray for us."

model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
batch = tokenizer(input_sentence, return_tensors='pt').to(device)

torch.manual_seed(0)
generated_ids_set = model.generate(batch['input_ids'], num_beams=30, num_return_sequences=10, early_stopping=True)
for generated_ids in generated_ids_set: 
  generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
  print(generated_sentence)


