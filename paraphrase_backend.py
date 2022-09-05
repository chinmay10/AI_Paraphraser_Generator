
from fastapi import FastAPI
import torch
from sentence_transformers import SentenceTransformer, util
import pylev
import Request

# Initializing models and tokenizers
app = FastAPI()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = T5TokenizerFast.from_pretrained('./Model')
model = T5ForConditionalGeneration.from_pretrained('./Model').to(device)
sen_model = SentenceTransformer('paraphrase-MiniLM-L3-v2').to(device)

# Format used in training to tell the model how much to vary inputted text
distance = {1: "small", 2: "medium", 3: "large", 4: "gigantic"}
word_change = {1: "reduce", 2: "match", 3: "expand"}

# Formats source text into the format used in the training data. 
def get_input_string(text, word_count_key, distance_key):
    return f"Paraphrase: {distance[distance_key]} changes, {word_change[word_count_key]} input. {text}"


# Calculates cosine similarity between sentence vectors of source and paraphrase
def get_similarity(a, b):
    a = sen_model.encode(a, convert_to_tensor=True)
    b = sen_model.encode(b, convert_to_tensor=True)
    return util.pytorch_cos_sim(a, b).item()

# Calculates the levenschtein distance to see how much the model varied the original text
def get_distance(a, b):
    return pylev.levenschtein(a.split(), b.split())

# Calls the model and returns the paraphrased output and some simple metrics. 
@app.post('/paraphrase/')
async def main(item: Request):
    userInput = await item.json()
    input_string = get_input_string(
        userInput['text'], userInput['wordCount'], userInput['distance'])
    tokenized_input = tokenizer(input_string, return_tensors='pt').to(device)
    encoded_output = model.generate(**tokenized_input, max_length=1000)[0]
    decoded = tokenizer.decode(encoded_output, skip_special_tokens=True)
    return {
        'Paraphrase': decoded, 
        'Similarity': get_similarity(decoded, userInput['text']), 
        "distance": get_distance(decoded, userInput['text'])}