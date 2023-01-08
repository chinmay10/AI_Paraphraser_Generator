
from fastapi import FastAPI
import torch
from sentence_transformers import SentenceTransformer, util
import pylev
from django.shortcuts import render
from django.http import JsonResponse
import transformers



# load the BART model 
bart = transformers.BartModel.from_pretrained('bart-large')

# Calculates cosine similarity between sentence vectors of source and paraphrase
def get_similarity(a, b):
    a = bart.encode(a, convert_to_tensor=True)
    b = bart.encode(b, convert_to_tensor=True)
    return util.pytorch_cos_sim(a, b).item()

def paraphrase(request):
    # get the input text from the user's request 
    input_text = request.GET.get('input_text', '')

    # encode the input text
    input_ids = torch.tensor([bart.encode(input_text)])

    # generate 5 paraphrased sentences
    output = bart.generate(input_ids, num_beams=5, max_length=len(input_text))

    # decode the output and compute the similarity scores
    paraphrases = []
    for out in output:
        paraphrase = bart.decode(out, skip_special_tokens=True)
        score =get_similarity(paraphrase, input_text)   # function to compute similarity score
        paraphrases.append((paraphrase, score))

    # return the paraphrased sentences and scores as a JSON response
    return JsonResponse({'paraphrases': paraphrases})