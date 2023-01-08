from django.http import JsonResponse
from rest_framework.decorators import api_view
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


@api_view(['POST'])
def paraphrase(request):
    text = request.POST.get('text')
    input_ids = torch.tensor([bart.encode(text)])
    output = bart.generate(input_ids, num_beams=5, max_length=len(text))
    paraphrased_texts = []
    for out in output:
        paraphrased_text = bart.decode(out, skip_special_tokens=True)  # Implement this function to do the actual paraphrasing
        similarity = get_similarity(paraphrase, text)  # Implement this function to compute the similarity score
        paraphrased_texts.append({
            'text': paraphrased_text,
            'similarity': similarity,
        })
    return JsonResponse(paraphrased_texts, safe=False)

