import torch
import spacy
import sys
import ipdb
from torchtext.data.metrics import bleu_score

def translate_sentence(model, sentence, german, english, device, max_length = 50):
    spacy_ger = spacy.load("de_core_news_sm")
    if type(sentence) == str:
        tokens = [tok.text.lower() for tok in spacy_ger(sentence)]
    else:
        tokens = [tok.lower() for tok in sentence]
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    text_to_indices = [german.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(0).to(device)
    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor, train = False)
        best_guess = output.argmax(2)[:, -1].item()
        outputs.append(best_guess)
        if best_guess == english.vocab.stoi["<eos>"]:
            break
    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]

def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def save_checkpoint(state, filename = "mycheckpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state, filename)
def load_checkpoint(checkpoint, model, optimizer):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
