import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, load_checkpoint, save_checkpoint

from torchtext.datasets import Multi30k
from torchtext.data import BucketIterator, Field

spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(init_token="<sos>", eos_token="<eos>", lower =True, tokenize=tokenize_ger)
english = Field(init_token="<sos>", eos_token="<eos>", lower =True, tokenize=tokenize_eng)

train_data, valid_data, test_data = Multi30k.splits(exts=(".de",".en"), fields= (german,english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def  __init__(self,
        embeding_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        src_vocab_size,
        trg_vocab_size,
        forward_expansion,
        max_len,
        dropout,
        src_pad_idx,
        device):
        super(Transformer, self).__init__()
        self.device = device
        self.src_word_embedding = nn.Embedding(src_vocab_size, embeding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embeding_size)
        self.src_position_embedding = nn.Embedding(max_len, embeding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embeding_size)
        self.transformer = nn.Transformer(embeding_size, num_heads, num_encoder_layers,\
            num_decoder_layers, forward_expansion, dropout)
        
        self.fc_out = nn.Linear(embeding_size, trg_vocab_size)
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim = -1)
        self.src_pad_idx = src_pad_idx
    def make_src_mask(self, src):
        src_mask = src.transpose(0,1) == self.src_pad_idx
        return src_mask

    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape

        src_positions = torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N).to(self.device)
        trg_positions = torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N).to(self.device)

        embed_src = self.dropout(self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        embed_trg = self.dropout(self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask= src_padding_mask,\
            tgt_mask= trg_mask)
        out = self.softmax(self.fc_out(out))
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = False

num_epochs = 5
learning_rate = 1e-3
batch_size = 32

src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embeding_size = 512
num_heads = 8 
num_encoder_layers = 3
num_decoder_layers = 3
forward_expansion = 4
max_len = 100
dropout = 0.1
src_pad_idx = german.vocab.stoi["<pad>"]

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), \
    batch_size = batch_size, sort_within_batch= True, sort_key= lambda  x: len(x.src), device= device)

model = Transformer(embeding_size,num_heads, num_encoder_layers, num_decoder_layers,\
    src_vocab_size, trg_vocab_size, forward_expansion, max_len, dropout, src_pad_idx, device).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index= pad_idx)

if load_model:
    load_checkpoint(torch.load("mycheckpoint.pth.tar"), model, optimizer)

sentence = "ein pferd geht unter einer brücke neben einem boot."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch}/{num_epochs}]")
    # # if save_model:
    # checkpoint = {
    #     "state_dict": model.state_dict(),
    #     "optimizer": optimizer.state_dict()
    # }
    # model.eval()
    # model.load_state_dict(checkpoint["state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # translated_sentence = translate_sentence(model, sentence, german, \
    #     english, device)

    # print(f"Translate sentence : \n {translated_sentence}")
    # model.train()
    model.eval()
    translated_sentence = translate_sentence(model, sentence, german, \
        english, device)

    print(f"Translate sentence : \n {translated_sentence}")
    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target[:-1,:])
        # print(output[0,0,:])
        output = output.reshape(-1, output.shape[2])
        #print(target.shape)
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()







        

