import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import random
import ipdb
from utils import translate_sentence, load_checkpoint, save_checkpoint

from torchtext.datasets import Multi30k
from torchtext.data import BucketIterator, Field

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention,self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size//heads
        assert(self.head_dim*heads == embed_size), "heads has to divide embed_size"

        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.values = nn.Linear(self.embed_size, self.embed_size,bias=False)
        self.fc_out = nn.Linear(self.head_dim*heads, embed_size)
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, query_len, key_len = values.shape[1], query.shape[1], keys.shape[1]

        values = self.values(values)
        queries = self.queries(query)
        keys =  self.keys(keys)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy/(self.embed_size**(1/2)), dim = 3)
        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values]).reshape(N, query_len, self.head_dim*self.heads)
        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention+query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out
    
class Encoder(nn.Module):
    def __init__(self,src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder,self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(embed_size,heads, dropout,forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device=self.device)
        out = self.dropout(self.word_embedding(x)+self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock,self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformerblock = TransformerBlock(embed_size, heads,dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, value, key , src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformerblock(value, key ,query , src_mask)
        return out
class Decoder(nn.Module):
    def __init__(self,trg_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([DecoderBlock(embed_size,heads,forward_expansion, dropout,device) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

    def forward(self, x, enc_out, src_mask, trg_mask ):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device=self.device)
        x = self.dropout(self.word_embedding(x)+self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(self,src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,device, embed_size,\
        num_layers, forward_expansion, heads, dropout, max_length):
        super(Transformer,self).__init__()
        self.encoder = Encoder(src_vocab_size,embed_size, num_layers, heads,device, forward_expansion,dropout, max_length)
        self.decoder = Decoder(trg_vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self._reset_parameters()

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    def make_trg_mask(self,trg):
        N,trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len,trg_len)
        return trg_mask.to(self.device)

    def forward(self,src,trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src,src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        if random.random()>0.5:
            inp = out.argmax(dim=-1)
            out = self.decoder(inp, enc_src, src_mask, trg_mask)
        return out
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 201
learning_rate = 3e-4
batch_size = 32

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

src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
src_pad_idx = german.vocab.stoi["<pad>"]
trg_pad_idx = english.vocab.stoi["<pad>"]
embed_size = 512
num_layers = 3
forward_expansion = 4
heads = 8
dropout = 0.1
max_length = 100

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), \
    batch_size = batch_size, sort_within_batch= True, sort_key= lambda  x: len(x.src), device= device)

model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,device, embed_size,\
        num_layers, forward_expansion, heads, dropout, max_length).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index= pad_idx)
sentence = "ein pferd geht unter einer brücke neben einem boot."
ipdb.set_trace()
for epoch in range(1, num_epochs):
    print(f"[Epoch {epoch}/{num_epochs}]")
    if epoch%10 == 0:
        checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, "mycheckpoint_"+str(epoch)+".pth.tar")

        model.eval()
        translated_sentence = translate_sentence(model, sentence, german, \
            english, device)
        print(f"Translated sentence : \n {translated_sentence}")
        model.train()
    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        output = model(inp_data, target[:,:-1])
        output = output.reshape(-1, output.shape[2])
        target = target[:,1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.tensor([[1,5,6,4,3,9,5,2,0],[1,8,7,3,4,5,6,7,2]]).to(device)
#     trg = torch.tensor([[1,5,6,2,4,7,6,2],[1,7,4,3,5,9,2,0]]).to(device)
#     src_pad_idx = 0
#     trg_pad_idx = 0
#     src_vocab_size = 10
#     trg_vocab_size = 10
#     model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device).to(device)
#     out = model(x, trg[:,:-1])
#     print(out.shape)

