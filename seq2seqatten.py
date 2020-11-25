import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
# from torch.utils.tensorboard import SummaryWriter

spacy_eng = spacy.load('en')
spacy_ger = spacy.load('de')

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

english = Field(tokenize= tokenize_eng, lower = True, init_token= '<sos>', eos_token='<eos>')
german = Field(tokenize= tokenize_ger, lower = True,init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts = ('.de','.en'),fields = (german,english))

english.build_vocab(train_data, max_size= 10000, min_freq=2)
german.build_vocab(train_data, max_size= 10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self,input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size,num_layers,dropout=p, bidirectional= True)
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

    def forward(self,x):
        embedding = self.dropout(self.embedding(x))
        encoder_states, (hidden,cell) = self.rnn(embedding)
        hidden = self.fc_hidden(torch.cat((hidden[0:1],hidden[1:2]),dim = 2))
        cell = self.fc_cell(torch.cat((cell[0:1],cell[1:2]),dim = 2))
        return encoder_states,hidden,cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers,p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size*2+embedding_size,hidden_size,num_layers,dropout=p)
        self.energy = nn.Linear(hidden_size*3,1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self,x,encoder_states, hidden,cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length,1,1)
        energy  = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim = 2)))
        attention = self.softmax(energy)
        attention = attention.permute(1,2,0)
        encoder_states = encoder_states.permute(1,0,2)
        context_vector = torch.bmm(attention,encoder_states).permute(1,0,2)
        rnn_input = torch.cat((context_vector,embedding),dim = 2)
        outputs, (hidden,cell) = self.rnn(rnn_input,(hidden,cell))
        predictions = self.fc(outputs).squeeze(0)
        return predictions,hidden,cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self,source, target, teacher_force_ratio =0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)
        outputs = torch.zeros(target_len,batch_size,target_vocab_size).to(device)
        encoder_states, hidden,cell = self.encoder(source)
        x  = target[0]
        for i in range(1,target_len):
            output, hidden, cell = self.decoder(x,encoder_states, hidden,cell)
            outputs[i] = output
            best_guess = output.argmax(1)
            x = target[i] if random.random()<0.5 else best_guess

        return outputs
#hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64
 
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# writer = SummaryWriter(f'runs/Loss_plot')
# step = 0
train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
        (train_data,validation_data,test_data), batch_size= batch_size,sort_within_batch=True,
        sort_key=lambda x: len(x.src),device=device
    )

encoder_net = Encoder(input_size_encoder,encoder_embedding_size,hidden_size,num_layers,enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder,decoder_embedding_size,hidden_size,num_layers,dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

for epoch in range(num_epochs):
    print(f'Epoch[{epoch}/{num_epochs}]')
    # step = 0
    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        output = model(inp_data, target)
        output = output[1:].reshape(-1,output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output,target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
        optimizer.step()
        # writer.add_scalar('Training loss', loss, global_step=step)
        # step+=1

