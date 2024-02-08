import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Define your NER model
class NERCRFModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NERCRFModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size , bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(2 * hidden_size, output_size)
        self.crf = CRF(output_size)

    def forward(self, inputs):
       
        embedded = self.embedding(inputs)
        lstm_out, _ = self.lstm(embedded)
        logits = self.hidden2tag(lstm_out)
        #emission_scores = nn.functional.log_softmax(logits, dim=-1)
        #return emission_scores    
        return logits

def train(model, train_dataloader, optimizer, criterion, device):
    model.train()
    for batch in train_dataloader: 
        
        inputs, tags = batch['text'].to(device), batch['tags'].to(device)
        #mask = (inputs != 0) 
        #mask[:, 0] = True
        #print(mask)
        #print(inputs[3],tags[3],mask[3])
        #inputs = inputs.to(device)
        #tags = tags.to(device)
        
        optimizer.zero_grad()
        emission_scores = model(inputs)
                
        loss = -criterion(emission_scores, tags)#, mask=mask)
        
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

def evaluate(model, valid_dataloader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_words = 0
        for batch in valid_dataloader:
            inputs, tags = batch['text'].to(device), batch['tags'].to(device)
            #mask = (inputs != 0) 
            emission_scores = model(inputs)
            loss = -criterion(emission_scores, tags)
            total_loss += loss.item()

            pred_tags = model.crf.decode(emission_scores)
            correct = (torch.tensor(pred_tags).t() == tags).sum().item() 
            
            total_correct += correct
            total_words += len(tags)

        accuracy = total_correct / total_words if total_words > 0 else 0.0
        print("Valid loss: {:.4f}, accuracy: {:.4f}".format(total_loss, accuracy))

        
# Prediction function
def predict(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['text'].to(device)

            emission_scores = model(inputs)
            pred_tags = model.crf.decode(emission_scores)
            correct = torch.tensor(pred_tags).t()
            predictions.extend(correct)

    
    return predictions

if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.getcwd())

    from tileai.core.preprocessing.data_reader import make_dataframe, read_from_json
    from tileai.core.preprocessing.crf_dataset import CRFClassifierDataset, make_list_entities_tags
    #from transformers import AutoTokenizer
    from tileai.core.tokenizer.standard_tokenizer import StandarTokenizer

    files = ["/home/lorenzo/Sviluppo/tiledesk/tiledesk-ai/domain/diet/nlu_diet.json"]

    data=[]
    for file in files:
        data += read_from_json(file=file)

    #tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    tokenizer = StandarTokenizer()


    df, entities_list, intents_list, synonym_dict = make_dataframe(data)
    words, tags = make_list_entities_tags(dataframe=df,  entities=entities_list)
    #print(words)
    #print(tags)
    # Create a vocabulary for words and tags
    word_vocab = {word for sent in words for word in sent} #{word for sent in sentences for word in sent['text']}
    tag_vocab = {tag for tag_seq in tags for tag in tag_seq} # {tag for tag_seq in sentences for tag in tag_seq['tags']}
    #print(word_vocab)
    #print(tag_vocab)
    
    word_to_idx = {word: idx + 1 for idx, word in enumerate(word_vocab)}
    tag_to_idx = {tag: idx + 1 for idx, tag in enumerate(tag_vocab)}
    #print(word_to_idx)
    print(tag_to_idx)
    

    # Convert tokens to indices
    texts_idx = [[word_to_idx[word] for word in sent] for sent in words]
    #texts_idx = [[word_to_idx[word] for word in sent['text']] for sent in sentences]
    tags_idx = [[tag_to_idx[tag] for tag in tag_sent] for tag_sent in tags] #tags_idx = [[tag_to_idx[tag] for tag in tag_seq['tags']] for tag_seq in sentences]
    #print(texts_idx)
    #print(tags_idx)
    

    # Pad sequences
    padded_texts = pad_sequence([torch.tensor(sent) for sent in texts_idx], batch_first=True)
    padded_tags = pad_sequence([torch.tensor(tag_seq) for tag_seq in tags_idx], batch_first=True)
    #print(padded_texts[0:2], padded_tags[0:2]) 

    # Split the dataset into training and validation sets
    split_ratio = 1.0
    split_index = int(len(words) * split_ratio)
    #print(split_index)

    train_dataset = CRFClassifierDataset(padded_texts[:split_index], padded_tags[:split_index])
    valid_dataset = CRFClassifierDataset(padded_texts[split_index:], padded_tags[split_index:])
    
    # Create data loaders
    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    INPUT_SIZE = len(word_vocab) +1   
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = len(tag_vocab) +1

    model = NERCRFModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

    optimizer = optim.Adam(model.parameters(),lr=0.01)
    criterion = model.crf
    

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, optimizer, criterion, device)
        #evaluate(model, valid_loader, criterion,device)

    print(predict(model, train_loader, device))





    