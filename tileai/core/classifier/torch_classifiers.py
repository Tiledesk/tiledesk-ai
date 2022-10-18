from torch import nn
from torch.nn import functional as F

class EmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, n_target_classes):
        super(EmbeddingClassifier, self).__init__()
        
        self.seq = nn.Sequential(
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=40),#25

            nn.Flatten(),

            nn.Linear(40*20, 256), ## 25 = embeding length, 50 = words we kept per text example lunhgehzza delle frai precendente 50
            nn.ReLU(),

            nn.Linear(256,128),
            nn.ReLU(),


            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64, n_target_classes),
        )

    def forward(self, X_batch):
        return self.seq(X_batch)

class EmbeddingClassifierAverage(nn.Module):
    def __init__(self,vocab_size, n_target_classes):
        super(EmbeddingClassifierAverage, self).__init__()
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=64)
        self.linear1 = nn.Linear(64, 128) ## 25 = embeding length, 50 = words we kept per sample
        self.linear2 = nn.Linear(128,64)
        self.linear3 = nn.Linear(64, n_target_classes)

    def forward(self, X_batch):
        x = self.word_embeddings(X_batch)
        x = x.mean(dim=1) ## Averaging embeddings

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = F.relu(self.linear3(x))

        return logits

class EmbeddingClassifierWBag(nn.Module):
    def __init__(self,vocab_size, n_target_classes):
        super(EmbeddingClassifierWBag, self).__init__()
        self.seq = nn.Sequential(
            nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=64, mode="mean"),

            nn.Linear(64, 128), ## 25 = embeding length, 50 = words we kept per sample
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, n_target_classes),
        )

    def forward(self, X_batch):
        return self.seq(X_batch)

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, n_target_classes):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim=512, sparse=True)
        
        self.lay1 = nn.Linear(512, 256)## 25 = embeding length, 50 = words we kept per sample
        self.relu1 = nn.ReLU()

        self.la2 = nn.Linear(256, 128)
        self.relu2= nn.ReLU()
                
        self.fc = nn.Linear(128, n_target_classes)

        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
      
        embedded = self.embedding(text, offsets)
        embedded = self.lay1(embedded)
        embedded = self.relu1(embedded)
        embedded = self.la2(embedded)
        embedded = self.relu2(embedded)
        
        return self.fc(embedded)

class LSTMClassificationModel(nn.Module):

    def __init__(self, vocab, target_classes):
        super(LSTMClassificationModel, self).__init__()
        #self.hidden_dim = 32
        #self.dropout = nn.Dropout(0.3)
        self.embedding = nn.Embedding(len(vocab), embedding_dim=64, padding_idx=0)
        
        self.lstm = nn.LSTM(input_size = 64, 
                           hidden_size = 32, 
                           num_layers  = 1, 
                           dropout     = 0.3,
                           batch_first=True)
                
        self.fc = nn.Linear(32, len(target_classes))

        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text): 
        
        embedded = self.embedding(text)
       
        _, (hidden, _) = self.lstm(embedded)
        #lstm_out, _ = self.lstm(embedded.view(len(text), 1, -1))
        #tag_space = self.fc(lstm_out.view(len(text), -1))
        #tag_scores = F.log_softmax(tag_space, dim=1)
        #return tag_scores
        
        return self.fc(hidden[-1])


