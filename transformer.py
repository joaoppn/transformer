
import numpy as np
import pandas
import torch
import torch.nn.functional as F



#Organizing Data
data = ["O cachorro correu do homem", "A mulher é uma rainha", "o homem é rei", "o homem tem um gato"]
tokens = []
vocab = []
for sentence in data:
    aux= []
    for word in sentence.split():
        aux.append(word)
        vocab.append(word)
    tokens.append(aux)
vocab = sorted(set(list(vocab)))
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
encode(tokens[0])


#Simple Self-Attention
class self_attention_simple:
    def __init__(self, tokens, embedding_size):

        #parameters
        self.emb_n = embedding_size
        self.dk = tokens.shape[0]
        self.X = tokens

        
        #generating ramdom initial weights for Query, Keys adn Values
        self.wq = np.random.randn(self.dk, self.emb_n)
        self.wk = np.random.randn(self.dk, self.emb_n)
        self.wv =  np.random.randn(self.dk, self.emb_n)
        print(f"Wquery = ({self.wq.shape})")
        print(f"Wkey = ({self.wk.shape})")
        print(f"Wvalue = ({self.wv.shape})")


        #calculating Queries, Keys and Values
        self.keys = np.matmul(self.wk,self.X.T).T
        self.query = np.matmul(self.wq, self.X.T)
        self.values = np.matmul(self.wv, self.X.T).T
        print(f"Query[q]({self.query.shape}) = {self.query}")
        print(f"Key[K]({self.keys.shape}) = {self.keys}")
        print(f"Values[v]({self.values.shape}) = {self.values}")
    
    def softmax(self,x):
        x = (x - x.mean()) / x.std()  #in case of overflow
        return np.exp(x) / sum(np.exp(x))
    
    def score(self, Q, K):
        W = Q.dot(K)
        print(f"Scores({W.shape}) = {W}")
        return W
    
    def attention_value(self):
        score = self.score(self.query, self.keys)
        score_probs = self.softmax(score)
        print(f"Scores softmax({score_probs.shape}) = {score_probs}")
        attention = (self.values.dot(score_probs))
        print(f"Attention Value({attention.shape}) = {attention}")
        
        return attention 

#MultiHead-Attention
class multiHead_attention: 
    def __init__(self, n_head, tokens, emb_n):
        self.tokens = tokens #number of tokens in the input
        self.n_head = n_head #number of headatentions
        self.emb_n = emb_n
    def values(self):
        self.Sattention = self_attention(self.n_head, self.tokens,  self.emb_n)
        return self.Sattention.attention_value()
    def get_query(self):
        SA = self.Sattention
        return SA.query
    def cross(self, new_Query):
        return self_attention(self.n_head,self.tokens, self.emb_n).cross_value(new_Query)

#Self-Attention for multiple heads with cross attention
class self_attention:
    def __init__(self, heads, tokens, embedding_size):

        #parameters
        self.emb_n = embedding_size
        self.dk = 25
        self.heads = heads
        self.X = torch.tensor(tokens).T.repeat(3,1,1).float() #multiply for headattention
        print(f"Tokens ({self.X.shape}) = {self.X}")
        
       #generating ramdom initial weights for Query, Keys adn Values
        self.wq = torch.nn.Parameter(torch.randn(self.heads, self.dk, self.emb_n))
        self.wk = torch.nn.Parameter(torch.randn(self.heads,  self.dk,self.emb_n))
        self.wv =  torch.nn.Parameter(torch.randn(self.heads, self.dk, self.emb_n))
        print(f"Wquery = ({self.wq.shape})")
        print(f"Wkey = ({self.wk.shape})")
        print(f"Wvalue = ({self.wv.shape})")


        #calculating Queries, Keys and Values
        self.keys = torch.bmm(self.wk,self.X)
        self.query = torch.bmm(self.wq, self.X)
        self.values = torch.bmm(self.wv, self.X)
        print(f"Query[q]({self.query.shape}) = {self.query}")
        print(f"Key[K]({self.keys.shape}) = {self.keys}")
        print(f"Values[v]({self.values.shape}) = {self.values}")
    
    def softmax(self,x):
        x = (x - x.mean()) / x.std()  #in case of overflow
        return torch.exp(x) / sum(torch.exp(x))
    
    def score(self, Q, K):
        Q = Q.reshape(self.heads, -1, self.dk) #reshape for multiply Q x K
        print(f"Q({Q.shape} X K{K.shape})")
        W = torch.matmul(Q, K)
        print(f"Scores({W.shape}) = {W}")
        return W
    
    def attention_value(self):
        score = self.score(self.query, self.keys)
        score_probs = self.softmax(score)
        print(f"Scores softmax({score_probs.shape}) = {score_probs}")
        print(f"Values{self.values.shape}score_probs({score_probs.shape}")
        attention = torch.bmm(self.values, (score_probs.reshape(self.heads, self.values.shape[2], -1)))
        print(f"Attention Value({attention.shape}) = {attention}")
        
        return attention 
    
    def cross_value(self, newQ): #takes another Query and compares with current Score(Q x K)
        score = self.score(newQ, self.keys) 
        score_probs = self.softmax(score)
        print(f"Scores softmax({score_probs.shape}) = {score_probs}")
        print(f"Values{self.values.shape} Score_probs({score_probs.shape}")
        attention = torch.bmm(self.values, (score_probs.reshape(self.heads, self.values.shape[2], -1)))
        print(f"Cross Attention Value({attention.shape}) = {attention}")
        
        return attention 

#Trnasformer calling self-attention classes    
class transformer:
    def __init__(self, sentences, vocab_n, emb_n):
        #Parameters
        self.vocab_n = vocab_n
        self.emb_n = emb_n
        self.inputs = [] #list of sentences

        #Encoding
        for sentence in sentences:
            self.inputs.append(encode(sentence))

        print(f"Sentences = {tokens}")
        print(f"tokens = {self.inputs}")


        #Embedding
        self.emb_table =np.random.randn(vocab_n, emb_n)
    
    def cross_attention(self): 
        #calculating self-attention
        print(f"Input[0] = {self.inputs[0]}")
        input1 = self.emb_table[self.inputs[0]] #sentence 1
        input2 = self.emb_table[self.inputs[2]] #sentence 2
        print(f"--------------------ATTENTION 1----------------------")
        print(f"Sentence = {data[0]}")
        attention1 = multiHead_attention(3,input1, self.emb_n) #multihead_attention of sentence 1
        attention1.values()
        print(f"--------------------ATTENTION 2----------------------")
        print(f"Sentence = {data[2]}")
        attention2 = multiHead_attention(3,input2, self.emb_n) #multihead_attention of sentence 2
        attention2.values()
        print(f"--------------------CROSS ATTENTION----------------------")
        print(f"Sentence Query = {data[0]}")
        print(f"Sentence Key = {data[2]}")
        cros12 =  multiHead_attention(3,input2, self.emb_n).cross(attention1.get_query()) #cros attention of sentece 2 with Query from sentence 1
        
    def simple(self): #simple self-attention
        input1 = self.emb_table[self.inputs[0]]
        resultado = self_attention_simple(input1, self.emb_n).attention_value()
