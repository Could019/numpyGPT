import numpy as np
from scipy.special import softmax
import pickle
import matplotlib.pyplot as plt

class Tokenizer:
     
    def __init__(self):

       self.vocab = []
       self.merges = []
       self.stoi = {}
       self.itos = {}

       self.EOS_TOKEN = b"<EOS>"
       self.vocab_size = 1000

    def merge(data , freq_pair):

        new_data = []
        i = 0
       
        while i < len(data):
          if i < (len(data) - 1) and (data[i] , data[i +1]) == freq_pair:
              new_data.append(data[i] + data[i + 1])
              i += 2
          else :
              new_data.append(data[i])
              i += 1
        return new_data
   
    def train(self , text):

        data = [bytes([b]) for b in text.encode("utf-8")]

        self.vocab = set(data)

        print("DEBUG data:", data, type(data))
        
        MAX_MERGES = 10
        MAX_TOKEN_LEN = 4

        while len(self.merges) < MAX_MERGES:
          #create the pair
          pairs = []
          for i in range(len(data) - 1):
             pairs.append((data[i] , data[i + 1]))

             #find the most freqent pair
             count = {}
          for p in pairs:
             if p in count:
               count[p] += 1
             else:
               count[p] = 1

          max_count = 0
          freq_pair = None

          for k , v in count.items():
               if v > max_count:
                  max_count = v
                  freq_pair = k

          if freq_pair is None:
            break

          if max_count <= 3:
            break

          self.merges.append(freq_pair)

          #merge
          data = Tokenizer.merge(data , freq_pair)

          if len(self.merges) > self.vocab_size:
             break

        for pair in self.merges:

          new_vocab = pair[0] + pair[1]
          self.vocab.add(new_vocab)

        self.vocab = sorted(self.vocab)

        self.vocab.append(self.EOS_TOKEN)

        for i , token in enumerate(self.vocab):
          self.stoi[token] = i
          self.itos[i] = token
        
        print(f'self.merges{self.merges}')
        print(f'self.vocab{self.vocab}')
        print(f'stoi{self.stoi}')
        print(f'itos{self.itos}')

        return self.vocab

       
    
    def encode(self , text):
       
       tokens = [bytes([b]) for b in text.encode("utf-8")]
       tokens.append(self.EOS_TOKEN)

       for pair in self.merges:
           tokens = Tokenizer.merge(tokens, pair)
       ids = [self.stoi[t] for t in tokens ]
       
       self.cache = ids

       return ids  
    
    def batch(self , B , T):
        
        ids = self.cache
        batch_list = []
        target_list = []
        max_start  = len(ids) - T - 1

        for _ in range(B):
           
           start = np.random.randint(0,max_start)
           batch = ids[start : start + T]
           target = ids[start + 1 : start + 1 + T]

           batch_list.append(batch)
           target_list.append(target)

        batch = np.array(batch_list)
        target = np.array(target_list)    

        return batch , target
    
    def decode(self , ids):
       
        tokens = [self.itos[i] for i in ids]

        text = b"".join(tokens).decode("utf-8")

        return text
    
    def save(self , path):
       
        data = {
          "vocab": self.vocab,
          "merge": self.merges,
          "stoi":  self.stoi,
          "itos":  self.itos,
          "EOS":   self.EOS_TOKEN
               }
        
        with open(path , "wb") as f:
           pickle.dump(data , f)

    def load(self , path):

       with open(path , "rb") as f:
          data = pickle.load(f)

       self.vocab = data["vocab"]
       self.merges = data["merge"]
       self.stoi = data["stoi"]
       self.itos = data["itos"]
       self.EOS_TOKEN = data["EOS"]
       
    def preprocessing(self , text):
        
        text = text.strip()

        return text

class Attention:

    def __init__(self):

        #Attention param
        self.paramters = {
        "WQ" : np.random.randn(512 , 512)*0.02,
        "WK" : np.random.randn(512 , 512)*0.02,
        "WV" : np.random.randn(512 , 512)*0.02
        }
        #Attention cache
        self.cache = None

        #Attention gradients
        self.gradients = {}

    def forward(self , vector):

        #Q , K , V
        Q = vector @ self.paramters["WQ"]
        K = vector @ self.paramters["WK"]
        V = vector @ self.paramters["WV"]

        B , T , d_model = Q.shape
        n_heads = 8
        head_dim = d_model // n_heads
     
        #change 02
        Q_split = Q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        K_split = K.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        V_split = V.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
      
        outputs = []
        attentions_weights = []
        for h in range(n_heads):
          qh = Q_split[:, h]
          kh = K_split[:, h]
          vh = V_split[:, h]   

          key_dimension = qh.shape[-1]
     
          attention_score_raw = (qh @ kh.transpose(0,2,1)) / np.sqrt(key_dimension)

          future_token_mask = np.triu(np.ones_like(attention_score_raw), k=1).astype(bool)

          attention_score_masked = np.where(future_token_mask, -1e10 , attention_score_raw)
          attention_score_stable = attention_score_masked - np.max(attention_score_masked , axis=-1 , keepdims=True)
          attention_weight = softmax(attention_score_stable , axis=-1)

          attention_output = attention_weight @ vh

          outputs.append(attention_output)
          attentions_weights.append(attention_weight)

        attention_output = np.stack(outputs, axis=1)
        attention_output = attention_output.transpose(0, 2, 1, 3)
        attention_output = attention_output.reshape(B, T, d_model)

        attention_weight = np.stack(attentions_weights, axis=1)

        self.cache = (Q , K , V , attention_weight , vector)

        return attention_output 
    
    def backward(self , upstream_gradient):

        Q , K , V , attention_weight , vector = self.cache

        B, T, d_model = Q.shape
        n_heads = 8
        head_dim = d_model // n_heads

        upstream_gradient = upstream_gradient.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3)
        attention_weight = attention_weight.reshape(B, T, n_heads, T).transpose(0,2,1,3)  # attention A'shape

        #change 03 : add a V'reshape
        V = V.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3)

        #change 04 : add a K'reshape
        K = K.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3)
        
        #change 05 : add a Q'reshape
        Q = Q.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3)

        d_Q_list , d_K_list , d_V_list = [], [], [] 

        for h in range(n_heads):
             
             d_V = attention_weight[:,h].transpose(0,2,1) @ upstream_gradient[:,h]

             #change 03 : add[:,h] behind V
             d_A = upstream_gradient[:,h] @ V[:,h].transpose(0,2,1)

             d_S = attention_weight[:,h] * (d_A - np.sum(d_A * attention_weight[:,h], axis=-1, keepdims=True))

             dk = Q[:,h].shape[-1]
             
             d_M = d_S / np.sqrt(dk)
             d_Q = d_M @ K[:,h]
             d_K = d_M.transpose(0,2,1) @ Q[:,h]

             d_Q_list.append(d_Q)
             d_K_list.append(d_K)
             d_V_list.append(d_V)
    
        d_Q_total = np.stack(d_Q_list, axis=1)   # (B,n_heads,T,head_dim)
        d_K_total = np.stack(d_K_list, axis=1)
        d_V_total = np.stack(d_V_list, axis=1)

        d_Q_total = d_Q_total.transpose(0,2,1,3).reshape(B,T,d_model)
        d_K_total = d_K_total.transpose(0,2,1,3).reshape(B,T,d_model)
        d_V_total = d_V_total.transpose(0,2,1,3).reshape(B,T,d_model)

        #change 06 : use paramters module
        grad_attention = (
          d_Q_total @ self.paramters["WQ"].T +
          d_K_total @ self.paramters["WK"].T +
          d_V_total @ self.paramters["WV"].T
             )

        #change 09 : a math wrong
        X = vector.reshape(-1 , d_model)
        dW_Q_total = X.T @ d_Q_total.reshape(-1 , d_model)
        dW_K_total = X.T @ d_K_total.reshape(-1 , d_model)
        dW_V_total = X.T @ d_V_total.reshape(-1 , d_model)

        self.gradients = {
            "WQ":dW_Q_total,
            "WK":dW_K_total,
            "WV":dW_V_total
             }
        
        return grad_attention   

class Layernorm:
    
    def __init__(self):

        #Layernorm param
        self.paramters = {
        "y_norm" : np.ones(512,),
        "b_norm" : np.zeros(512,)
        }
        #Layernorm cache
        self.cache = None

        #Layernorm gradients
        self.gradients = {}

    def forward(self , attention_output, residual_input,):    
        #Add First
        residual_output = attention_output + residual_input
        #Norm Second
        mean = np.mean(residual_output,axis=-1,keepdims=True)
        var = np.var(residual_output,axis=-1,keepdims=True,ddof=0)
        normalized_output = (residual_output - mean) / np.sqrt(var + 1e-5)
        layernorm_output = self.paramters["y_norm"] * normalized_output + self.paramters["b_norm"]

        self.cache = (residual_output , normalized_output , self.paramters["y_norm"])

        return layernorm_output
    
    def backward(self , upstream_gradient):

        residual_output , normalized_output , self.paramters["y_norm"] = self.cache

        d_y_norm = np.sum(upstream_gradient * normalized_output,axis=(0,1)) 
        d_b_norm = np.sum(upstream_gradient , axis=(0,1))
        grad_normalized = upstream_gradient * self.paramters["y_norm"]
        d = residual_output.shape[1]
        var = np.var(residual_output,axis=1,keepdims=True,ddof=0)
        std = np.sqrt(var + 1e-5)
        term_scaled_grad = d * grad_normalized
        term_mean_grad = np.sum(grad_normalized,keepdims=True,axis=1)
        term_projection = normalized_output * np.sum(grad_normalized * normalized_output,keepdims=True,axis=1)
        grad_residual_input = (1 / (d * std))*(term_scaled_grad - term_mean_grad - term_projection)
        grad_residual_branch =grad_residual_input

        self.gradients = {
            "y_norm":d_y_norm,
            "b_norm":d_b_norm
             }
        
        return grad_residual_input , grad_residual_branch
    
class FFN:

    def __init__(self):

        #FFN param
        self.paramters = {
        "W_1" : np.random.randn(512, 2048)*0.02,
        "b_1" : np.random.randn(2048,)*0.02,

        "W_2" : np.random.randn(2048, 512)*0.02,
        "b_2" : np.random.randn(512,)*0.02
        }
        #FFN cache
        self.cache = None

        #FFN gradients
        self.gradients = {}

    def forward(self , layernorm_output):

        ffn_hidden_linear = layernorm_output @ self.paramters["W_1"] + self.paramters["b_1"]
        ffn_hidden_activation = np.maximum(0, ffn_hidden_linear) 
        ffn_output = ffn_hidden_activation @ self.paramters["W_2"] + self.paramters["b_2"]
        
        self.cache = (ffn_hidden_activation , ffn_hidden_linear , layernorm_output)

        return ffn_output
    
    def backward(self , upstream_gradient):

        ffn_hidden_activation , ffn_hidden_linear , layernorm_output = self.cache

        d_W_2 = ffn_hidden_activation.transpose(0,2,1) @ upstream_gradient 
        d_W_2 = np.sum(d_W_2,axis=0)
        d_b_2 = np.sum(upstream_gradient , axis=(0,1))
        d_A1 = upstream_gradient @ self.paramters["W_2"].T
        d_Z1 = d_A1 * (ffn_hidden_linear > 0)
        d_W_1 = layernorm_output.transpose(0,2,1) @ d_Z1
        d_W_1 = np.sum(d_W_1,axis=0) 
        d_b_1 = np.sum(d_Z1 , axis=(0,1))
        grad_ffn = d_Z1 @ self.paramters["W_1"].T

        self.gradients = {
            "W_1": d_W_1 ,
            "b_1": d_b_1 ,
            "W_2": d_W_2 ,
            "b_2": d_b_2
        }

        return grad_ffn
    
class TransformerBlock:

    def __init__(self):

        self.attention = Attention()
        self.layernorm1 = Layernorm()
        self.ffn = FFN()
        self.layernorm2 = Layernorm()

    def forward(self , x):
        
        attention_output = self.attention.forward(x)

        layernorm1_output = self.layernorm1.forward(attention_output , x)

        ffn_output = self.ffn.forward(layernorm1_output)

        layernorm2_output = self.layernorm2.forward(ffn_output , layernorm1_output)

        return layernorm2_output
    
    def backward(self , dx):

        grad_residual2_input , grad_residual2_branch = self.layernorm2.backward(dx)

        grad_ffn = self.ffn.backward(grad_residual2_input)

        d_norm2_total = grad_residual2_branch + grad_ffn

        grad_residual1_input , grad_residual1_branch = self.layernorm1.backward(d_norm2_total)

        grad_attention = self.attention.backward(grad_residual1_input)

        d_norm1_total = grad_residual1_branch + grad_attention

        return d_norm1_total
    
class Model:
    
    def __init__(self):

        self.blocks = [
            TransformerBlock()
            ]
     
        self.token_embedding = np.random.randn(len(vocab) , 512)*0.02

        MAX_LEN = 512
        self.position_embedding = np.random.randn(MAX_LEN, 512)*0.02

        self.linear_prediction_b = np.random.randn(len(vocab),)*0.02
        
    def forward(self , batch):

        B , T = batch.shape

        token_vectors = self.token_embedding[batch]      # (B, T, 512)

        pos = np.arange(T)
        position_vectors = self.position_embedding[pos]  # (T, 512)

        vector = token_vectors + position_vectors[None, : , :]

        x = vector

        for block in self.blocks:

            x = block.forward(x)

        linear_prediction_w = self.token_embedding.T
        logit =  x @ linear_prediction_w + self.linear_prediction_b    

        self.cache = x
        self.cache_ids = batch

        return logit

    def  backward(self , dlogit):
 
        #change 11 : add a gradient from output token embedding
        x = self.cache
        B , T , V = dlogit.shape 
        D = x.shape[-1]
        
        dlogit_flat = dlogit.reshape(-1 , V)
        x_flat = x.reshape(-1 , D)

        d_token_embedding_from_output = dlogit_flat.T @ x_flat

        #change 08: add  a gradient of linear prediction
        d_linear_prediction_b = np.sum(dlogit_flat, axis = 0)
        #change 01
        dx = dlogit @ self.token_embedding 

        for block in reversed(self.blocks):
            dx = block.backward(dx)

        B , T = self.cache_ids.shape

        d_token_embedding = np.zeros_like(self.token_embedding)
        d_position_embedding = np.zeros_like(self.position_embedding)

        for b in range(B):
            for t in range(T):

                token_id = self.cache_ids[b,t]

                d_token_embedding[token_id] += dx[b,t]

                d_position_embedding[t] += dx[b,t]

        self. embedding_gradient = {
            "token_embedding": d_token_embedding + d_token_embedding_from_output,
            "position_embedding": d_position_embedding,
            "linear_prediction_b":d_linear_prediction_b
        }       

        return dx 
    
    def paramters(self):

        param =[]

        param.append(self.token_embedding)
        param.append(self.position_embedding)
        param.append(self.linear_prediction_b)

        for block in self.blocks:

            modules = [
                block.attention,
                block.layernorm1,
                block.ffn,
                block.layernorm2
            ]
        
            for module in modules:

                for name in module.paramters:

                 param.append(module.paramters[name])

        return param

    def gradients(self):

        grad = []

        grad.append(self.embedding_gradient["token_embedding"])
        grad.append(self.embedding_gradient["position_embedding"])
        grad.append(self.embedding_gradient["linear_prediction_b"])

        for block in self.blocks:
    
            modules = [
                block.attention,
                block.layernorm1,
                block.ffn,
                block.layernorm2
            ]         

            for module in modules:

                for name in module.gradients:

                    grad.append(module.gradients[name])

        return grad
                
class CrossEntropyloss:

    def forward(self , logit , target):

        softmax_linear_prediction = softmax(logit , axis = -1)

        EOS = len(vocab) - 1
        P = softmax_linear_prediction

        B , T , V = P.shape
        loss = np.mean(-np.log(P[np.arange(B)[:,None] , np.arange(T) , target] + 1e-9))

        self.cache = (B , T , V , P , target)

        return loss
    
    def backward(self):

        B , T , V , P , target = self.cache

        dlogit = P.copy()

        for b in range(B):
          dlogit[b, np.arange(T) , target[b]] -= 1
          
        dlogit /= (B * T)

        return dlogit
    
class AdamW:

    def  __init__(self, model):

        #==hyerparameters==
        self.c = 1
        self.b1 = 0.9
        self.b2 = 0.98
        self.lr_max = 3e-4
        self.lr_min = 3e-5
        self.lambda_ = 0.01
        self.total_step = 100
        self.warmup_step = 40
        self.decay_step = 60

        #==Paramters access==
        self.model = model
        param = self.model.paramters()

        #==Optimizer states==
        self.m = [np.zeros_like(p) for p in param]
        self.v = [np.zeros_like(p) for p in param]
        self.step_count = 0

    def step(self):
       
       param = self.model.paramters()
       grad = self.model.gradients()

       total_gi2 = 0.0
       for g in grad:
         gi2 = np.sum(g**2)
         total_gi2 += gi2
       g_norm = np.sqrt(total_gi2)
       scale = min(1 , self.c/(g_norm + 1e-6))
       clipped_gradient = [g * scale for g in grad]

       self.m = [self.b1 * m_i + (1 - self.b1) * g_t for m_i , g_t in zip(self.m , clipped_gradient)]
       self.v = [self.b2 * v_i + (1 - self.b2) * (g_t**2) for v_i , g_t in zip(self.v , clipped_gradient)]
    
       self.step_count += 1
       t = self.step_count
                       
       m_hat = [m_t / (1 - self.b1**t) for m_t in self.m]
       v_hat = [v_t / (1 - self.b2**t) for v_t in self.v]

       T_w = self.warmup_step
       #T_d:start   #decay_step:long
       T_d = self.total_step - self.decay_step
       #warmup 
 
       lr_w = self.lr_max * t/T_w

       #decay 
       process = min(1 , (t - T_d) / self.decay_step)
       lr_d = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * process))
       #Total
       if  T_w > 0 and t < T_w:
        lr = lr_w

       elif t < T_d:
        lr = self.lr_max

       else :
        lr = lr_d
    
       epsilon = 1e-8
       update = [m_h / (np.sqrt(v_h) + epsilon) for m_h , v_h in zip(m_hat , v_hat)]
       new_weight = [w - lr * (u + self.lambda_ * w) for w , u in zip(param , update)]

      #change the weight
       for p , new_p in zip(param , new_weight):
           p[:] = new_p

class Inference:
   def inference(prompt):
      
      
      
#training loop
text = 'hello world hello world hello world hello world hello world'
       

tokenizer = Tokenizer()
vocab = tokenizer.train(text)
ids = tokenizer.encode(text)

print(f'len(ids){len(ids)}')
print(f'ids{ids}')
print(len(vocab))

model = Model()
adamw = AdamW(model)
loss_history = []
fig , ax = plt.subplots()

for i in range(200):
 
 batch , target = tokenizer.batch(5 , 2)

 logit = model.forward(batch)

 crossentropyloss = CrossEntropyloss()
 loss = crossentropyloss.forward(logit , target)

 print(loss)

 #update the loss curve in real time
 loss_history.append(loss)

 ax.clear()
 ax.plot(loss_history)

 ax.set_title("Loss Curve ")
 ax.set_xlabel("step")
 ax.set_ylabel("loss")

 plt.pause(0.01)

 dlogit = crossentropyloss.backward()

 dx = model.backward(dlogit)

 optimizer = adamw.step()
