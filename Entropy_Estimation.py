#!/usr/bin/env python
# coding: utf-8

# # Entropy estimation

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import tqdm
from scipy.optimize import fsolve
import os



def local_predictor(p, *args):
    
    l, n, precision = args
    
    
    q = 1-p
    x = 1+q*(p**l)
    
    
    for i in range(int(precision)):
        x = x + (((l+1)**(i+1))*((q*(p**l))**(i+2)))     #check equation
    
    return ((1-p*x)/((l+2-(l+1)*x)*q*(x**(n+1))))-0.99

# In[2]:


def data_collection(pos, size):
    
    '''
    The function returns binary data of size 'size' 
    starting from 'pos' in 'random_file'
    '''
    
    
    with open(f'{random_file}','rb') as fr:    
        fr.seek(pos,0)
        data = fr.read(size)
    
    return data








def data_preprocess(data):
    
    '''
    The function does preprocessing of data, 
    which is transferred to neural network.
    '''
    
    
    byte_string = "{0:08b}".format(int.from_bytes(data,'big'))       #converts binary data into binary digits
    
    data_size = len(byte_string)                            #size of binary data
    
    X_data = [float(char) for char in byte_string[:-(data_size%(input_dim+1))]]   #trim data from last to make tensor, and convert binary string to list
    Y = X_data[input_dim :: input_dim + 1]               #take the Y values from the list and clean the original list
    del X_data[input_dim :: input_dim + 1]
    

    X_data = torch.tensor(X_data)             
    X_data = X_data.view((data_size//(input_dim+1)), input_dim)          #create tensor of appropriate dimensions
    
    Y = torch.tensor(Y)
    Y = Y.view((data_size//(input_dim+1)), 1)
    
    return X_data, Y






class Neural_Network(nn.Module):
       
    '''
    The class contains architecture of neural network
    '''
    
    def __init__(self, input_dim, num_classes):
        
        super(Neural_Network, self).__init__()
        #self.fc1 = nn.Linear(stats_count, 20)
        #self.fc2 = nn.Linear(20, 1)
        self.fc3 = nn.Linear(input_dim, 128)
        self.fc4 = nn.Linear(128, 20)
        self.fc5 = nn.Linear(20, num_classes)
    
    
    def forward(self, input_data):
        #x = F.relu(self.fc1(stats))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(torch.cat((input_data, x), dim=1)))
        x=F.relu(self.fc3(input_data))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))        #softmax
        return x

model = Neural_Network(input_dim, 1)
print(model)











#lag = 16              #parameter for auto-correlation
#stats_count = lag + 4       #stats parameters
input_dim = 20         #input random numbers

training_set_ratio = 0.7       #ratio of training set in total data


batch_size_appr = 1000         #approximate batch size(in bytes)
random_file = 'random_files/dev-random.bin'      #file containing random numbers


batch_size = math.ceil((batch_size_appr - ((batch_size_appr*8)%(input_dim+1)))/8)


# ## Data-Preprocessing

# In[3]:


with open(f'{random_file}','rb') as fr:
    fr.seek(0,2)
    file_size = fr.tell()
    training_set_size = math.floor(training_set_ratio*file_size)


# In[4]:




# In[5]:




# In[6]:


train_batches = math.floor(training_set_size/batch_size)            #total train batches
test_batches = math.ceil(file_size/batch_size)-train_batches         #total test batches(from memory constraints)


# ## Neural Network

# In[7]:




# In[8]:


loss_function = nn.L1Loss()      # absolute mean loss function
total_epochs = 2


learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)    #Adam optimizer


#reducing learning rate when plateau occurs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001)


# ### Model training

# In[18]:



PATH = 'saved_models/'

os.makedirs(PATH, exist_ok=True)

'''
model.load_state_dict(torch.load(PATH+save_dirs[-1]))
model.eval()
'''


for epoch in tqdm(range(total_epochs)):
    
    
    #tqdm.write(f'Epoch: {epoch}')
    
    for batch in tqdm(range(train_batches)):
    
        batch_data = data_collection(batch*batch_size, batch_size)
        X_data, Y = data_preprocess(batch_data)
        
        
        model.zero_grad()
        output = model(X_data)
        
        
        loss = loss_function(output, Y)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        scheduler.step(loss)
        

        with open(PATH+'Loss.txt','a') as l:
            l.write(f'Batch:{batch}\n')
            l.write(f'{loss.item()}')
            l.write('\n')
            
            
        if(batch%50 == 0):
            os.makedirs(PATH+f'Epoch_{epoch}/', exist_ok=True)
            torch.save(model.state_dict(), PATH+f'Epoch_{epoch}/{batch}.txt')
        
        #tqdm.write(f'Batch: {batch} \t Loss: {loss}')


# ### Model Testing

# In[ ]:


correct = 0
total = 0
longest_run = 0


with torch.no_grad():
    
    for batch in tqdm(range(test_batches)):
        
        batch_data = data_collection((train_batches + batch)*batch_size, batch_size)
        X_data, Y = data_preprocess(batch_data)
        
        
        output = model(X_data)
        run = 0
    
        
        for idx,i  in enumerate(output):
            
            if math.floor(float(output[idx])+0.5) == Y[idx]:
                run = run + 1
                correct = correct+1
            else:
                longest_run = max(longest_run, run)
                run = 0
            total = total+1
    
        loss = loss_function(output, Y)
        
    print('Loss')
    print(loss)
    print('Correct: '+str(correct))
    print('Total:'+str(total))
    print('correct_ratio: '+str(correct/total))
    print(longest_run)


# ## Entropy calculation

# In[ ]:


n = total       #number of bits produced
c = correct       #number of correct bits
l = longest_run        #longest run


# In[ ]:


prediction_global = c/n

if prediction_global == 0 :
    prediction_global_normalized = 1 - (0.01**(1/n))

else:
    prediction_global_normalized = min(1, prediction_global+2.579*(((prediction_global*(1-prediction_global))/(n-1))**(1/2))) #99% confidence


# In[ ]:



# In[ ]:


#Calculating local predictor upto a level of precision


precision = 0
efselon = 1e-5


predict =  fsolve(local_predictor, 0.5, (l, n, precision))
precision = precision+1

predict_new = fsolve(local_predictor, 0.5, (l, n, precision))


while abs(predict-predict_new)>efselon:
    precision = precision+1
    predict = predict_new
    predict_new = fsolve(local_predictor, 0.5, (l, n, precision))


prediction_local = predict_new


# In[ ]:


min_Entropy = -math.log(max(prediction_global_normalized, prediction_local),2)

print(min_entropy)
# In[ ]:




# In[ ]:




