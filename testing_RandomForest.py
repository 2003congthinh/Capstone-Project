import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import pickle
import random

# Load the model
with open('random_forest_model.pkl', 'rb') as f:
    frst_model = pickle.load(f)

# Read the CSV file into a DataFrame
df = pd.read_csv('C:/Users/ADMIN/Desktop/code_folders/Capstone_Project/CIC-IDS/CSV-2023/MERGED_CSV/Merged59.csv')

# Grouping
DDoS = ['DDOS-ACK_FRAGMENTATION',     
        'DDOS-UDP_FLOOD',       
        'DDOS-SLOWLORIS',               
        'DDOS-ICMP_FLOOD',        
        'DDOS-RSTFINFLOOD',    
        'DDOS-PSHACK_FLOOD',        
        'DDOS-HTTP_FLOOD',              
        'DDOS-UDP_FRAGMENTATION',     
        'DDOS-TCP_FLOOD',  
        'DDOS-SYN_FLOOD',        
        'DDOS-SYNONYMOUSIP_FLOOD',    
        'DDOS-ICMP_FRAGMENTATION']

DoS = ['DOS-UDP_FLOOD',    
       'DOS-TCP_FLOOD',    
       'DOS-SYN_FLOOD',    
       'DOS-HTTP_FLOOD']

Spoofing = ['MITM-ARPSPOOFING',     
            'DNS_SPOOFING']

Brute_force = ['DICTIONARYBRUTEFORCE']

Recon = ['RECON-HOSTDISCOVERY', 
         'RECON-OSSCAN',     
         'RECON-PORTSCAN',               
         'RECON-PINGSWEEP',         
         'VULNERABILITYSCAN']

Web_based = ['SQLINJECTION',        
             'BROWSERHIJACKING',         
             'COMMANDINJECTION',       
             'XSS',         
             'BACKDOOR_MALWARE',          
             'UPLOADING_ATTACK']

Mirai = ['MIRAI-GREETH_FLOOD',     
         'MIRAI-UDPPLAIN',     
         'MIRAI-GREIP_FLOOD']



def classify_attacks(data):
    data['Label'].replace(DDoS,'DDoS',inplace=True)
    data['Label'].replace(DoS,'DoS',inplace=True)
    data['Label'].replace(Spoofing,'Spoofing',inplace=True)
    data['Label'].replace(Brute_force,'Brute_Force',inplace=True)
    data['Label'].replace(Recon,'Recon',inplace=True)
    data['Label'].replace(Web_based,'Web_based',inplace=True)
    data['Label'].replace(Mirai,'Mirai',inplace=True)
classify_attacks(df)

# Preprocessing
df.drop_duplicates(keep='first', inplace = True)
df.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace inf and -inf with NaN, then drop the resulting NaNs
df.dropna(inplace=True)    # dropna() removes any rows that contain NaN (missing) values; reset_index() resets the row index after dropping, so it's clean and continuous.
df.reset_index(drop=True, inplace=True)

data_clean = df.drop(columns=['fin_count', 'rst_count', 'Tot size', 'syn_count', 'ack_count', 'Std', 'Header_Length'])

# Create test data
X_test = data_clean.drop(columns=['Label'])
Y_test = data_clean['Label']

# for i in range(10):
#     sample = X_test[i].reshape(1, -1)  # Must be 2D
#     pred = frst_model.predict(sample)[0]
#     actual = Y_test[i]

#     pred_label = labelencoder.inverse_transform([int(pred)])[0]
#     actual_label = labelencoder.inverse_transform([int(actual)])[0]

#     print(f"Sample {i}: Predicted = {pred_label}, Actual = {actual_label}")

rand = random.randint(0,len(X_test))
test_data = X_test.iloc[[rand]]

pred_label = frst_model.predict(test_data)
print("Predicted class label:", pred_label)
real_label = Y_test.iloc[rand]
print("Real class label:", real_label)