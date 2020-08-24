import os
import sys
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from utils import load_testing_data
from preprocess import Preprocess
from data import TwitterDataset

def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1 # 大於等於 0.5 為負面
            outputs[outputs<0.5] = 0 # 小於 0.5 為正面
            ret_output += outputs.int().tolist()
    
    return ret_output

# Variabls
testing_prefix = sys.argv[1]
output_prefix = sys.argv[2]
batch_size = 128
sen_len = 30
w2v_path = './w2v_all.model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("batch_size:{}, sen_len:{}".format(batch_size, sen_len))
print("loading testing data ...")
test_x = load_testing_data(testing_prefix)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
print('\nload model ...')
model = torch.load('./ckpt.model')
outputs = testing(batch_size, test_loader, model, device)

# 寫到 csv 檔案供上傳 Kaggle
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
print("save csv ...")
tmp.to_csv(output_prefix, index=False)
print("Finish Predicting")
