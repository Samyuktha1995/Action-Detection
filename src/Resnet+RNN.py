import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utility.models import *
from utility.data_loader import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

from google.colab import drive
drive.mount('/content/gdrive/')

data_path = "/content/gdrive/My Drive/HDMI_Data/Data_image/"    
save_model_path = "/content/gdrive/My Drive/HDMI_Data/Model/"
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   
res_size = 224        
dropout_p = 0.3       
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256
k = 2           
epochs = 40        
batch_size = 40  
learning_rate = 1e-3
log_interval = 10  
begin_frame, end_frame, skip_frame = 1, 20, 1

def train(log_interval, model, device, train_loader, optimizer, epoch):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()
    losses = []
    scores = []
    N_count = 0   
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)
        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))   
        loss = F.cross_entropy(output, y)
        losses.append(loss.item())
        y_pred = torch.max(output, 1)[1]  
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores

def validation(model, device, optimizer, test_loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).view(-1, )
            output = rnn_decoder(cnn_encoder(X))
            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 
            y_pred = output.max(1, keepdim=True)[1]  
            all_y.extend(y)
            all_y_pred.extend(y_pred)
    test_loss /= len(test_loader.dataset)
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      
    print("Epoch {} model saved!".format(epoch + 1))
    return test_loss, test_score

def plot_curves(A, B, C, D):
  fig = plt.figure(figsize=(10, 4))
  plt.subplot(121)
  plt.plot(np.arange(1, epochs + 1), A[:, -1])  
  plt.plot(np.arange(1, epochs + 1), C)         
  plt.title("model loss")
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend(['train', 'test'], loc="upper left")
  plt.subplot(122)
  plt.plot(np.arange(1, epochs + 1), B[:, -1])  
  plt.plot(np.arange(1, epochs + 1), D)         
  plt.title("training scores")
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.legend(['train', 'test'], loc="upper left")
  title = "./fig_Resnet+RNN.png"
  plt.savefig(title, dpi=600)
  plt.show()

def main():  
  action_names = []
  action_names.append('Not Walking')
  action_names.append('Walking')

  use_cuda = torch.cuda.is_available()                   
  device = torch.device("cuda" if use_cuda else "cpu")   
  params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
  le = LabelEncoder()
  le.fit(action_names)
  list(le.classes_)
  action_category = le.transform(action_names).reshape(-1, 1)
  enc = OneHotEncoder()
  enc.fit(action_category)
  actions = []
  fnames = os.listdir(data_path)
  all_names = []
  for f in fnames:
      all_names.append(f)
      if 'walk' in f:
          actions.append("Walking")
      else:
          actions.append("Not Walking")           
  labels = labels2cat(le, actions) 
  all_X_list = all_names                  
  all_y_list = labels2cat(le, actions)    
  train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)

  transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

  train_set, valid_set = Dataset_CNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                        Dataset_CNN(data_path, test_list, test_label, selected_frames, transform=transform)

  train_loader = data.DataLoader(train_set, **params)
  valid_loader = data.DataLoader(valid_set, **params)
  # Load model
  cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
  rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                          h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

  # Check for GPU
  if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs!")
      cnn_encoder = nn.DataParallel(cnn_encoder)
      rnn_decoder = nn.DataParallel(rnn_decoder)
      crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                    list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
                    list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

  elif torch.cuda.device_count() == 1:
      print("Using", torch.cuda.device_count(), "GPU!")
      crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                    list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                    list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

  optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)

  # record training process
  epoch_train_losses = []
  epoch_train_scores = []
  epoch_test_losses = []
  epoch_test_scores = []

  # start training
  for epoch in range(epochs):
      # train, test model
      print(epoch)
      train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
      epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

      # save results
      epoch_train_losses.append(train_losses)
      epoch_train_scores.append(train_scores)
      epoch_test_losses.append(epoch_test_loss)
      epoch_test_scores.append(epoch_test_score)

      # save all train test results
      A = np.array(epoch_train_losses)
      B = np.array(epoch_train_scores)
      C = np.array(epoch_test_losses)
      D = np.array(epoch_test_scores)
      np.save('./CRNN_epoch_training_losses.npy', A)
      np.save('./CRNN_epoch_training_scores.npy', B)
      np.save('./CRNN_epoch_test_loss.npy', C)
      np.save('./CRNN_epoch_test_score.npy', D)

  plot_curves(A, B, C, D)

if __name__ == "__main__":
  main()
