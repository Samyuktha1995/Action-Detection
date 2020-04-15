{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "videoClassification_3DCNN.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "SUL6qVoHSXpt",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import os\n",
        "import numpy as np\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torchvision.models as models\n",
        "import torchvision.transforms as transforms\n",
        "import torch.utils.data as data\n",
        "import torchvision\n",
        "from torch.autograd import Variable\n",
        "import matplotlib.pyplot as plt\n",
        "from data_loader import *\n",
        "from model import *\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import OneHotEncoder, LabelEncoder\n",
        "from sklearn.metrics import accuracy_score\n",
        "import pickle"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "khcby-vbSfQa",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "L4dink_jWKXe",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "fc_hidden1 = 256\n",
        "fc_hidden2 = 256\n",
        "dropout = 0.0        \n",
        "k = 2            \n",
        "epochs = 15\n",
        "batch_size = 30\n",
        "learning_rate = 1e-4\n",
        "log_interval = 10\n",
        "img_x= 256\n",
        "img_y = 342 \n",
        "begin_frame, end_frame, skip_frame = 1, 24, 1"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aCK6o8M8SiGv",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def train(log_interval, model, device, train_loader, optimizer, epoch):\n",
        "    model.train()\n",
        "    losses = []\n",
        "    scores = []\n",
        "    N_count = 0  \n",
        "    for batch_idx, (X, y) in enumerate(train_loader):\n",
        "        X, y = X.to(device), y.to(device).view(-1, )\n",
        "        N_count += X.size(0)\n",
        "        optimizer.zero_grad()\n",
        "        output = model(X)  \n",
        "        loss = F.cross_entropy(output, y)\n",
        "        losses.append(loss.item())\n",
        "        y_pred = torch.max(output, 1)[1]  \n",
        "        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())\n",
        "        scores.append(step_score)        \n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        if (batch_idx + 1) % log_interval == 0:\n",
        "            print('Train Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}, Accu: {:.2f}%'.format(\n",
        "                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))\n",
        "    return losses, scores"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "j6mK60x_S3gk",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def validation(model, device, optimizer, test_loader):\n",
        "    model.eval()\n",
        "    test_loss = 0\n",
        "    all_y = []\n",
        "    all_y_pred = []\n",
        "    with torch.no_grad():\n",
        "        for X, y in test_loader:\n",
        "            X, y = X.to(device), y.to(device).view(-1, )\n",
        "            output = model(X)\n",
        "            loss = F.cross_entropy(output, y, reduction='sum')\n",
        "            test_loss += loss.item()                 \n",
        "            y_pred = output.max(1, keepdim=True)[1]  \n",
        "            all_y.extend(y)\n",
        "            all_y_pred.extend(y_pred)\n",
        "    test_loss /= len(test_loader.dataset)\n",
        "    all_y = torch.stack(all_y, dim=0)\n",
        "    all_y_pred = torch.stack(all_y_pred, dim=0)\n",
        "    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())\n",
        "    print('\\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\\n'.format(len(all_y), test_loss, 100* test_score))\n",
        "    torch.save(model.state_dict(), os.path.join(save_model_path, '3dcnn_epoch{}.pth'.format(epoch + 1)))  \n",
        "    torch.save(optimizer.state_dict(), os.path.join(save_model_path, '3dcnn_optimizer_epoch{}.pth'.format(epoch + 1)))\n",
        "    print(\"Epoch {} model saved!\".format(epoch + 1))\n",
        "    return test_loss, test_score"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "99wfe1chTH-x",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def get_data(data_path):\n",
        "    action_names = []\n",
        "    action_names.append('Not Walking')\n",
        "    action_names.append('Walking')\n",
        "  \n",
        "    use_cuda = torch.cuda.is_available()                   \n",
        "    device = torch.device(\"cuda\" if use_cuda else \"cpu\")   \n",
        "    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}\n",
        "\n",
        "    le = LabelEncoder()\n",
        "    le.fit(action_names)\n",
        "    action_category = le.transform(action_names).reshape(-1, 1)\n",
        "    enc = OneHotEncoder()\n",
        "    enc.fit(action_category)\n",
        "    \n",
        "    actions = []\n",
        "    fnames = os.listdir(data_path)\n",
        "    all_names = []\n",
        "    j = 0\n",
        "    for f in fnames:\n",
        "        all_names.append(f)\n",
        "        if 'walk' in f:\n",
        "          j = j+1\n",
        "          actions.append(\"Walking\")\n",
        "        else:\n",
        "          actions.append(\"Not Walking\")           \n",
        "    labels = labels2cat(le, actions)    \n",
        "    return all_names, labels"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "j2_5WuGXUAGv",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def train_test_data(data_path, train_x, train_y):\n",
        "    train_list, test_list, train_label, test_label = train_test_split(train_x, train_y, test_size=0.25, random_state=42)\n",
        "    transform = transforms.Compose([transforms.Resize([img_x, img_y]),\n",
        "                                    transforms.ToTensor(),\n",
        "                                    transforms.Normalize(mean=[0.5], std=[0.5])])\n",
        "    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()\n",
        "    train_set = Dataset_CNN(data_path, train_list, train_label, selected_frames, transform=transform)\n",
        "    test_set = Dataset_CNN(data_path, test_list, test_label, selected_frames, transform=transform)\n",
        "    train_loader = data.DataLoader(train_set, **params)\n",
        "    test_loader = data.DataLoader(test_set, **params)\n",
        "    return train_loader, test_loader"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MgCasyx6VSVn",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def main():\n",
        "    data_path = '/content/drive/My Drive/data/Data-image'\n",
        "    save_model_path = \"/content/drive/My Drive/data/\"\n",
        "\n",
        "    train_x, train_y = get_data(data_path)\n",
        "    train_loader, test_loader = train_test_data(data_path, train_x, train_y)\n",
        "    cnn3d = CNN3D(t_dim=len(selected_frames), img_x=img_x, img_y=img_y,\n",
        "                 drop_p=dropout, fc_hidden1=fc_hidden1,  fc_hidden2=fc_hidden2, num_classes=k).to(device)\n",
        "    optimizer = torch.optim.Adam(cnn3d.parameters(), lr=learning_rate)   # optimize all cnn parameters\n",
        "    epoch_train_losses = []\n",
        "    epoch_train_scores = []\n",
        "    epoch_test_losses = []\n",
        "    epoch_test_scores = []\n",
        "\n",
        "    # start training\n",
        "    epoch_train_losses, epoch_train_scores, epoch_test_losses, epoch_test_scores = [], [], [], []\n",
        "    for epoch in range(epochs):\n",
        "        print(epoch)\n",
        "        train_losses, train_scores = train(log_interval, cnn3d, device, train_loader, optimizer, epoch)\n",
        "        epoch_test_loss, epoch_test_score = validation(cnn3d, device, optimizer, valid_loader)\n",
        "        epoch_train_losses.append(train_losses)\n",
        "        epoch_train_scores.append(train_scores)\n",
        "        epoch_test_losses.append(epoch_test_loss)\n",
        "        epoch_test_scores.append(epoch_test_score)\n",
        "        # save results\n",
        "        A = np.array(epoch_train_losses)\n",
        "        B = np.array(epoch_train_scores)\n",
        "        C = np.array(epoch_test_losses)\n",
        "        D = np.array(epoch_test_scores)\n",
        "        np.save('./3DCNN_epoch_training_losses.npy', A)\n",
        "        np.save('./3DCNN_epoch_training_scores.npy', B)\n",
        "        np.save('./3DCNN_epoch_test_loss.npy', C)\n",
        "        np.save('./3DCNN_epoch_test_score.npy', D)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8CkSSfsQlzoh",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "if __name__ == \"__main__\":\n",
        "  main()"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}