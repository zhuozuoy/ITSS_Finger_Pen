import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from img_model import SimpleCNN

# torch.manual_seed(1) # reproducible

NUM_CLASS = 62
TRAIN_EPOCH = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-2

# Load EMNIST Dataset

# data_transform = transforms.Compose([transforms.Grayscale(1),
#                                      transforms.RandomHorizontalFlip(p=0.5),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean=[0.5], std=[0.5])
#                                      ])
train_dataset = torchvision.datasets.EMNIST(
    root='./data',
    split='byclass',
    train=True,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

test_dataset = torchvision.datasets.EMNIST(
    root='./data',
    split='byclass',
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

# View data
# def get_mapping(num, with_type='byclass'):
#   if with_type == 'byclass':
#     if num <= 9:
#       return chr(num + 48) # digits
#     elif num <= 35:
#       return chr(num + 55) # uppercase letter
#     else:
#       return chr(num + 61) # lowercase letter

# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#   sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
#   img, label = train_dataset[sample_idx]
#   print(label)
#   figure.add_subplot(rows, cols, i)
#   plt.title(get_mapping(label))
#   plt.axis("off")
#   plt.imshow(img.squeeze(), cmap="gray")
#   plt.show()

print("Total number of train samples:", len(train_dataset))
print("Total number of test samples:", len(test_dataset))
print(train_dataset.train_data.shape)
print(test_dataset.test_data.shape)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=True)
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=2,
                         pin_memory=True)

# Handwritten Image-based Classification
cnn = SimpleCNN()
# print(cnn)

if torch.cuda.is_available():
     cnn = cnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

# Training
for epoch in range(TRAIN_EPOCH):
    print('Epoch {}'.format(epoch + 1) + '/{}'.format(TRAIN_EPOCH))
    print('*' * 12)
    running_loss = 0.0
    running_acc = 0.0

    for i, data in enumerate(train_loader, 1):
        img, label = data
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        out = cnn(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / len(train_dataset)
    ))

# Testing
    cnn.eval()
    eval_loss = 0
    eval_acc = 0
    for i, data in enumerate(test_loader, 1):
        img, label = data
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        out = cnn(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        eval_acc += num_correct.item()

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / len(test_dataset)))

    # Save the Trained Model
    torch.save(cnn.state_dict(), f'./ckpt/CNN_model_{epoch}.pkl')

