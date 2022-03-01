import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from sr_dataloader import TrainDataset, TestDataset
import cv2
import numpy as np

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

model = SRCNN()
batch_size = 16
learning_rate = 1e-4 # 0.0001
training_epochs = 15
loss_function = nn.MSELoss()

optimizer = optim.Adam([
    {'params': model.conv1.parameters()},
    {'params': model.conv2.parameters()},
    {'params': model.conv3.parameters(), 'lr': learning_rate * 0.1}
], lr=learning_rate)

train_dataset = TrainDataset()
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

test_dataset = TestDataset()
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(train_dataloader)

    for data in train_dataloader:
        inputImg, labelImg = data

        predImg = model(inputImg)

        loss = loss_function(predImg, labelImg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_cost += loss / total_batch
    print('Epoch: %d Loss = %f'%(epoch+1, avg_cost))
torch.save(model.state_dict(),"srcnn.pth")
print('Learning finished')

def calc_psnr(orig, pred):
    return 10. * torch.log10(1. / torch.mean((orig - pred) ** 2))

bicubic_PSNRs = []
srcnn_PSNRs = []

for data in test_dataloader:
    inputImg, labelImg, imgName = data

    with torch.no_grad():
        predImg = model(inputImg).clamp(0.0, 1.0)
    bicubic_PSNRs.append(calc_psnr(labelImg, inputImg))
    srcnn_PSNRs.append(calc_psnr(labelImg, predImg))

    predImg = np.array(predImg * 255, dtype=np.uint8)
    predImg = np.transpose(predImg[0, :, :, :], [1, 2, 0])

    cv2.imwrite(".\\SR_dataset\\Set5_Pred\\" + imgName[0], predImg)


print('Average PSNR (bicubic)\t: %.4fdB' % (sum(bicubic_PSNRs) / len(bicubic_PSNRs)))
print('Average PSNR (SRCNN)\t: %.4fdB' % (sum(srcnn_PSNRs) / len(srcnn_PSNRs)))


