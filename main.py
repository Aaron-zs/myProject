from myDataset import *
from torch.utils.data import DataLoader
from LSTM import *
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import random_split
myDataset = MyDataset()
train_dataset, test_dataset = random_split(
    dataset=myDataset,
    lengths=[2700, 136],
    generator=torch.Generator().manual_seed(0)
)
test_data_size = len(test_dataset)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, drop_last=False, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, drop_last=False, shuffle=True)


# 定义训练的设备
device = torch.device("cuda")
tudui = BiLSTM(2, 4)
tudui = tudui.to(device)
loss_fn = nn.BCEWithLogitsLoss()
loss_fn = loss_fn.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(tudui.parameters(), lr=learning_rate)
# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10000
writer = SummaryWriter("logs_train")


for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        targets = targets.unsqueeze(1)
        imgs = imgs.to(torch.float32)
        targets = targets.to(torch.float32)
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 1 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
        #print("targets：{}, outputs: {}".format(targets.item(), outputs.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            targets = targets.unsqueeze(1)
            imgs = imgs.to(torch.float32)
            targets = targets.to(torch.float32)
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            #输出大于0.5替换成1，否则是0
            zero = torch.zeros_like(outputs)
            one = torch.ones_like(outputs)
            outputs = torch.where(outputs > 0.5, one, outputs)
            outputs = torch.where(outputs < 0.5, zero, outputs)
            accuracy = (outputs == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1
writer.close()

