# coding:UTF-8
'''
对原始的 eeg 信号，使用 CNN_3D 进行情感分类。
Created by Xiao Guowen.
'''
from utils.tools import build_preprocessed_eeg_dataset_CNN_3D, RawEEGDataset, subject_independent_data_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



# 加载数据，整理成所需要的格式
data_folder_path = '../../local_data_3D/'
feature_vector_dict, label_dict = build_preprocessed_eeg_dataset_CNN_3D(data_folder_path)


def leave_one_cross_validation():
    '''
            使用留一法进行测试，统计 subject-independent 方式的准确率
        :return None:
    '''
    accuracy = []
    for subject in range(1, 15):
        train_feature, train_label, test_feature, test_label = subject_independent_data_split(feature_vector_dict,
                                                                                              label_dict,
                                                                                              {str(subject)})

        desire_shape = [1, 200, 8, 9]
        norm_dim = 1
        train_data = RawEEGDataset(train_feature, train_label, desire_shape, norm_dim)
        test_data = RawEEGDataset(test_feature, test_label, desire_shape, norm_dim)

        # 超参数设置
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_epochs = 10
        num_classes = 3
        batch_size = 24
        learning_rate = 0.0001

        # Data loader
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
        test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

        model = ConvNet_3D(num_classes).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)

        print('当前测试对象为 subject {}...'.format(subject))
        subject_accuracy = train(model, train_data_loader, test_data_loader, optimizer, scheduler, criterion,
                                 num_epochs, device)
        print('当前测试对象准确率为 {}'.format(subject_accuracy))
        accuracy.append(subject_accuracy)
    print('使用留一法，1-15号 subject 的准确率分别为 {}'.format(accuracy))


# 定义卷积网络结构
class ConvNet_3D(nn.Module):
    def __init__(self, num_class):
        super(ConvNet_3D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(9, 3, 3), stride=(2, 1, 1), padding=(4, 1, 1), bias=True),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=True),
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=True),
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(128 * 25 * 8 * 9, 256, bias=True)
        self.fc2 = nn.Linear(256, num_class, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# Train the model
def train(model, train_data_loader, test_data_loader, optimizer, scheduler, criterion, num_epochs, device):
    print('Start Training...')
    writer = SummaryWriter('../../log')
    total_step = len(train_data_loader)
    batch_cnt = 0
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_data_loader):
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            batch_cnt += 1
            writer.add_scalar('train_loss', loss, batch_cnt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                                                         total_step, loss.item()))
        scheduler.step()
        accuracy = test(model, test_data_loader, device)
    torch.save(model.state_dict(), '../../model/model.ckpt')
    return accuracy


# Test the model
def test(model, data_loader, device, is_load=False):
    if is_load:
        model.load_state_dict(torch.load('../../model/model.ckpt'))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy is {}%'.format(100 * correct / total))
        accuracy = 100 * correct / total
    return accuracy


leave_one_cross_validation()
