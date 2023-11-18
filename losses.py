import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 300
BATCH_SIZE = 128

#학습 데이터셋 준비 및 미니배치 처리를 위한 데이터로더 생성
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./.data', #CIFAR-10 데이터셋 로드
        train = True,
        download = True,
        transform = transforms.Compose([
            #학습 효과를 늘려주기 위해 학습할 이미지량 늘려주기
            transforms.RandomCrop(32, padding = 4), #이미지 무작위로 자르기
            transforms.RandomHorizontalFlip(), #이미지 뒤집기
            transforms.ToTensor(), #numpy array를 torch(Tensor) 이미지로 축 변경
            #numpy image : H x W x C
            # torch image : C X H X W
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
    batch_size = BATCH_SIZE, shuffle = True)

#실험 데이터셋 준비 및 미니배치 처리를 위한 데이터로더 생성
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./.data', #CIFAR-10 데이터셋 로드
        train = False,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
    batch_size = BATCH_SIZE, shuffle = True)

#Basic Block class 정의
class BasicBlock(nn.Module):
#학습 모듈 정하기
    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        # parameters : input channel size, output volume size, kernel size(filter), padding
        # 배치정규화(batch normalization) 을 수행 : 학습 중 각 계층에 들어가는 입력을 평균과 분산으로 정규화 함으로써 학습을 효율적으로 만들어 냄. (드랍아웃은 간접적으로 과적합을 막는다면 배치 정규화는 신경망 내부 데이터에 직접 영향을 주는 방식)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding =1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        #short cut 설정, nn.Sequential()은 다른 module들을 포함하는 module로, 그 module들을 순차적으로 적용하여 출력을 생성, 각 레이어를 데이터가 순차적으로 지나갈 때 사용하면 코드를 간결하게 만듦
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes: #stride 가 1이 아니거나, in_planes와 planes가 같지 않다면
            self.shortcut = nn.Sequential(
            #계층과 활성화 함수를 정의하여 순서대로 값을 전달해 처리
                nn.Conv2d(in_planes, planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 모델 정의
class ResNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ResNet, self).__init__()
        #self._make_layers()함수에서 각 층을 만들 때 전 층의 채널 출력값을 기록, 16으로 초기화
        self.in_planes = 16
        #3x3의 커널 크기를 가지며, 3색의 채널을 16개로 만들어 냄
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self. bn1 = nn.BatchNorm2d(16)
        # 16x32x32크기의 텐서를 만들어 냄
        self.layer1 = self._make_layer(16, 2, stride =1)
        #32x16x16의 텐서를 만들어 냄(stride = 2 이기 때문에, 증폭하였고 stride != 1이어서 shorcut 모듈을 가짐)
        self.layer2 = self._make_layer(32, 2, stride = 2)
        #64x8x8의 텐서를 만들어 냄(stride = 2 이기 때문에)
        self.layer3 = self._make_layer(64, 2, stride = 2)
        self.linear = nn.Linear(64, num_classes)

#BasicBlock 클래스를 하나의 모듈로 객체화 되어 ResNet모델의 주요 층을 이룸.
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks -1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
            #여러 Basic Block을 모듈 하나로 묶어주는 역할
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #평균 풀링을 통해 텐서에 있는 원소개수를 64개로 만듬(64x8x8 → 64x1x1)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        #64개의 입력을 받아 레이블 마다 예측값을 냄
        out = self. linear(out)
        return out

model = ResNet().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 0.0005)
#50번 호출 될 때마다 학습률에 0.1(gamma 값)을 곱해주어서 학습률이 계속 낮아지는 학습률 감소 기법 사용
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)

#학습
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
#측정
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            #배치 오차를 합산
            test_loss += F.cross_entropy(output, target, reduction = "sum").item()
            #가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    #테스트 로스값
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for epoch in range(1, EPOCHS +1):
    train(model, train_loader, optimizer, epoch)
    scheduler.step()
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("[{}] Test Loss: {:.4f}, Accuracy : {:.2f}%".format(epoch, test_loss, test_accuracy))
