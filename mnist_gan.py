import torch # ~ pytorch 모듈 임포트
import torch.nn as nn # ~ pytorch 모듈 임포트
import torch.utils as utils # ~ pytorch 모듈 임포트
import torch.nn.init as init # ~ pytorch 모듈 임포트
from torch.autograd import Variable # ~ pytorch 모듈 임포트
import torchvision.utils as v_utils # ~ torchvision 모듈 임포트
import torchvision.datasets as dset # ~ torchvision 모듈 임포트
import torchvision.transforms as transforms # ~ torchvision 모듈 임포트
import numpy as np # numpy 임포트
import matplotlib.pyplot as plt #pyplot 임포트
from collections import OrderedDict #ordereddict 임포트
import time #time 임포트

epoch = 500 #epoch = 500
batch_size = 512 #한 번에 돌아가는 batch = 512
learning_rate = 0.0002 #학습률 = 0.0002
num_gpus = 1 #gpu수 = 1 - 사용 안함
z_size = 50 #generator의 인풋 z size = 50
middle_size = 200 #중간 레이어 노드 수 200

# Download Data

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True) #mnist의 data 받아오기. tensor로 변형.

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True,drop_last=True) #mnist dataset 생성. batch_size, shuffle 설정. 배치수만큼 돌리다 남은것은 버림.


class Generator(nn.Module): #생성 모델
    def __init__(self): #생성자
        super(Generator,self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([ #첫번째 레이어 구성
                        ('fc1',nn.Linear(z_size,middle_size)), #첫 FC층, z_size에서 middle_size로
                        ('bn1',nn.BatchNorm1d(middle_size)), #batchnorm1d 시행 - 공변량시프트 막기.
                        ('act1',nn.ReLU()), #활성함수 ReLU통과
        ]))
        self.layer2 = nn.Sequential(OrderedDict([ #두번째 레이어 구성
                        ('fc1',nn.Linear(middle_size,middle_size)), #두번째 FC층, middle_size에서 Middle_size로
                        ('bn1',nn.BatchNorm1d(middle_size)),#batchnorm1d 시행 - 공변량시프트 막기.
                        ('act1',nn.ReLU()), #활성함수 ReLU통과
        ]))
        self.layer3 = nn.Sequential(OrderedDict([ #세번째 레이어 구성
                        ('fc2', nn.Linear(middle_size,784)),#세번째 FC층, middle_size에서 28*28=784개 노드로
                        #('bn2', nn.BatchNorm2d(784)),
                        ('tanh', nn.Tanh()), #활성함수 tanh 통과 
        ]))
    def forward(self,z): #전방함수
        out = self.layer1(z) #layer1통과한 out생성
        out = self.layer2(out)#out을 layer2 통과
        out = self.layer3(out)#out을 layer3 통과
        out = out.view(batch_size,1,28,28) #784개 출력을 28*28로 변형

        return out #out 리턴



class Discriminator(nn.Module): #분별 모델
    def __init__(self):#생성자
        super(Discriminator,self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([ #첫번째 레이어 구성
                        ('fc1',nn.Linear(784,middle_size)), #첫 FC층, 28*28=784에서 middle_size로
                        #('bn1',nn.BatchNorm1d(middle_size)),
                        ('act1',nn.LeakyReLU()),  #활성함수 leakyReLU 통과
            
        ]))
        self.layer2 = nn.Sequential(OrderedDict([ #두번째 레이어 구성
                        ('fc1',nn.Linear(middle_size,middle_size)), #두번째 FC층, middle_size에서 Middle_size로
                        #('bn1',nn.BatchNorm1d(middle_size)),
                        ('act1',nn.LeakyReLU()),  #활성함수 leakyReLU통과
            
        ]))
        self.layer3 = nn.Sequential(OrderedDict([ #세번째 레이어 구성
                        ('fc2', nn.Linear(middle_size,1)), #세번째 FC층, middle_size에서 1개 노드로
                        #('bn2', nn.BatchNorm2d(1)),
                        ('act2', nn.Sigmoid()), #활성함수 sigmoid통과
        ]))
                                    
    def forward(self,x): #전방함수
        out = x.view(batch_size, -1) #batch_size씩 쪼개지는 변형
        #print(out.shape)
        out = self.layer1(out) #out을 layer1 통과
        out = self.layer2(out) #out을 layer2 통과
        out = self.layer3(out) #out을 layer3 통과

        return out #out 리턴

generator = nn.DataParallel(Generator()) #generator 인스턴스 생성 (gpu사용 코드에서 변형)
discriminator = nn.DataParallel(Discriminator()) #discriminator 인스턴스 생성 (gpu사용 코드에서 변형)


loss_func = nn.MSELoss() #Loss 함수 : MSE
gen_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate,betas=(0.5,0.999)) #생성 모델 Adam으로 최적화, 학습률은 Learning_rate, 적응모멘텀, 적응학습률 beta=(0.5,0.999)
dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,betas=(0.5,0.999)) #분별 모델 Adam으로 최적화, 학습률은 Learning_rate, 적응모멘텀, 적응학습률 beta=(0.5,0.999)

ones_label = Variable(torch.ones(batch_size,1)) #batch_size *1크기 1로만 되어있는 label 생성
zeros_label = Variable(torch.zeros(batch_size,1)) #batch_size *1크기 0으로만 되어있는 label 생성

# train

for i in range(epoch): #epoch 만큼 루프
    start_time = time.time() #시작시간 기록
    for j,(image,label) in enumerate(train_loader): #batch_size만큼 받아오면서 루프
         
        image = Variable(image) #dataset에 있는 data를 pytorch variable로 변형
        
        # discriminator
        
        dis_optim.zero_grad() # 분별 모델 gradient 초기화
        
        z = Variable(init.normal(torch.Tensor(batch_size,z_size),mean=0,std=0.1)) #랜덤 z값 생성.
        gen_fake = generator.forward(z) #z값으로 생성모델 통과해 fake 생성
        dis_fake = discriminator.forward(gen_fake) #생성한 fake를 분별모델 통과해 label 생성
        
        dis_real = discriminator.forward(image) #dataset의 진짜 이미지를 분별모델 통과시켜 label 생성
        dis_loss = torch.sum(loss_func(dis_fake,zeros_label)) + torch.sum(loss_func(dis_real,ones_label)) #loss값 연산(진짜는 1과, 가짜는 0과 MSE연산 후 합.)
        dis_loss.backward(retain_graph=True) #gradient 연산
        dis_optim.step() #weight 갱신
        
        # generator
        
        gen_optim.zero_grad() #생성모델 gradient 초기화
        
        z = Variable(init.normal(torch.Tensor(batch_size,z_size),mean=0,std=0.1)) #랜덤 z값 생성
        gen_fake = generator.forward(z) #z값으로 생성모델 통과해 fake 생성
        dis_fake = discriminator.forward(gen_fake) #생성한 fake를 분별모델 통과해 lable 생성
        
        gen_loss = torch.sum(loss_func(dis_fake,ones_label)) #fake가 진짜로 판별되도록 loss함수 연산
        gen_loss.backward() #gradient 연산
        gen_optim.step() #weight 갱신
    
       
    
        # model save
        if j % 500 == 0: #500번 돌때마다 저장
            print(gen_loss,dis_loss) #loss 출력
            torch.save([generator,discriminator],'./model_mnist/vanilla_gan.pkl') #model 저장

            print("{}th iteration gen_loss: {} dis_loss: {}".format(i,gen_loss.data,dis_loss.data))
            v_utils.save_image(gen_fake.data[0:25],"./result_mnist/gen_{}_{}.png".format(i,j), nrow=5) #생성한 이미지 저장
    print("--- %s seconds ---" %(time.time() - start_time)) #epoch 돌때마다 시간 측정.