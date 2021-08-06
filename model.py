import paddle
from paddle import nn
import numpy as np
from paddle import optimizer
import os
from data_utils import img_save
from tqdm import tqdm
from paddle.vision.models import resnet18
from Models.Unet_Generator import Unet_Generator
# from paddle.distributed import fleet


class block_unit(nn.Layer):
    def __init__(self, input_channels=1, output_channels=3, residual=True, activation=nn.ReLU):
        super().__init__()
        self.residual = residual
        conv1 = nn.Conv2D(input_channels, output_channels, kernel_size=(3, 3), padding='SAME')
        conv2 = nn.Conv2D(output_channels, output_channels, kernel_size=(3, 3), padding='SAME')
        self.bn = nn.BatchNorm2D(output_channels)
        self.conv1x1 = nn.Conv2D(input_channels, output_channels, kernel_size=(1, 1), padding='SAME')
        self.act = activation()
        self.block_stem = nn.Sequential(
            conv1,
            #self.bn,
            self.act,
            #conv2,
            self.bn
        )

    def forward(self, x):
        if self.residual:
            x1 = self.block_stem(x)
            x2 = self.conv1x1(x)
            #x2 = self.bn(x2)
            x = x1+x2
        y = self.act(x)
        return y


# def get_block(sampling_mode='up',scale_factor=2):
#     if sampling_mode == 'up':
#         sampling = nn.Upsample(scale_factor=scale_factor)
#     elif sampling_mode == 'down':
#         sampling = nn.MaxPool2D(kernel_size=(scale_factor,scale_factor))
#     else:
#         print('{}\n\tERROR: The value of sampling_mode expected \'up\' or \'down\', but got {}.'.format(
#         __file__, sampling_mode))
#         exit()


class default_Generator(nn.Layer):
    def __init__(self, residual=True, block_nums=5):
        super(default_Generator, self).__init__()
        self.block = block_unit(3, 3, residual)
        self.sampling = nn.Upsample(scale_factor=2)
        modules = [block_unit(1, 3, residual), self.sampling]
        for i in range(block_nums - 2):
            modules.append(self.block)
            if i+1 == (block_nums-2)//2:
                modules.append(self.sampling)
        modules.append(self.sampling)
        modules.append(block_unit(3, 3, residual, activation=nn.Tanh))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    def input_generate(self, batch_size=16, constant=False, input_size=(128, 128)):
        # if batch_size is None:
        #     batch_size = self.batch_size
        height, width = input_size
        if constant:
            input_mat = np.ones((batch_size, 1, height, width), dtype=np.float32)
        else:
            input_mat = np.random.rand(batch_size, 1, height, width).astype(np.float32)
        input_tensor = paddle.to_tensor(input_mat)
        return input_tensor


class ResNet_Discriminator(nn.Layer):
    def __init__(self):
        super(ResNet_Discriminator, self).__init__()
        self.resnet = resnet18(pretrained=True, num_classes=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        y = self.act(x)
        return y


class default_Discriminator(nn.Layer):  # 如果采用Flatten，那么输入应当固定，如果采用GAP，则输入可以不定
    def __init__(self, residual=True, block_nums=6):
        super(default_Discriminator, self).__init__()
        self.sampling = nn.MaxPool2D(kernel_size=(2, 2))
        input_channels, output_channels = 3, 64
        modules = [block_unit(input_channels, output_channels, residual=True),
                   self.sampling]
        for i in range(block_nums - 2):
            input_channels = output_channels
            output_channels = min(512, input_channels * 2)
            modules.append(block_unit(input_channels, output_channels, residual))
            if i+1 == (block_nums-2)//2:
                modules.append(self.sampling)
        modules.append(self.sampling)
        input_channels = output_channels
        output_channels = min(2048, input_channels * 2)
        modules.append(block_unit(input_channels, output_channels, residual))
        self.backbone = nn.Sequential(*modules)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2D(output_size=1),
                                  nn.Conv2D(output_channels, output_channels, kernel_size=(1, 1)),
                                  # nn.Linear(output_channels, output_channels),
                                  nn.ReLU(),
                                  nn.Conv2D(output_channels, 1, kernel_size=(1, 1))
                                  # nn.Linear(output_channels, 1)
                                  )
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = paddle.squeeze(x, axis=(2, 3))
        y = self.output_act(x)
        return y


class General_GAN(nn.Layer):
    def __init__(self, Generator=Unet_Generator,
                 Discriminator=ResNet_Discriminator,
                 optimizer_generator=optimizer.Adam,
                 optimizer_discriminator=optimizer.Adam,
                 lr_generator=1e-3, lr_discriminator=1e-3, batch_size=16,
                 input_shape=None, output_shape=(512, 512), input_generate=None,
                 model_root='models'):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.opt_G = optimizer_generator(parameters=self.generator.parameters(),
                                         learning_rate=lr_generator)
        self.opt_D = optimizer_discriminator(parameters=self.discriminator.parameters(),
                                             learning_rate=lr_discriminator)
        self.batch_size = batch_size
        if input_shape is None:
            self.input_shape = tuple([i//4 for i in output_shape])
        self.output_shape = output_shape
        self.model_root = model_root
        self.total_step = 0
        if input_generate is None:
            self.input_generate = self.generator.input_generate
        else:
            self.input_generate = input_generate

        # fleet.init(is_collective=True)
        # self.generator = fleet.distributed_model(self.generator)
        # self.discriminator = fleet.distributed_model(self.discriminator)
        # self.opt_D = fleet.distributed_optimizer(self.opt_D)
        # self.opt_G = fleet.distributed_optimizer(self.opt_G)
        # self.generator = paddle.DataParallel(self.generator)
        # self.discriminator = paddle.DataParallel(self.discriminator)

    # def input_generate(self, batch_size=None, constant=False, input_size=(128, 128)):
    #     if batch_size is None:
    #         batch_size = self.batch_size
    #     height, width = input_size
    #     if constant:
    #         input_mat = np.ones((batch_size, 1, height, width), dtype=np.float32)
    #     else:
    #         input_mat = np.random.rand(batch_size, 1, height, width).astype(np.float32)
    #     input_tensor = paddle.to_tensor(input_mat)
    #     return input_tensor

    def fake_tensors2imgs(self, fake_tensors):
        fakes = fake_tensors.numpy()
        fakes = fakes.transpose((0, 2, 3, 1))
        fakes = [i for i in fakes]
        return fakes

    def img_generate(self, input_mat=None, input_constant=False):
        if input_mat is not None:
            input_tensor = paddle.to_tensor(input_mat)
        else:
            input_tensor = self.input_generate(constant=input_constant, input_size=self.input_shape)
        img_fake = self.generator(input_tensor)
        img_fake = self.fake_tensors2imgs(img_fake)
        return img_fake

    def save_model(self):
        if not os.path.exists(self.model_root):
            os.makedirs(self.model_root)
        generator_params = self.generator.state_dict()
        discriminator_params = self.discriminator.state_dict()
        opt_G_params = self.opt_G.state_dict()
        opt_D_params = self.opt_D.state_dict()
        save_str = 'epoch_{}-step_{}-MeanScore_{}'.format(self.epoch, self.step, self.MeanScore_Fake)
        paddle.save(generator_params, os.path.join(self.model_root, 'generator-{}.pdparams'.format(save_str)))
        paddle.save(discriminator_params, os.path.join(self.model_root, 'discriminator-{}.pdparams'.format(save_str)))
        paddle.save(opt_G_params, os.path.join(self.model_root, 'opt_G-{}.params'.format(save_str)))
        paddle.save(opt_D_params, os.path.join(self.model_root, 'opt_D-{}.params'.format(save_str)))

    def fit(self, epochs=2000,
            train_generator=None, val_generator=None,
            loss_function=nn.MSELoss,
            lr_generator=None, lr_discriminator=None,
            input_constant=False, new_batch4generator_training=True,
            log_file=None, steps_tqdm=10):
        if lr_generator is not None:
            self.opt_G.learning_rate = lr_generator
        if lr_discriminator is not None:
            self.opt_D.learning_rate = lr_discriminator
        loss_function = loss_function()

        self.generator.train()
        self.discriminator.train()
        for i in range(epochs):
            self.epoch = i
            self.step = 0
            with tqdm(total=steps_tqdm, ncols=100) as bar:
                bar.set_description('Epoch:{}, step:{}'.format(self.epoch, self.step))
                while True:
                    self.step += 1
                    self.total_step += 1

                    img_real = next(train_generator)
                    if img_real is None:
                        break
                    batch_size = img_real.shape[0]
                    if (self.total_step-1) % 30 == 0:  # 减1使得最开始的循环会先更新一次
                        bar.set_postfix_str('Updating Discriminator')
                        input_tensor = self.input_generate(batch_size=batch_size)  #, constant=input_constant,
                                                       # input_size=self.input_shape)
                        img_fake = self.generator(input_tensor)

                        score_real = self.discriminator(img_real)
                        score_fake = self.discriminator(img_fake)

                        # backward propogation for discriminator
                        label_real = np.ones((img_real.shape[0], 1))
                        label_real = paddle.to_tensor(label_real, dtype=np.float32)
                        label_fake = np.zeros((batch_size, 1))
                        label_fake = paddle.to_tensor(label_fake, dtype=np.float32)

                        '''想到两种方法进行反向传播
                        1. loss = loss_fake+loss_real
                           loss.backward()
                        2. loss_fake.backward()
                           loss_real.backward()
                        看起来第二种能够更加省显存
                        
                        其他涉及detach的地方果然也就主要是节省显存以此加速，那么同理更新生成器时重新生成一批数据应该也是为了节省显存，但增加时间。
                        https://blog.csdn.net/einstellung/article/details/102494795
                        
                        如果算了score即使不算loss，不做backward就会一直增加显存'''


                        loss_real = loss_function(score_real, label_real)
                        loss_fake = loss_function(score_fake, label_fake)
                        self.opt_D.clear_grad()  # 如若不先清空梯度，则后续的backward会累加梯度
                        loss_real.backward()
                        if not new_batch4generator_training:
                            loss_fake.backward(retain_graph=True)
                        else:
                            loss_fake.backward()
                        self.opt_D.step()
                        # print('Updated Discriminator')
                    ''' Error encountered without `retain_graph=True` :
                    RuntimeError: (NotFound) Inputs and outputs of sigmoid_grad do not exist. This may be because:
                    1. You use some output variables of the previous batch as the inputs of the current batch. Please try to call "stop_gradient = True" or "detach()" for these variables.
                    2. You calculate backward twice for the same subgraph without setting retain_graph=True. Please set retain_graph=True in the first backward call.
                    '''

                    # Training Generator
                    bar.set_postfix_str('Updating Generator')
                    if new_batch4generator_training:
                        input_tensor = self.input_generate(batch_size=batch_size)  # constant=input_constant, input_size=self.input_shape
                        img_fake = self.generator(input_tensor)
                        score_fake = self.discriminator(img_fake)

                    # backward propogation for generator
                    label_fake_reverse = np.ones((batch_size, 1))
                    label_fake_reverse = paddle.to_tensor(label_fake_reverse, dtype=np.float32)
                    loss_fake_generator = loss_function(score_fake, label_fake_reverse)
                    self.opt_G.clear_grad()
                    loss_fake_generator.backward()
                    self.opt_G.step()
                    bar

                    bar.update(1)

                    if self.total_step % 20 == 0:
                        self.MeanScore_Fake = np.mean(score_fake.numpy())
                        self.save_model()
                        log_str = 'Epoch:{}, step:{}, loss_fake_discriminator:{}, loss_real_discriminator:{}, loss_fake_generator:{}, Mean_score_fake:{}'.format(
                                self.epoch, self.step, loss_fake.numpy(), loss_real.numpy(), loss_fake_generator.numpy(), np.mean(score_fake.numpy()))
                        print(log_str)
                        img_fake = self.fake_tensors2imgs(img_fake)
                        img_save(img_fake, './imgs_generated/', 'epoch_{}-step_{}'.format(i, self.step), save_num=1)
                        if log_file:
                            f = open(log_file, 'a')
                            f.write(log_str+'\n')
                            f.close()
