import time
import sys
import os
from os import listdir
from os.path import join

import torch
import torchvision
import shutil

from Dataset import  load_image, is_image_file
from DataParallel import  DataParallelCriterion , DataParallelModel, DataParallelMetric

LEARNING_RATE = 1e-3
LR_THRESHOLD = 1e-7
DEGRADATION_TOLERANCY = 6
SCHEDULER_STEP = 20
SCHEDULER_FACTOR = 0.2

class NeuralPhotoFilter(object):
    def __init__(self, generator,  criterion, accuracy, dimension, image_size):

        self.cudas = list(range(torch.cuda.device_count()))
        self.generator = DataParallelModel(generator, device_ids=self.cudas, output_device=self.cudas)
        self.criterion = DataParallelCriterion(criterion, device_ids=self.cudas, output_device=self.cudas)
        self.accuracy  = DataParallelMetric(accuracy, device_ids=self.cudas, output_device=self.cudas)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.criterion.to(self.device)
        self.accuracy.to(self.device)

        self.dimension = dimension
        self.image_size= image_size

        self.optimizerG = torch.optim.Adam(self.generator.module.parameters(), lr = LEARNING_RATE)
        self.optimizerD = torch.optim.Adam(self.criterion.module.discriminator.parameters(), lr = LEARNING_RATE)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, mode='max', factor=SCHEDULER_FACTOR, patience=DEGRADATION_TOLERANCY, verbose=True, min_lr=LR_THRESHOLD)
        self.schedulerD = torch.optim.lr_scheduler.StepLR(self.optimizerD, step_size=SCHEDULER_STEP, gamma=SCHEDULER_FACTOR)

        self.iteration = int(0)
        self.tensoration = torchvision.transforms.ToTensor()
        self.best_lossD = 1e6

        config = str(generator.__class__.__name__) + '_' + str(generator.deconv1.__class__.__name__) +  '_' + str(generator.activation.__class__.__name__)
        config += '_' + str(criterion.__class__.__name__)
        directory = './RESULTS/'
        reportPath = os.path.join(directory, config + "/report/")

        flag = os.path.exists(reportPath)
        if flag != True:
            os.makedirs(reportPath)
            print('os.makedirs("reportPath")')

        self.modelPath = os.path.join(directory, config + "/model/")

        flag = os.path.exists(self.modelPath)
        if flag != True:
            os.makedirs(self.modelPath)
            print('os.makedirs("/modelPath/")')

        self.images = os.path.join(directory, config + "/images/")
        flag = os.path.exists(self.images)
        if flag != True:
            os.makedirs(self.images)
            print('os.makedirs("/images/")')
        else:
            shutil.rmtree(self.images)

        self.report = open(reportPath  + '/' + config + "_Report.txt", "w")
        _stdout = sys.stdout
        sys.stdout = self.report
        print(config)
        print(generator)
        print(criterion)
        self.report.flush()
        sys.stdout = _stdout
        print(self.device)
        print(torch.cuda.device_count())

    def __del__(self):
        self.report.close()

    def approximate(self, dataloaders, num_epochs = 20, resume_train = False):
        if resume_train:
            self.load()

        since = time.time()
        best_acc = 0.0

        for epoch in range(num_epochs):
            _stdout = sys.stdout
            sys.stdout = self.report
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            self.report.flush()
            sys.stdout = _stdout
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.generator.train()
                    self.criterion.train()
                else:
                    self.generator.eval()
                    self.criterion.eval()

                running_lossG = 0.0
                running_lossD = 0.0
                running_corrects = 0
                
                for data in dataloaders[phase]:
                    inputs, targets = data[0], data[1]
                    inputs  = inputs.to(self.device)
                    targets = targets.to(self.device)
                    self.optimizerG.zero_grad()
                    self.optimizerD.zero_grad()
                    outputs = self.generator(inputs)
                    acc = self.accuracy(outputs, targets)

                    if phase == 'train':
                        lossG, lossD = self.criterion(outputs, targets)
                        lossG.backward()
                        self.optimizerG.step()
                        self.generator.zero_grad()
                        lossD.backward()
                        self.optimizerD.step()
                        self.criterion.zero_grad()
                        running_lossG += float(lossG.item()) * inputs.size(0)
                        running_lossD += float(lossD.item()) * inputs.size(0)

                    if phase == 'val':
                        self.display(inputs, outputs, float(acc.mean()), epoch)
                        running_lossG = float("Nan")
                        running_lossD = float("Nan")

                    running_corrects += float(acc.mean()) * inputs.size(0)

                epoch_lossG = running_lossG / len(dataloaders[phase].dataset)
                epoch_lossD = running_lossD / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects / len(dataloaders[phase].dataset)

                _stdout = sys.stdout
                sys.stdout = self.report
                print('{} LossG: {:.4f}  LossD: {:.4f} Accuracy  {:.4f} '.format(
                    phase, epoch_lossG, epoch_lossD, epoch_acc))
                self.report.flush()

                sys.stdout = _stdout
                print('{} LossG: {:.4f}  LossD: {:.4f} Accuracy  {:.4f} '.format(
                    phase, epoch_lossG, epoch_lossD, epoch_acc))
                self.report.flush()

                if phase == 'val':
                    self.schedulerG.step(epoch_acc)
                    self.schedulerD.step()

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        print('curent best_loss ', best_acc)
                        self.save('Best')
                    else:
                        self.save('Regular')

        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val best_acc: {:4f}'.format(best_acc))
        return best_acc

    def estimate(self, test_loader, modelPath=None):
        counter = 0
        self.load(modelPath)
        since = time.time()
        running_corrects = 0
        path = self.images + '/test/'
        os.makedirs(path)
        test_loader.dataset.deprocess = True

        for data in test_loader:
            inputs, targets = data[0], data[1]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.generator(inputs)
            acc = self.accuracy(outputs, targets)
            metric = float(acc.item())

            for i in range(len(outputs)):
                counter = counter + 1
                torchvision.utils.save_image(outputs[i].data, path + "Input_OutPut_Target_" + str(counter) + '_SSIM=' + str("{0:.2f}".format(metric)) + '.jpg')

            running_corrects += acc.item() * inputs.size(0)

        epoch_acc = float(running_corrects) / float(len(test_loader.dataset))

        time_elapsed = time.time() - since

        print('Evaluating complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print(' Accuracy {:.4f} '.format(epoch_acc))

    def save(self, type):
        self.generator.module.to("cpu")
        x = torch.zeros(1, self.dimension, self.image_size, self.image_size)
        path = self.modelPath + "/" + str(self.generator.module.__class__.__name__) + str(self.generator.module.deconv1.__class__.__name__) + str(self.generator.module.activation.__class__.__name__)
        source = "Color" if self.dimension == 3 else "Gray"
        dest =  "2Color" if self.dimension == 3 else "2Gray"
        torch_out = torch.onnx._export(self.generator.module, x, path + source + dest + str(self.image_size)+ "_" + type + ".onnx", export_params=True)
        torch.save(self.generator.module.state_dict(), path + "_" + type  + ".pth")
        self.generator.module.to(self.device)

    def load(self, modelPath=None):
        if modelPath is not None:
            self.generator.module.load_state_dict(torch.load(modelPath))
            print('load generator model')
        else:
            path = self.modelPath + "/" + str(self.generator.module.__class__.__name__) + str(self.generator.module.deconv1.__class__.__name__) + str(self.generator.module.activation.__class__.__name__)
            if os.path.isfile(path + '_Best.pth'):
                self.generator.module.load_state_dict(torch.load(path + '_Best.pth'))
                print('load Best generator ')

    def display(self, inputs, outputs, metric, epoch):
        path = self.images + '/epoch' + str(epoch) + '/'
        flag = os.path.exists(path)
        if flag != True:
            os.makedirs(path)
            self.iteration = 0

        for i in range(len(outputs)):
            self.iteration = self.iteration + 1
            torchvision.utils.save_image( outputs[i].data, path + "OutPut_" + str(self.iteration) + '_SSIM=' + str("{0:.2f}".format(metric)) + '.jpg')

    def process(self, image_dir, modelPath=None):
        c = 0
        self.load(modelPath)

        directory = self.images + '/processed/'
        os.makedirs(directory)
        image_pathes  = [join(image_dir , x) for x in listdir(image_dir) if is_image_file(x)]

        for path in image_pathes:
            image = load_image(path, self.DIMENSION)
            input = self.tensoration(image).unsqueeze(0)
            print(input.shape)
            input =  input.to(self.device)
            output,_,_ = self.generator(input)
            c = c + 1
            torchvision.utils.save_image(output.data, directory +'00000'+ str(c) + '.png', nrow=input.size(0))
            print("Processed : " ,directory  + str(c) + '.png')