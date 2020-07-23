import time
import sys
import os
from os import listdir
from os.path import join

import torch
from torch.autograd import Variable
import torchvision
import shutil

from Dataset import  load_image, is_image_file
from DataParallel import  DataParallelCriterion , DataParallelModel, DataParallelMetric
LEARNING_RATE = 1e-3
LR_THRESHOLD = 1e-7
TRYING_LR = 3
DEGRADATION_TOLERANCY = 7
ACCURACY_TRESHOLD = float(0.0625)
ITERATION_LIMIT = int(1e6)


class NeuralPhotoFilter(object):
    def __init__(self, generator,  criterion, accuracy, dimension, image_size):
        self.cudas = list(range(torch.cuda.device_count()))
        self.generator = DataParallelModel(generator, device_ids=self.cudas, output_device=self.cudas)
        self.criterion = DataParallelCriterion(criterion, device_ids=self.cudas, output_device=self.cudas)
        self.accuracy = DataParallelMetric(accuracy)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.criterion.to(self.device)
        self.accuracy.to(self.device)

        self.dimension = dimension
        self.image_size= image_size

        self.optimizerG = torch.optim.Adam(self.generator.module.parameters(), lr = LEARNING_RATE)
        self.optimizerD = torch.optim.Adam(self.criterion.module.discriminator.parameters(), lr = LEARNING_RATE)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, mode='max', factor=0.1, patience=6, verbose=True, min_lr=LR_THRESHOLD)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, mode='min', factor=0.1, patience=6, verbose=True, min_lr=LR_THRESHOLD)

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
                    self.generator.module.train(True)
                    self.criterion.module.train(True)
                else:
                    self.generator.module.train(False)
                    self.criterion.module.train(False)

                running_lossG = 0.0
                running_lossD = 0.0
                running_corrects = 0
                
                for data in dataloaders[phase]:
                    inputs, targets = data[0], data[1]
                    inputs = Variable(inputs.to(self.device))
                    targets = Variable(targets.to(self.device))
                    outputs = self.generator(inputs)
                    acc = self.accuracy(outputs, targets)  # .mean()

                    if phase == 'train':
                        self.optimizerG.zero_grad()
                        self.optimizerD.zero_grad()
                        lossG, lossD = self.criterion(outputs, targets)
                        lossG.backward()
                        self.optimizerG.step()
                        lossD.backward()
                        self.optimizerD.step()
                        running_lossG += lossG.mean() * inputs.size(0)
                        running_lossD += lossD.mean() * inputs.size(0)

                    if phase == 'val':
                        self.display(outputs, float(acc.mean()), epoch)
                        running_lossG = float('NaN')
                        running_lossD = float('NaN')

                    running_corrects += acc.mean() * inputs.size(0)

                epoch_lossG = float(running_lossG) / float(len(dataloaders[phase].dataset))
                epoch_lossD = float(running_lossD) / float(len(dataloaders[phase].dataset))
                epoch_acc = float(running_corrects) / float(len(dataloaders[phase].dataset))

                _stdout = sys.stdout
                sys.stdout = self.report
                print('{} Loss: {:.4f} Accuracy  {:.4f} '.format(
                    phase, epoch_lossG, epoch_acc))
                self.report.flush()

                sys.stdout = _stdout
                print('{} Loss: {:.4f} Accuracy  {:.4f} '.format(
                    phase, epoch_lossG, epoch_acc))
                self.report.flush()

                if phase == 'val':
                        self.schedulerG.step(epoch_acc)
                        self.schedulerD.step(epoch_lossD)

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
            inputs, targets = Variable(inputs.to(self.device)), Variable(targets.to(self.device))
            outputs = self.generator(inputs)
            acc = self.accuracy(outputs, targets)
            metric = float(acc.item())
            counter = counter + 1

            if len(data) > 2:
                denoised = data[2]
                denoised = denoised.to(self.device)
                acc2 = self.accuracy(denoised, targets)
                metric2 = float(acc2.item())
                result = torch.cat([inputs.data, outputs.data, denoised, targets.data], dim=0)
                torchvision.utils.save_image(result, path + "Input_DeepNeural_Conventional_Target_" + str(counter) + '_SSIM(dnn)=' + str("{0:.2f}".format(metric)) +'_SSIM(non_dnn)='+ str("{0:.2f}".format(metric2)) + '.png', nrow=inputs.size(0))
            else:
                result = torch.cat([inputs.data, outputs.data, targets.data], dim=0)
                torchvision.utils.save_image(result, path + "Input_DeepNeural_Target_" + str(counter) + '_SSIM=' + str(
                    "{0:.2f}".format(metric)) + '.png', nrow=inputs.size(0))

            running_corrects += acc.item() * inputs.size(0)

        epoch_acc = float(running_corrects) / float(len(test_loader.dataset))

        time_elapsed = time.time() - since

        print('Evaluating complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print(' Accuracy {:.4f} '.format(epoch_acc))

    def save(self, type):
        self.generator.module.to("cpu")
        x = Variable(torch.zeros(1, self.dimension, self.image_size, self.image_size))
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

    def display(self, outputs, metric, epoch):
        path = self.images + '/epoch' + str(epoch) + '/'
        flag = os.path.exists(path)
        if flag != True:
            os.makedirs(path)
            self.iteration = 0

        for i in range(len(outputs)):
            self.iteration = self.iteration + 1
            torchvision.utils.save_image(outputs[i].data, path + "Input_OutPut_Target_" + str(self.iteration) + '_SSIM=' + str("{0:.2f}".format(metric)) + '.jpg')

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
            input =  Variable(input.to(self.device))
            output,_,_ = self.generator(input)
            c = c + 1
            torchvision.utils.save_image(output.data, directory +'00000'+ str(c) + '.png', nrow=input.size(0))
            print("Processed : " ,directory  + str(c) + '.png')