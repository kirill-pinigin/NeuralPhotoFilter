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
from  NeuralModel import NeuralModel

LEARNING_RATE = 1e-3
LR_THRESHOLD = 1e-7
TRYING_LR = 3
DEGRADATION_TOLERANCY = 7
ACCURACY_TRESHOLD = float(0.0625)
ITERATION_LIMIT = int(1e6)


class NeuralPhotoFilter(object):
    def __init__(self, generator,  criterion, accuracy, optimizer):
        self.cudas = list(range(torch.cuda.device_count()))
        self.model = torch.nn.DataParallel(NeuralModel(generator, criterion))
        self.accuracy  = torch.nn.DataParallel(accuracy, device_ids= self.cudas)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.criterion.to(self.device)
        self.accuracy.to(self.device)
        self.optimizer = optimizer
        self.iteration = int(0)
        self.tensoration = torchvision.transforms.ToTensor()

        config = str(generator.__class__.__name__) + '_' + str(generator.deconv1.__class__.__name__) +  '_' + str(generator.activation.__class__.__name__)
        config += '_' + str(criterion.__class__.__name__)
        config += "_" + str(optimizer.__class__.__name__)
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
        path = self.modelPath +"/"+ str(self.generator.module.__class__.__name__) + str(self.generator.module.deconv1.__class__.__name__) + str(self.generator.module.activation.__class__.__name__)
        if resume_train and os.path.isfile(path + '_Best.pth'):
            print( "RESUME training load Best generator")
            self.generator.module.load_state_dict(torch.load(path + '_Best.pth'))
            self.generator.to(self.device)

        since = time.time()
        best_acc = 0.0
        best_loss = 1e6
        counter = 0
        i = int(0)
        degradation = 0

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
                    self.generator.train(True)
                else:
                    self.generator.train(False)

                running_loss = 0.0
                running_corrects = 0
                for data in dataloaders[phase]:
                    inputs, targets = data[0], data[1]
                    inputs = Variable(inputs.to(self.device))
                    targets = Variable(targets.to(self.device))
                    self.optimizer.zero_grad()
                    outputs = self.generator(inputs)
                    acc = self.accuracy(outputs, targets)#.mean()
                    loss = self.criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    #if phase == 'val' :#and acc.item() > best_acc:
                        #self.display(inputs, outputs, targets, float(acc.item()), epoch)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += acc.item() * inputs.size(0)

                epoch_loss = float(running_loss) / float(len(dataloaders[phase].dataset))
                epoch_acc = float(running_corrects) / float(len(dataloaders[phase].dataset))

                _stdout = sys.stdout
                sys.stdout = self.report
                print('{} Loss: {:.4f} Accuracy  {:.4f} '.format(
                    phase, epoch_loss, epoch_acc))
                self.report.flush()

                sys.stdout = _stdout
                print('{} Loss: {:.4f} Accuracy  {:.4f} '.format(
                    phase, epoch_loss, epoch_acc))
                self.report.flush()

                if phase == 'val' and epoch_acc > best_acc:
                    counter = 0
                    degradation = 0
                    best_acc = epoch_acc
                    print('curent best_loss ', best_acc)
                    self.save('Best')
                else:
                    counter += 1
                    self.save('Regular')

            if counter > TRYING_LR * 2:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                    if lr >= LR_THRESHOLD:
                        param_group['lr'] = lr * 2e-1
                        print('! Decrease LearningRate !', lr)

                counter = 0
                degradation += 1
            if degradation > DEGRADATION_TOLERANCY:
                print('This is the end! Best val best_loss: {:4f}'.format(best_acc))
                return best_acc

        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val best_acc: {:4f}'.format(best_acc))
        return best_acc

    def estimate(self, test_loader, modelPath=None):
        counter = 0
        if modelPath is not None:
            self.generator.load_state_dict(torch.load(modelPath))
            print('load generator model')
        else:
            path = self.modelPath + "/" + str(self.generator.__class__.__name__) + str(self.generator.deconv1.__class__.__name__) + str(self.generator.activation.__class__.__name__)
            self.generator.load_state_dict(torch.load(path + '_Best.pth'))
            print('load Best generator ')
        print(len(test_loader.dataset))
        i = 0
        since = time.time()
        self.generator.train(False)
        self.generator.eval()
        self.generator.to(self.device)
        running_loss = 0.0
        running_corrects = 0
        path = self.images + '/test/'
        os.makedirs(path)
        test_loader.dataset.deprocess = True

        for data in test_loader:
            inputs, targets = data[0], data[1]
            inputs, targets = Variable(inputs.to(self.device)), Variable(targets.to(self.device))

            outputs = self.generator(inputs)
            acc = self.accuracy(outputs, targets)
            loss = self.criterion(outputs, targets)

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

            running_loss += loss.item() * inputs.size(0)
            running_corrects += acc.item() * inputs.size(0)

        epoch_loss = float(running_loss) / float(len(test_loader.dataset))
        epoch_acc = float(running_corrects) / float(len(test_loader.dataset))

        time_elapsed = time.time() - since

        print('Evaluating complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Loss: {:.4f} Accuracy {:.4f} '.format( epoch_loss, epoch_acc))

    def save(self, type):
        model = self.generator.module
        model = model.cpu()
        model.eval()
        x = Variable(torch.zeros(1, DIMENSION, IMAGE_SIZE, IMAGE_SIZE))
        path = self.modelPath + "/" + str(self.generator.module.__class__.__name__) + str(self.generator.module.deconv1.__class__.__name__) + str(self.generator.module.activation.__class__.__name__)
        source = "Color" if DIMENSION == 3 else "Gray"
        dest =  "2Color" if DIMENSION == 3 else "2Gray"
        torch_out = torch.onnx._export(model, x, path + source + dest + str(IMAGE_SIZE)+ "_" + type + ".onnx", export_params=True)
        torch.save(model.state_dict(), path + "_" + type  + ".pth")
        self.generator.to(self.device)

    def display(self, inputs, outputs, targets, metric, epoch):
        path = self.images + '/epoch' + str(epoch) + '/'
        flag = os.path.exists(path)
        if flag != True:
            os.makedirs(path)
            self.iteration=0
        self.iteration = self.iteration + 1

        for i in range(len(inputs)):
            result = torch.cat([inputs[i].data, outputs[i].data, targets[i].data], dim=0)
            torchvision.utils.save_image(result, path + "Input_OutPut_Target_" + str(self.iteration) + '_SSIM=' + str("{0:.2f}".format(metric)) + '.jpg', nrow= 8)

    def process_dataset(self, test_loader, modelPath=None):
        counter = 0
        if modelPath is not None:
            self.generator.load_state_dict(torch.load(modelPath))
            print('load generator model')
        else:
            path = self.modelPath + "/" + str(self.generator.__class__.__name__) + str(self.generator.deconv1.__class__.__name__) + str(self.generator.activation.__class__.__name__)
            self.generator.load_state_dict(torch.load(path + '_Best.pth'))
            print('load Best generator ')
        print(len(test_loader.dataset))
        i = 0
        since = time.time()
        self.generator.train(False)
        self.generator.eval()
        self.generator.to(self.device)
        path1 = self.images + '/bad/'
        os.makedirs(path1)
        path2 = self.images + '/good/'
        os.makedirs(path2)
        test_loader.dataset.deprocess = True

        for data in test_loader:
            inputs, targets = data[0], data[1]
            inputs, targets = Variable(inputs.to(self.device)), Variable(targets.to(self.device))
            outputs = self.generator(inputs)
            counter = counter + 1
            torchvision.utils.save_image(outputs.data, path1 + "00000" + str(counter) + '.png', nrow=inputs.size(0))
            torchvision.utils.save_image(targets.data, path2 + "00000" + str(counter) + '.png', nrow=inputs.size(0))
            print("Processed : " , path + "00000" + str(counter) + '.png')


    def process(self, image_dir, modelPath=None):
        counter = 0
        if modelPath is not None:
            self.generator.load_state_dict(torch.load(modelPath))
            print('load generator model')
        else:
            path = self.modelPath + "/" + str(self.generator.__class__.__name__) + str(self.generator.deconv1.__class__.__name__) + str(self.generator.activation.__class__.__name__)
            self.generator.load_state_dict(torch.load(path + '_Best.pth'))
            print('load Best generator ')

        self.generator.train(False)
        self.generator.eval()
        device = "cpu"
        self.generator.to(device)
        directory = self.images + '/processed/'
        os.makedirs(directory)
        image_pathes  = [join(image_dir , x) for x in listdir(image_dir) if is_image_file(x)]

        for path in image_pathes:
            image = load_image(path, DIMENSION)
            input = self.tensoration(image).unsqueeze(0)
            print(input.shape)
            input =  Variable(input.to(device))
            output = self.generator(input)
            counter = counter + 1
            torchvision.utils.save_image(output.data, directory +'00000'+ str(counter) + '.png', nrow=input.size(0))
            print("Processed : " ,directory  + str(counter) + '.png')
