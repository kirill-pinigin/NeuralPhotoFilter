import torch

class NeuralModel(torch.nn.Module):
    def __init__(self, generator, criterion):
        super(NeuralModel, self).__init__()
        self.generator = generator
        self.criterion = criterion

    def forward(self, inputs ,targets):
       outputs = self.generator(inputs)
       lossG = self.criterion(outputs,targets)
       lossD = self.criterion.update(outputs,targets)
       return outputs, lossG, lossD
