
import torch.nn as nn
import torch.nn.functional as F


class FrozenCLIP(nn.Module):
    def __init__(self, clip_model, number_of_classes):
        super(FrozenCLIP, self).__init__()
        self.clip_model = clip_model
        self.num_classes = number_of_classes
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Dyn-ANN module
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, self.num_classes)


    def forward(self, x):
        x = self.clip_model.encode_image(x)
        x = x.to(self.fc1.weight.dtype)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x






