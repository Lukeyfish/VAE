from torch import nn
import torch.nn.functional as F

# VGG11 architecture
class VGG11(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x



# VGG16 architecture
class VGG16(nn.Module):
    def __init__(self, in_channels, n_classes, dropout):
        super(VGG16, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.dropout = dropout

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), # Conv 1-1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Conv 1-2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Pool 1
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Conv 2-1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # Conv 2-2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Pool 2
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Conv 3-1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Conv 3-2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Conv 3-3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Pool 3
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # Conv 4-1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv 4-2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv 4-3
            nn.MaxPool2d(kernel_size=2, stride=2), # Pool 4
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv 5-1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv 5-2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv 5-3
            nn.MaxPool2d(kernel_size=2, stride=2), # Pool 5
        )
        
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # Average Pooling before FC layers
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),  # FC1
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 4096),  # FC2
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, n_classes),  # Output layer
        )
        
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    
class BasicCNN(nn.Module):
    def __init__(self, dropout_rate):
        super(BasicCNN, self).__init__()
        # Input: [batch_size, 1, 28, 28]
        self.conv1 = nn.Conv2d(1, 32, 3)  # Output: [batch_size, 32, 26, 26]
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3)  # Output: [batch_size, 64, 24, 24]
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Flattening: [batch_size, 64*5*5]
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(dropout_rate)  # Dropout with 50% probability

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        # Input: [batch_size, 1, 28, 28]
        x = F.relu(self.bn1(self.conv1(x)))
        # Shape: [batch_size, 32, 26, 26]
        x = F.max_pool2d(x, 2)
        # Shape: [batch_size, 32, 13, 13]
        
        x = F.relu(self.bn2(self.conv2(x)))
        # Shape: [batch_size, 64, 11, 11]
        x = F.max_pool2d(x, 2)
        # Shape: [batch_size, 64, 5, 5]
        
        x = x.view(-1, 64 * 5 * 5)  # Flattening
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
