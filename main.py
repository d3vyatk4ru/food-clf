
from prepare_data import dataloaders, dataset_sizes
from resnet import ResNet, BasicBlock
import torch
from utils import imshow, train_model

CLASSES = ['Bread', 'Dessert', 'Meat', 'Soup']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = ResNet(BasicBlock, [5, 3, 5, 3], num_classes=4)
model_ft.to(device)

criterion = torch.nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=1e-2, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.2)

model_ft, loss_train, acc_train, loss_test, acc_test = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    dataloaders,
    dataset_sizes,
    device,
    num_epochs=25)