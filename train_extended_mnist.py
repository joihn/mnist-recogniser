# %%
from labelLoader import Loader
import torch
from lenet5_like import LeNet5_like
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# %%
def train(train_input, train_label, model, optimizer, log_interval, epoch, criterion):
    model.train()
    for g in range(train_input.shape[0]):
        inputs = train_input[g].view(-1, 1, 28, 28)
        labels = train_label[g].view(-1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        print(f"train epoch {epoch}, batch {g}, loss {loss}")


def test(test_input, test_label, model, criterion, plotFlag=False):
    model.eval()  # Sets the module in evaluation mode.

    # `with` allows you to ensure that a resource is "cleaned up"
    # when the code that uses it finishes running, even if exceptions are thrown.
    with torch.no_grad():  # It will reduce memory consumption for computations that would otherwise have requires_grad=True.

        input = test_input.view(-1, 1, 28, 28)
        labels = test_label.view(-1).long()

        outputs = model(input)
        test_loss = criterion(outputs, labels).item()  # sum up batch loss
        pred = outputs.argmax(dim=1, keepdim=True)

        test_label[test_label in ["J", "Q", "K"]] = 10

        correctPred = pred.eq(labels.view_as(pred)).sum().item()
        logArrayClass = (labels==10)
        correctPredClass10 = (pred[logArrayClass]==10).sum()

    print(f"the test loss is {test_loss}, the test error is {(1 - correctPred / (test_input.shape[0]*test_input.shape[1])) * 100}% \
            the class 10 error is {(1- correctPredClass10/logArrayClass.sum())* 100} %")

    if plotFlag:
        wrongPred = torch.bitwise_not(pred.eq(labels.view_as(pred)))
        for i, w in enumerate(wrongPred):
            if w:
                plt.imshow(input[i, 0, :, : ])
                plt.title(f"I guessed wrongly {pred[i]} instead of {labels[i]} // item.n {i} which is round {i%13} and player {i%4}")
                plt.show()

# %%
loader = Loader("C:/Users/maxim/Google Drive/Epfl/MA4/Img analysis/project/iapr/project/train_games/")

model = LeNet5_like()

model.load_state_dict(torch.load("saved_models/mod1.pkl"))
model.fc3 = nn.Linear(84, 11)

img, lab = loader.getNumberAndLabel(6, 5, 4)
meanMnist, stdMnist = 0.1307, 0.3081

tempImg = torch.from_numpy(img).view(-1, 1, 28, 28).float().sub(meanMnist).div(stdMnist)
output = model(tempImg)



model.conv1.weight.requires_grad = False
model.conv2.weight.requires_grad = False
model.fc1.weight.requires_grad = True
model.fc2.weight.requires_grad = True
model.fc3.weight.requires_grad = True

model.conv1.bias.requires_grad = False
model.conv2.bias.requires_grad = False
model.fc1.bias.requires_grad = True
model.fc2.bias.requires_grad = True
model.fc3.bias.requires_grad = True


# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)

# optimizer = optim.Adam(model.parameters(), lr=1e-2)

criterion = nn.CrossEntropyLoss(reduction='mean')

epochs = 10  # how many epochs to train for
log_interval = 1  # how many batches to wait before logging training status
train_input, train_label = loader.getTrain()
test_input, test_label = loader.getTest()

test(test_input, test_label, model, criterion)

for epoch in range(1, epochs + 1):
    train(train_input, train_label, model, optimizer, log_interval, epoch, criterion)
    test(test_input, test_label, model, criterion, True)

#%%
torch.save(model.state_dict(), r"C:\Users\maxim\Google Drive\Epfl\MA4\Img analysis\project\mnist recogniser\saved_models\modExtended.pkl")


