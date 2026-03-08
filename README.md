# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

### STEP 1: 

Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding,
and encode the target class (Segmentation).

### STEP 2: 

Split the dataset into training and testing sets, then normalize the input features using 
StandardScaler for better neural network performance.

### STEP 3: 

Convert the scaled training and testing data into PyTorch tensors and create
DataLoader objects for batch-wise training and evaluation.

### STEP 4: 

Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, 
ending with an output layer for multi-class classification.

### STEP 5: 

Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, 
loss calculation, backpropagation, and weight updates over multiple epochs.


### STEP 6: 


Evaluate the trained model on test data using accuracy, confusion matrix,
and classification report, and perform prediction on a sample input.


## PROGRAM

### Name: Monika A
 
### Register Number: 212224240094

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 =nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)





    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x

        
# Initialize the Model, Loss Function, and Optimizer
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')'

# Initialize model

model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)

```

### Dataset Information

<img width="1267" height="256" alt="image" src="https://github.com/user-attachments/assets/0e1fc17f-1106-4aef-ad63-cba615abd387" />


### OUTPUT

![WhatsApp Image 2026-03-08 at 5 42 46 PM](https://github.com/user-attachments/assets/4bb7b89b-94c0-4bd8-b4d8-ec5704f2614c)

## Confusion Matrix

![WhatsApp Image 2026-03-08 at 5 47 55 PM](https://github.com/user-attachments/assets/7c1baf21-c153-414f-a7db-62e132a28b5a)

## Classification Report

![WhatsApp Image 2026-03-08 at 5 48 52 PM](https://github.com/user-attachments/assets/656f104b-e966-4d99-9d0d-cd3fb5814dd3)


### New Sample Data Prediction

![WhatsApp Image 2026-03-08 at 5 50 12 PM](https://github.com/user-attachments/assets/3fcd219d-634d-48d8-bf62-354246e18559)


## RESULT

This program has been executed successfully.

