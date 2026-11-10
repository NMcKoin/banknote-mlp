import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim



# for training reproducability.
torch.manual_seed(1234)


# Load the dataset
banknotes = pd.read_csv('./data/banknote-auth.csv')

print("\nDataset Shape:")
print(banknotes.shape)

print("\nFirst 10 Rows:")
print(banknotes.head(10))

print("\nDataset Information:")
banknotes.info()
print("\n \n") #cleaning up the terminal view here.

# Separate input features and target labels
X = banknotes[
    ["variance",
     "skewness",
     "kurtosis",
     "entropy"]
].values

y = banknotes["label"].values

# Split data into train and test sets 80% training 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# convert numpy arrays to Pytorch tensors - Float32 req for neural network computation
X_train = torch.tensor(X_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)

#Reshape labels to match network output shape
y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).view(-1, 1)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
).view(-1, 1)

# ======================
# DEFINE MLP MODEL
# ======================
# The network accepts 4 input features from the Banknote dataset and predicts
# whether a banknote is authentic or counterfeit

class BanknoteMLP(nn.Module):
  """
  Simple multilayer perceptron
  Input layer - 4 features
  Hidden layer - 8 neurons with ReLU
  Output layer - 1 neuron with Sigmoid
  """

  def __init__(self):
      super().__init__()

      self.network = nn.Sequential(
         nn.Linear(4, 8),
         nn.ReLU(),
         nn.Linear(8, 1),
         nn.Sigmoid()
      )

  def forward(self, x):
      return self.network(x)

# =======================
# Training Loop
#========================
# init model
model = BanknoteMLP()

#loss function used to measure prediction error (Binary Cross Entropy)
criterion = nn.BCELoss()

#Optimizer used to update model weights. Adam is an adaptive optimization algorithm that converges faster than gradient descent
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

# Train the neural network for 20 complete passes
epochs = 20

print("Training:\n")

for epoch in range(epochs):

    #forward pass
    predictions = model(X_train)

    #calculate loss
    loss = criterion(
        predictions,
        y_train
    )

    #clear old gradients
    optimizer.zero_grad()

    #backprop
    loss.backward()

    #update network weights
    optimizer.step()


    #display training progress every 5 epochs.
    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )


#=======================
# Validation
#=======================

# Disable gradient calcs during evaluation to ensure model weights are not updated while making predictions
with torch.no_grad():
    
    # predictions for the training dataset
    train_predictions = (
        model(X_train) >= 0.5
    ).float()

    # predictions for the testing dataset
    test_predictions = (
        model(X_test) >= 0.5
    ).float()

#Calculate training accuracy
train_accuracy = (
    train_predictions.eq(y_train)
    .sum()
    .item()
    / len(y_train)
)

# Calculate testing accuracy
test_accuracy = (
    test_predictions.eq(y_test)
    .sum()
    .item()
    / len(y_test)
)

print("\nModel Performance:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy:  {test_accuracy:.4f}")

#=================
#Confusion Matrix
#=================

cm = confusion_matrix(
    y_test.numpy(),
    test_predictions.numpy()
)

print("\nConfusion Matrix:")
print(cm)

#Visualize the confusion matrix - heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Authentic', 'Counterfeit'],
    yticklabels=['Authentic', 'Counterfeit']
)

plt.title("Banknote Classificaton Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.savefig("./output/confusion_matrix.png")

plt.close()