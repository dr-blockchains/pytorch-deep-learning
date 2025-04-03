# Daniel Armani, PhD @2025

import sys
print(f"Python version: {sys.version}")

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import torch
print(f"PyTorch version: {torch.__version__}")

import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Generate Some fake data:

np.random.seed(2025)
torch.manual_seed(2025)

n_samples = 10000
n_inputs = 5
n_outputs = 3

# Introduce correlations among X0, X1, X2 & X3 but no correlation with X4!
correlation_matrix = np.eye(n_inputs)
correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.6
correlation_matrix[2, 3] = correlation_matrix[3, 2] = -0.2

correlation_matrix

# Cholesky decomposition for correlated random variables
cov_matrix = np.linalg.cholesky(correlation_matrix)
X = np.dot(np.random.randn(n_samples, n_inputs), cov_matrix)

X.shape

# Linear relationship:
raw_y1 = 1.5*X[:, 0] - 0.5*X[:, 1] + 0.1*X[:, 2] + np.random.normal(0, 0.5, size=(n_samples,))

# With interactions:
raw_y2 = 0.7*X[:, 1] + 0.3*X[:, 2] + 0.5*X[:, 0]*X[:, 3] + np.random.uniform(-0.5, 0.5, size=(n_samples,))

# Complex non-linear interactions:
epsilon = 1e-10
raw_y3 = (
    np.sin(2*X[:, 0]) * np.cos(X[:, 1])  # Trigonometric interaction
    + np.tanh(X[:, 2] + X[:, 3])  # Non-linear transformation of sum
    + np.exp(X[:, 4]/5) / (1 + np.exp(X[:, 4]/5))  # Sigmoid-like function
    + 0.3 * np.log(np.abs(X[:, 0]*X[:, 2]) + epsilon)  # Log of interaction
    - 0.5 * (X[:, 1]**2 * X[:, 3])  # Higher-order interaction
    + 0.2 * np.maximum(0, X[:, 2])  # ReLU-like function
    + np.random.normal(0, 0.3, size=(n_samples,))  # Noise
)

# Get thresholds for exactly 50% split binary outcome
threshold_y1 = np.median(raw_y1)
threshold_y2 = np.median(raw_y2)
threshold_y3 = np.median(raw_y3)

y1 = (raw_y1 > threshold_y1).astype(int)
y2 = (raw_y2 > threshold_y2).astype(int)
y3 = (raw_y3 > threshold_y3).astype(int)

Y = np.column_stack([y1, y2, y3])

Y.shape

# Split and prepare data:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2025)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
Y_train_tensor = torch.FloatTensor(Y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
Y_test_tensor = torch.FloatTensor(Y_test)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

def num_param(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 1. Multivariate Logistic Regression:

lr_models = []
lr_predictions = np.zeros((X_test.shape[0], n_outputs))

for i in range(n_outputs):
    lr = LogisticRegression(C=1e6, random_state=2, max_iter=10000, tol=1e-8)
    lr.fit(X_train_scaled, Y_train[:, i])
    lr_models.append(lr)
    print(f"\n Coefficients: {lr.coef_[0]}, Intercept: {lr.intercept_[0]:.3f}")

    lr_predictions[:, i] = lr.predict(X_test_scaled)

    print(f"Output y{i+1} accuracy: {lr.score(X_test_scaled, Y_test[:, i])}")

# 2. Simple Neural Network:

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

simple_nn = SimpleNN(n_inputs, n_outputs)

print(f"Number of parameters: {num_param(simple_nn)}")
simple_nn

# 3. Neural Network with one linear hidden layer with no activation function:

class LinearHiddenNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearHiddenNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

linear_hidden = LinearHiddenNN(n_inputs, 3, n_outputs)
print(f"Number of parameters: {num_param(linear_hidden)}")
linear_hidden

# 4. Neural Network with a hidden layer with ReLU activation function:

class ReluHiddenNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ReluHiddenNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

MAX_NEURONS = 30
MAX_NEURONS

# Create an array of grayscale values from light to dark
grays = np.linspace(0.7, 0.2,MAX_NEURONS)
colors = [(g, g, g) for g in grays]

# Create instances with different hidden sizes

relu_hidden = []
for hidden_size in range(1, MAX_NEURONS+1):
    instance = ReluHiddenNN(n_inputs, hidden_size, n_outputs)
    relu_hidden.append(instance)
    print(f"\n Linear Hidden with {hidden_size} Neurons and {num_param(instance)} parameters:")
    print(instance)

# 5. The Neural Network with two hidden layers (ReLU):

class TwoHiddenNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(TwoHiddenNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

two_hidden = TwoHiddenNN(n_inputs, 8, 8, n_outputs) # 16 Neurons in the hidden layers
print(f"Number of parameters: {num_param(two_hidden)}")
two_hidden

# Function to train the Neural Network models:

def train_model(model, X_train, Y_train, max_epochs=9000, lr=0.1, tol=1e-20):
    model = model.to(device)

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20,gamma=0.9)

    model.train()

    losses = []
    prev_loss = float('inf')

    start_time = time.time()

    for epoch in range(max_epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, Y_train)
        losses.append(loss.item())
        current_loss = loss.item()

        if abs(prev_loss - current_loss) < tol:
            print(f"Converged at epoch {epoch} with loss {current_loss:.8f}")
            break

        if (epoch) % 100 == 0:
            print(f"Epoch {epoch}/{max_epochs}, Loss: {loss.item():.4f}")

        prev_loss = current_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        scheduler.step()

    if epoch == max_epochs - 1:
        print(f"Reached max epochs {max_epochs} with loss {current_loss:.8f}")

    print(f"Training duration = {time.time() - start_time} seconds")

    return losses

train_model

# Training Models and Obtaining Losses:

print(f"\n Simple NN, training {num_param(simple_nn)} parameters: ")
simple_losses = train_model(simple_nn, X_train_tensor, Y_train_tensor)

print(f"\n One Linear Hidden NN, training {num_param(linear_hidden)} parameters: ")
linear_hidden_losses= train_model(linear_hidden, X_train_tensor, Y_train_tensor)

relu_hidden_losses = [[] for _ in range(len(relu_hidden))]
for i, instance in enumerate(relu_hidden):
  print(f"\n ReLU Hidden Layer with {i+1} Neurons, training {num_param(instance)} parameters: ")
  relu_hidden_losses[i] = train_model(instance, X_train_tensor, Y_train_tensor)

print(f"\n Two ReLU Hidden NN, training {num_param(two_hidden)} parameters: ")
two_hidden_losses = train_model(two_hidden, X_train_tensor, Y_train_tensor)

print(f"Simple No Hidden Layer: {simple_losses[-1]}")

print(f"One Linear Hidden Layer : {linear_hidden_losses[-1]}")

for i, instance in enumerate(relu_hidden):
  print(f"One Relu Hidden {i+1}: {relu_hidden_losses[i][-1]}")

print(f"Two ReLU Hidden Layers     : {two_hidden_losses[-1]}")

# Plots for the losses:

plt.figure(figsize=(15, 10))

plt.plot(simple_losses, label='Simple NN', color='b')
plt.plot(linear_hidden_losses, label='One Linear Hidden', color='g')

for i, instance in enumerate(relu_hidden):
  if i == 15: color = 'r'
  else: color = colors[i]
  plt.plot(relu_hidden_losses[i], label=f'One ReLU Hidden {i+1}', color=color)

plt.plot(two_hidden_losses, label='Two ReLU Hidden', color='m')

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss for Different Neural Networks')

plt.legend()
plt.grid(True)
plt.show()

# Plots for the losses:

plt.figure(figsize=(15, 10))

plt.plot(simple_losses, label='Simple NN', color='b')
plt.plot(linear_hidden_losses, label='Linear Hidden', color='g')
plt.plot(relu_hidden_losses[0], label=f'One ReLU Hidden {1}', color=colors[0])
plt.plot(relu_hidden_losses[1], label=f'One ReLU Hidden {2}', color=colors[1])
plt.plot(relu_hidden_losses[2], label=f'One ReLU Hidden {3}', color=colors[2])
plt.plot(relu_hidden_losses[15], label=f'One ReLU Hidden {16}', color='r')
plt.plot(relu_hidden_losses[29], label=f'One ReLU Hidden {30}', color=colors[29])
plt.plot(two_hidden_losses, label='Two ReLU Hidden', color='m')

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss for Selected Neural Networks')

plt.legend()
plt.grid(True)
plt.show()

# Plot for final losses for the linear hidden layer

plt.figure(figsize=(20, 10))

plt.axhline(y=simple_losses[-1], color='g', linestyle='--', label='Simple NN')
plt.axhline(y=linear_hidden_losses[-1], color='g', linestyle='--', label='Linear Hidden')
plt.plot(range(1, MAX_NEURONS + 1), [relu_hidden_losses[i][-1] for i in range(MAX_NEURONS)], marker='o') #, color='k'
plt.axhline(y=two_hidden_losses[-1], color='m', linestyle='--', label='Two Hidden (8,8)')

plt.xticks(range(1, MAX_NEURONS + 1))
plt.xlabel('Number of Neurons in the ReLU Hidden Layer')
plt.ylabel('Final Loss')
plt.title('Final Loss vs. Number of Neurons in the ReLU Hidden Layer')

plt.legend()
plt.grid(True)
plt.show()

# Function to Evaluate the Neural Network Models:

def evaluate_model(y_true, y_pred, message):
    accuracies = []
    print(f"\n--- {message} Results ---")

    for i in range(n_outputs):
        acc = (y_true[:, i] == y_pred[:, i]).mean()
        accuracies.append(acc)
        print(f"Output y{i+1} Accuracy: {acc:.4f}")

    # print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    return accuracies

evaluate_model

# Prediction and Evaluations:

simple_nn.eval()
linear_hidden.eval()

for i,instance in enumerate(relu_hidden):
  instance.eval()
relu_hidden_pred = []

two_hidden.eval()

X_test_tensor = X_test_tensor.to(device)
Y_test_tensor = Y_test_tensor.to(device)

with torch.no_grad():
    simple_nn_pred = (simple_nn(X_test_tensor) > 0.5).float().cpu().numpy()
    linear_hidden_pred = (linear_hidden(X_test_tensor) > 0.5).float().cpu().numpy()

    for y in relu_hidden:
        relu_hidden_pred.append((y(X_test_tensor) > 0.5).float().cpu().numpy())

    two_nn_pred = (two_hidden(X_test_tensor) > 0.5).float().cpu().numpy()

regression_accuracies = evaluate_model(Y_test, lr_predictions, "Multivariate Logistic Regression")
simple_nn_accuracies = evaluate_model(Y_test, simple_nn_pred, "Neural Network with No Hidden Layer")
linear_nn_accuracies = evaluate_model(Y_test, linear_hidden_pred, "NN with One Linear Hidden Layer)")

relu_nn_accuracies = [[] for _ in range(len(relu_hidden))]

for i in range(len(relu_hidden_pred)):
  relu_nn_accuracies[i] = evaluate_model(Y_test, relu_hidden_pred[i], f"NN with One ReLU Hidden Layer ({i+1}N))")

two_nn_accuracies = evaluate_model(Y_test, two_nn_pred, "NN with Two ReLU Hidden Layers)")

# Barcharts to Compare the Accuracies across Models:

labels = ['Output y1', 'Output y2', 'Output y3']
x = np.arange(len(labels))
width = 0.04

fig, ax = plt.subplots(figsize=(15, 15))
rects1 = ax.bar(x - 3*width, regression_accuracies, width, label='Logistic Regression', color='c')
rects2 = ax.bar(x - 2*width, simple_nn_accuracies, width, label='Simple NN', color='b')
rects3 = ax.bar(x - width, linear_nn_accuracies, width, label='Linear Hidden NN', color='g')

for i in range(len(relu_nn_accuracies)):
  if i == 15: color = 'r'
  else: color = colors[i]
  rects4 = ax.bar(x + i*width/2, relu_nn_accuracies[i], width/2, label=f'Relu Hidden ({i+1}N))', color=color)

# rects4 = ax.bar(x + 15*width/2, relu_nn_accuracies[15], width/2, label=f'Relu Hidden ({16}N))', color='r')

gap = (len(relu_hidden)+1)*width/2

rects5 = ax.bar(x + gap, two_nn_accuracies, width, label='Two ReLU Hidden', color='m')

ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Model Accuracies on Different Outputs')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1, 1))

plt.grid(True)
plt.show()

# Compare weights and coefficients:

print("\n--- Comparison: Logistic Regression vs Simple Neural Network ---")
with torch.no_grad():
    for i in range(n_outputs):
        print(f"\nOutput y{i+1}:")
        # Extract sklearn logistic regression weights (coefficients and intercept)
        lr_weights = np.concatenate([lr_models[i].coef_[0], [lr_models[i].intercept_[0]]])

        # Extract PyTorch weights (weights and bias)
        nn_weights = np.concatenate([
            simple_nn.linear.weight.data[i].cpu().numpy(),
            [simple_nn.linear.bias.data[i].cpu().numpy()]
        ])

        print(f"Logistic Regression Coefficients: {lr_weights.round(4)}")
        print(f"Simple NN weights: {nn_weights.round(4)}")
        print(f"Mean absolute difference: {np.mean(np.abs(lr_weights - nn_weights)):.6f}")

