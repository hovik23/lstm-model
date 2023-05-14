# LSTM Model using PyTorch

This repository contains an implementation of a function regression with a Long Short-Term Memory (LSTM) model using PyTorch. The model is trained on a sequential dataset generated from the following function:

$$8000 \cdot (x + 20 \cdot \sin(x))$$


## Requirements

- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Implementation

### Preparing dataset

A dataset was generated from $8000(x + 20\sin(x))$ function. $X$ is an array of $4000$ equally spaced values between -20 and 20. The output of the function is stored in $y$. Also, the values were converted to pandas DataFrame `training_data` to display the values and then separate them into corresponding variables for training:

```
def func(x):
    return 8000 * (x + 20 * np.sin(x))

x = np.linspace(-20, 20, 4000, dtype=np.float32)
y = func(x)

training_data = pd.DataFrame(zip(x, y))
training_data
```

After, the dataset is split into input `x_raw` and target `y_raw` arrays:

```
x_raw = training_data.iloc[:, :].values
y_raw = training_data.iloc[:, -1:].values
```

Then, to improve the performance of the LSTM model, the input and target arrays are normalized using the `MinMaxScaler` class from `scikit-learn`:

```
sc_x = MinMaxScaler()
sc_y = MinMaxScaler()

x_scaled = sc_x.fit_transform(x_raw)
y_scaled = sc_y.fit_transform(y_raw)
```

After scaling, the `x_scaled` and `y_scaled` arrays are used as inputs and targets to train the LSTM model.

### LSTM Model

Defining a PyTorch LSTM model using the nn.Module class:

```
# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
```

### Training

Before training the LSTM model, we need to prepare the input data by creating input and target sequences with a given lookback value (i.e. "how far back" the model should look in the dataset while training). In this example, the lookback value is set to 6 points, so while predicting the model will take into account 6 previous values:

```
lookback = 6
x, y = [], []

for i in range(training_data.shape[0] - lookback):
    x.append(x_scaled[i:(i + lookback)])
    y.append(y_scaled[i + lookback])

x, y = np.array(x), np.array(y)
```

The x and y arrays are then converted to PyTorch tensors and split into training and testing sets:

```
lim = int(len(x) * 0.2)

x_train, y_train = torch.tensor(x[:-lim]), torch.tensor(y[:-lim])
x_test, y_test = torch.tensor(x[-lim:]), torch.tensor(y[-lim:])
```

Finally, we convert the x and y arrays to PyTorch tensors:

```
x, y = torch.tensor(x), torch.tensor(y)
```

These tensors will be used to train the LSTM model.

## Results

The result is shown as a figure with two subplots, each showing the true and predicted values of the time series. The first subplot displays the full dataset, while the second subplot displays only the test set. The y-axis represents the value of the time series, and the x-axis represents the index of each time step. The scaled predicted and true values were converted back to their original scales. The vertical line in the first subplot indicates the split between the training and test sets.

![Results](https://github.com/hovik23/lstm-model/blob/main/images/result.png)