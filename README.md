## Stock Prediction(TF and Keras)
Stock Prediction for Uber stocks (data taken from Yahoo stocks) for 30 days and 1 year. 

1. Libraries used: Tensorflow, Keras, Pandas, Numpy, Matplotlib, sklearn. 
2. Optimizer = Adam
3. Layers = LSTM, Dropout, Dense
4. Evaulation metrics = accuracy, MSE, RMSE


## Reasoning: 
1. LSTM: LSTMs are suited for time series forecasting. They can remember long input sequences as they have memory gates. These help in capturing trends and patterns.
2. Number of Neurons: Each LSTM layer has 50 neurons. This number allows the model to capture complex patterns without being too computationally expensive.
3. Dropout Layers: To prevent overfitting by randomly dropping neurons.
4. Number of Epochs: 50 epochs is selected. This number provides a balance in order to avoid underfitting and overfitting.
5. Optimizer: Adam is a robust optimizer which provides fast convergence, and is a reliable choice for different deep learning models.
6. Batch Size: 32 is selected. This is to prevent overfitting and to provide better regularization.
7. Activation Functions: Tanh and ReLu - They bring non-linearities into the model. This makes the model computationally efficient and effective by avoiding vanishing gradient problem.
8. Validation Methodology: The dataset is split into 80% for training and 20% for testing, which is a standard practice in machine learning. This methodology helps in avoiding overfitting and to validate that the model can generalize well.
