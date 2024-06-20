# Copyright (C) 2024
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

import os
import random
import copy
import warnings

try:
    import numpy as np
except:
    print("ANN - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        print("ANN - numpy library installed correctly.")
    except:
        warnings.warn("ANN - Error: Could not import numpy.")

try:
    import pandas as pd
except:
    print("DGL - Installing required pandas library.")
    try:
        os.system("pip install pandas")
    except:
        os.system("pip install pandas --user")
    try:
        import pandas as pd
        print("ANN - pandas library installed correctly.")
    except:
        warnings.warn("ANN - Error: Could not import pandas.")

try:
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except:
    print("ANN - Installing required torch library.")
    try:
        os.system("pip install torch")
    except:
        os.system("pip install torch --user")
    try:
        import torch
        import torch.optim as optim
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        print("ANN - torch library installed correctly.")
    except:
        warnings.warn("ANN - Error: Could not import torch.")

try:
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, fetch_california_housing
except:
    print("ANN - Installing required scikit-learn library.")
    try:
        os.system("pip install -U scikit-learn")
    except:
        os.system("pip install -U scikit-learn --user")
    try:
        from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
        from sklearn.model_selection import KFold, train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, fetch_california_housing
        print("ANN - scikit-learn library installed correctly.")
    except:
        warnings.warn("ANN - Error: Could not import scikit. Please install it manually.")

class _ANNModel(nn.Module):
    def __init__(self,
                 inputSize=1,
                 outputSize=1,
                 taskType='classification',
                 validationRatio=0.2,
                 hiddenLayers=[12,12,12],
                 learningRate=0.001,
                 epochs=10,
                 activation="relu",
                 batchSize=1,
                 patience=4,
                 earlyStopping = True,
                 randomState = 42,
                 holdout=True,
                 kFolds=3,
                 ):
                 
        super(_ANNModel, self).__init__()
        
        # Initialize parameters
        self.hidden_layers = hiddenLayers
        self.output_size = outputSize
        self.activation = activation
        self.learning_rate = learningRate
        self.epochs = epochs
        self.validation_ratio = validationRatio
        self.holdout = holdout
        self.k_folds = kFolds
        self.batch_size = batchSize
        self.patience = patience
        self.early_stopping = earlyStopping
        self.random_state = randomState
        self.task_type = taskType

        self.training_loss_list = []
        self.validation_loss_list = []
        self.training_accuracy_list = []
        self.validation_accuracy_list = []
        self.training_mae_list = []
        self.validation_mae_list = []
        self.labels = []
        self.predictions = []
        
        
        # Define layers
        layers = []
        previous_size = inputSize
        
        # Create hidden layers
        for h in self.hidden_layers:
            layers.append(nn.Linear(previous_size, h))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")
            previous_size = h
        
        # Output layer
        layers.append(nn.Linear(previous_size, self.output_size))
        
        if self.task_type == 'classification':
            if self.output_size == 1:
                layers.append(nn.Sigmoid())  # Use Sigmoid for binary classification
            else:
                layers.append(nn.LogSoftmax(dim=1))  # Use LogSoftmax for multi-category classification
        elif self.task_type != 'regression':
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        self.model = nn.Sequential(*layers)
        
        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Define the loss function
        if self.task_type == 'classification':
            if self.output_size == 1:
                self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
            else:
                self.criterion = nn.NLLLoss()  # Negative Log Likelihood Loss for multi-category classification
        elif self.task_type == 'regression':
            self.criterion = nn.MSELoss()
    
    def forward(self, x):
        return self.model(x)
    
    def train(self, X, y):
        self.training_loss_list = []
        self.validation_loss_list = []
        self.training_accuracy_list = []
        self.validation_accuracy_list = []
        self.training_mae_list = []
        self.validation_mae_list = []
        if self.holdout == True or self.k_folds == 1:
            self._train_holdout(X, y)
        else:
            self._train_kfold(X, y)
    
    def _train_holdout(self, X, y):
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_ratio, random_state=self.random_state)
        
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        self._train_epochs(train_loader, val_loader)
    
    def _train_kfold(self, X, y):
        kf = KFold(n_splits=self.k_folds, shuffle=True)
        fold = 0
        total_loss = 0.0
        for train_idx, val_idx in kf.split(X):
            fold += 1
            print(f"Fold {fold}/{self.k_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            self._train_epochs(train_loader, val_loader)
    
    def _train_epochs(self, train_loader, val_loader):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                
                # Ensure labels have the same shape as outputs
                labels = labels.view(-1, 1) if outputs.shape[-1] == 1 else labels

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # Calculate training accuracy or MAE for regression
                if self.task_type == 'classification':
                    if outputs.shape[-1] > 1:
                        _, predicted = torch.max(outputs, 1)
                    else:
                        predicted = (outputs > 0.5).float()
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()
                elif self.task_type == 'regression':
                    correct_train += torch.abs(outputs - labels).sum().item()
                    total_train += labels.size(0)

            train_loss = running_loss / len(train_loader)
            if self.task_type == 'classification':
                train_accuracy = 100 * correct_train / total_train
            elif self.task_type == 'regression':
                train_accuracy = correct_train / total_train

            # Calculate validation loss and accuracy/MAE
            val_loss, val_accuracy = self.evaluate_loss(val_loader)
            self.training_loss_list.append(train_loss)
            self.validation_loss_list.append(val_loss)
            if self.task_type == 'classification':
                # print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
                #     f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
                self.training_accuracy_list.append(train_accuracy)
                self.validation_accuracy_list.append(val_accuracy)
            elif self.task_type == 'regression':
                # print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training MAE: {train_accuracy:.4f}, "
                #     f"Validation Loss: {val_loss:.4f}, Validation MAE: {val_accuracy:.4f}")
                self.training_mae_list.append(train_accuracy)
                self.validation_mae_list.append(val_accuracy)

            # Early stopping
            if self.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state = self.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        # print(f'Early stopping! Best validation loss: {best_val_loss}')
                        break
        # Update the epochs parameter to reflect the actual epochs ran.
        self.epochs = epoch + 1

        # Load the best model state
        if best_model_state:
            self.load_state_dict(best_model_state)
    
    def evaluate_loss(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self(inputs)
                labels = labels.view(-1, 1) if outputs.shape[-1] == 1 else labels

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Calculate validation accuracy or MAE for regression
                if self.task_type == 'classification':
                    if outputs.shape[-1] > 1:
                        _, predicted = torch.max(outputs, 1)
                    else:
                        predicted = (outputs > 0.5).float()
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                elif self.task_type == 'regression':
                    correct_val += torch.abs(outputs - labels).sum().item()
                    total_val += labels.size(0)
        
        avg_loss = total_loss / len(data_loader)
        if self.task_type == 'classification':
            accuracy = 100 * correct_val / total_val
        elif self.task_type == 'regression':
            accuracy = correct_val / total_val
        
        return avg_loss, accuracy


    def evaluate(self, X_test, y_test):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs).view(-1) if self.task_type == 'classification' and self.model[-1].__class__.__name__ == 'Sigmoid' else self(inputs)
                if self.task_type == 'classification':
                    if self.model[-1].__class__.__name__ == 'Sigmoid':
                        # Convert probabilities to binary predictions (0 or 1)
                        preds = [1 if x >= 0.5 else 0 for x in outputs.cpu().numpy()]
                    else:
                        # Get predicted class indices
                        _, preds = torch.max(outputs.data, 1)
                        preds = preds.cpu().numpy()
                elif self.task_type == 'regression':
                    preds = outputs.cpu().numpy()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
        self.labels = all_labels
        self.predictions = all_preds
        return all_labels, all_preds
    
    def metrics(self, labels, predictions):
        from sklearn import metrics
        if self.task_type == 'regression':
            metrics = {
                "Mean Squared Error": mean_squared_error(labels, predictions),
                "Mean Absolute Error": mean_absolute_error(labels, predictions),
                "R-squared": r2_score(labels, predictions)
            }
        elif self.task_type == 'classification':
            metrics = {
                "Accuracy": accuracy_score(labels, predictions),
                "Precision": precision_score(labels, predictions, average='weighted'),
                "Recall": recall_score(labels, predictions, average='weighted'),
                "F1 Score": f1_score(labels, predictions, average='weighted'),
                "Confusion Matrix": metrics.confusion_matrix(labels, predictions)
            }
        else:
            metrics = None
        return metrics
    
    def save(self, path):
        if path:
            # Make sure the file extension is .pt
            ext = path[len(path)-3:len(path)]
            if ext.lower() != ".pt":
                path = path+".pt"
            torch.save(self.state_dict(), path)
    
    def load(self, path):
        if path:
            self.load_state_dict(torch.load(path))


class ANN():
    @staticmethod
    def DatasetByCSVPath(path, taskType='classification', trainRatio=0.6, randomState=42):
        """
        Returns a dataset according to the input CSV file path.

        Parameters
        ----------
        path : str
            The path to the folder containing the necessary CSV and YML files.
        taskType : str , optional
            The type of evaluation task. This can be 'classification' or 'regression'. The default is 'classification'.
        trainRatio : float , optional
            The ratio of the data to use for training and validation vs. the ratio to use for testing. The default is 0.6
            which means that 60% of the data will be used for training and validation while 40% of the data will be reserved for testing.
        randomState : int , optional
            The randomState parameter is used to ensure reproducibility of the results. When you set the randomState parameter to a specific integer value,
            it controls the shuffling of the data before splitting it into training and testing sets.
            This means that every time you run your code with the same randomState value and the same dataset, you will get the same split of the data.
            The default is 42 which is just a randomly picked integer number. Specify None for random sampling.
        Returns
        -------
        list
            Returns the following list:
            X_train, X_test, y_train, y_test, taskType
            X_train is the list of features used for training
            X_test is the list of features used for testing
            y_train is the list of targets used for training
            y_test is the list of targets used for testing
            taskType is the type of task ('classification' or 'regression'). This is included for compatibility with DatasetBySample()

        """
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(path)

        # Assume the last column is the target
        features = df.iloc[:, :-1].values
        target = df.iloc[:, -1].values
        
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        y = target

        # Ensure target is in the correct format
        if taskType == 'classification' and len(np.unique(y)) == 2:
            y = y.reshape(-1, 1)  # Reshape for binary classification
        elif taskType == 'classification':
            y = y.astype(np.int64)  # Convert to long for multi-class classification
        
        y = y.astype(np.float32)  # Convert to float32 for PyTorch

        input_size = X.shape[1]  # Number of features
        num_classes = len(np.unique(y))
        output_size = 1 if taskType == 'regression' or num_classes == 2 else num_classes

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1.0 - trainRatio), random_state=randomState)

        return {'XTrain': X_train,
                'XTest': X_test,
                'yTrain': y_train,
                'yTest': y_test,
                'inputSize': input_size,
                'outputSize': output_size}
    
    @staticmethod
    def DatasetBySampleName(name, trainRatio=0.6, randomState=42):
        """
        Returns a dataset from the scikit-learn dataset samples.

        Parameters
        ----------
        name : str
            The  name of the dataset. This can be one of ['breast_cancer', 'california_housing', 'digits', 'iris', 'wine']

        trainRatio : float , optional
            The ratio of the data to use for training and validation vs. the ratio to use for testing. The default is 0.6
            which means that 60% of the data will be used for training and validation while 40% of the data will be reserved for testing.
        randomState : int , optional
            The randomState parameter is used to ensure reproducibility of the results. When you set the randomState parameter to a specific integer value,
            it controls the shuffling of the data before splitting it into training and testing sets.
            This means that every time you run your code with the same randomState value and the same dataset, you will get the same split of the data.
            The default is 42 which is just a randomly picked integer number. Specify None for random sampling.
        Returns
        -------
        list
            Returns the following list:
            X_train, X_test, y_train, y_test
            X_train is the list of features used for training
            X_test is the list of features used for testing
            y_train is the list of targets used for training
            y_test is the list of targets used for testing

        """
        from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, fetch_california_housing
        from sklearn.model_selection import train_test_split

        if name == 'breast_cancer':
            dataset = load_breast_cancer()
        elif name == 'california_housing':
            dataset = fetch_california_housing()
        elif name == 'digits':
            dataset = load_digits()
        elif name == 'iris':
            dataset = load_iris()
        elif name == 'wine':
            dataset = load_wine()
        else:
            print(f"ANN.DatasetBySampleName - Error: Unsupported dataset: {name}. Returning None.")
            return None

        # Standardize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(dataset.data)
        y = dataset.target

        task_type = ANN.HyperparametersBySampleDatasetName(name)['taskType']
        # For binary classification, ensure the target is in the correct format (1D tensor)
        if task_type == 'classification' and len(np.unique(y)) == 2:
            y = y.astype(np.float32)
        elif task_type == 'classification':
            y = y.astype(np.int64)

        input_size = X.shape[1]  # Number of features
        num_classes = len(np.unique(y))
        output_size = 1 if task_type == 'regression' or num_classes == 2 else num_classes

        # First split: train and temp (remaining)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1.0 - trainRatio), random_state=randomState)
        
        return {'XTrain': X_train,
                'XTest': X_test,
                'yTrain': y_train,
                'yTest': y_test,
                'inputSize': input_size,
                'outputSize': output_size}

    @staticmethod
    def DatasetSamplesNames():
        """
        Returns the names of the available sample datasets from sci-kit learn.

        Parameters
        ----------

        Returns
        ----------
        list
            The list of names of available sample datasets
        """
        return ['breast_cancer', 'california_housing', 'digits', 'iris', 'wine']
    
    @staticmethod
    def DatasetSplit(X, y, trainRatio=0.6, randomState=42):
        """
        Splits the input dataset according to the input ratios.

        Parameters
        ----------
        X : list
            The list of features.
        y : list
            The list of targets.
        trainRatio : float , optional
            The ratio of the data to use for training. The default is 0.6.
            This means that 60% of the data will be used for training and validation while 40% of the data will be reserved for testing.
        randomState : int , optional
            The randomState parameter is used to ensure reproducibility of the results. When you set the randomState parameter to a specific integer value,
            it controls the shuffling of the data before splitting it into training and testing sets.
            This means that every time you run your code with the same randomState value and the same dataset, you will get the same split of the data.
            The default is 42 which is just a randomly picked integer number. Specify None for random sampling.

        Returns
        -------
        list
            Returns the following list:
            X_train, X_test, y_train,y_test
            X_train is the list of features used for training
            X_test is the list of features used for testing
            y_train is the list of targets used for training
            y_test is the list of targets used for testing

        """
        # First split: train and temp (remaining)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1.0 - trainRatio), random_state=randomState)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def HyperparametersByInput(taskType='classification',
                               validationRatio=0.2,
                               hiddenLayers= [12,12,12],
                               learningRate = 0.001,
                               epochs = 10,
                               activation = 'relu',
                               batchSize = 1,
                               patience = 5,
                               earlyStopping = True,
                               randomState = 42,
                               holdout = True,
                               kFolds = 3):
        """
        taskType : str , optional
            The desired task type. This can be either 'classification' or 'regression' (case insensitive).
            Classification is a type of supervised learning where the model is trained to predict categorical labels (classes) from input data.
            Regression is a type of supervised learning where the model is trained to predict continuous numerical values from input data.
        validationRatio : float , optional
            The split ratio between training and validation. The default is 0.2. This means that
            80% of the data will be used for training and 20% will be used for validation.
        hiddenLayers : list , optional
            The number of hidden layers and the number of nodes in each layer.
            If you wish to have 3hidden layers with 8 nodes in the first
            16 nodes in the second, and 4 nodes in the last layer, you specify [8,16,4].
            The default is [12,12,12]
        learningRate : float, optional
            The desired learning rate. The default is 0.001. See https://en.wikipedia.org/wiki/Learning_rate
        epochs : int , optional
            The desired number of epochs. The default is 10. See https://en.wikipedia.org/wiki/Neural_network_(machine_learning)
        activation : str , optional
            The type of activation layer. See https://en.wikipedia.org/wiki/Activation_function
            Some common alternatives include:
            'relu' : ReLU (Rectified Linear Unit) is an activation function that outputs the input directly if it is positive; otherwise, it outputs zero.
            'sigmoid' : The sigmoid activation function, which maps inputs to a range between 0 and 1.
            'tanh' : The hyperbolic tangent activation function, which maps inputs to a range between -1 and 1.
            'leaky_relu': A variant of the ReLU that allows a small, non-zero gradient when the unit is not active.
            'elu' : Exponential Linear Unit, which improves learning characteristics by having a smooth curve.
            'swish' : An activation function defined as x . sigmoid(x)
            'softmax' : Often used in the output layer of a classification network, it normalizes the outputs to a probability distribution.
            'linear' : A linear activation function, which is often used in the output layer of regression networks.
            The default is 'relu'.
        batchSize : int , optional
            The desired number of samples that will be propagated through the network at one time before the model's internal parameters are updated. Instead of updating the model parameters after every single training sample
            (stochastic gradient descent) or after the entire training dataset (batch gradient descent), mini-batch gradient descent updates the model parameters after
            a specified number of samples, which is determined by batchSize. The default is 1.
        patience : int , optional
            The desired number of epochs with no improvement in the validation loss after which training will be stopped if early stopping is enabled.
        earlyStopping : bool , optional
            If set to True, the training will stop if the validation loss does not improve after a certain number of epochs defined by patience. The default is True.
        randomState : int , optional
            The randomState parameter is used to ensure reproducibility of the results. When you set the randomState parameter to a specific integer value,
            it controls the shuffling of the data before splitting it into training and testing sets.
            This means that every time you run your code with the same randomState value and the same dataset, you will get the same split of the data.
            The default is 42 which is just a randomly picked integer number. Specify None for random sampling.
        holdout : bool , optional
            If set to True, the Holdout cross-validation method is used. Otherwise, the K-fold method is used. The default is True.
        kFolds : int , optional
            The number of splits (folds) to use if K-Fold cross validation is selected. The default is 5.
        
        Returns
        -------
        dict
            Returns a dictionary with the following keys:
            'taskType'
            'validationRatio'
            'hiddenLayers'
            'learningRate'
            'epochs'
            'activation'
            'batchSize'
            'patience'
            'earlyStopping'
            'randomState'
            'holdout'
            'kFolds'
        """
        return {
                'taskType': taskType,
                'validationRatio': validationRatio,
                'hiddenLayers': hiddenLayers,
                'learningRate': learningRate,
                'epochs': epochs,
                'activation': activation,
                'batchSize': batchSize,
                'patience': patience,
                'earlyStopping': earlyStopping,
                'randomState': randomState,
                'holdout': holdout,
                'kFolds': kFolds }
        
    @staticmethod
    def HyperparametersBySampleDatasetName(name):
        """
        Returns the suggested initial hyperparameters to use for the dataset named in the name input parameter.
        You can get a list of available sample datasets using ANN.SampleDatasets().

        Parameters
        ----------
        name : str
            The input name of the sample dataset. This must be one of ['breast_cancer', 'california_housing', 'digits', 'iris', 'wine']

        Returns
        -------
        dict
            Returns a dictionary with the following keys:
            'taskType'
            'validationRatio'
            'hiddenLayers'
            'learningRate'
            'epochs'
            'activation'
            'batchSize'
            'patience'
            'earlyStopping'
            'randomState'
            'holdout'
            'kFolds'

        """
        hyperparameters = {
            'breast_cancer': {
                'taskType': 'classification',
                'validationRatio': 0.2,
                'hiddenLayers': [30, 15],
                'learningRate': 0.001,
                'epochs': 100,
                'activation': 'relu',
                'batchSize': 32,
                'patience': 10,
                'earlyStopping': True,
                'randomState': 42,
                'holdout': True,
                'kFolds': 3
            },
            'california_housing': {
                'taskType': 'regression',
                'validationRatio': 0.2,
                'hiddenLayers': [50, 25],
                'learningRate': 0.001,
                'epochs': 150,
                'activation': 'relu',
                'batchSize': 32,
                'patience': 10,
                'earlyStopping': True,
                'randomState': 42,
                'holdout': True,
                'kFolds': 3
            },
            'digits': {
                'taskType': 'classification',
                'validationRatio': 0.2,
                'hiddenLayers': [64, 32],
                'learningRate': 0.001,
                'epochs': 50,
                'activation': 'relu',
                'batchSize': 32,
                'patience': 10,
                'earlyStopping': True,
                'randomState': 42,
                'holdout': True,
                'kFolds': 3
            },
            'iris': {
                'taskType': 'classification',
                'validationRatio': 0.2,
                'hiddenLayers': [10, 5],
                'learningRate': 0.001,
                'epochs': 100,
                'activation': 'relu',
                'batchSize': 16,
                'patience': 10,
                'earlyStopping': True,
                'randomState': 42,
                'holdout': True,
                'kFolds': 3
            },
            'wine': {
                'taskType': 'classification',
                'validationRatio': 0.2,
                'hiddenLayers': [50, 25],
                'learningRate': 0.001,
                'epochs': 100,
                'activation': 'relu',
                'batchSize': 16,
                'patience': 10,
                'earlyStopping': True,
                'randomState': 42,
                'holdout': True,
                'kFolds': 3
            }
        }
        
        if name in hyperparameters:
            return hyperparameters[name]
        else:
            print(f"ANN-HyperparametersBySampleDatasetName - Error: Dataset name '{name}' not recognized. Available datasets: {list(hyperparameters.keys())}. Returning None.")
            return None

    @staticmethod
    def ModelData(model):
        """
        Returns the data of the model

        Parameters
        ----------
        model : Model
            The input model.

        Returns
        -------
        dict
            A dictionary containing the model data. The keys in the dictionary are:
            'epochs'
            'trainingLoss'
            'validationLoss'
            'trainingAccuracy' (for classification tasks only)
            'validationAccuracy' (for classification tasks only)
            'trainingMAE' (for regression tasks only)
            'validationMAE' (for regression tasks only)

        """

        return {
                ' epochs': model.epochs,
                'trainingLoss': model.training_loss_list,
                'validationLoss': model.validation_loss_list,
                'trainingAccuracy': model.training_accuracy_list,
                'validationAccuracy': model.validation_accuracy_list,
                'trainingMAE': model.training_mae_list,
                'validationMAE': model.validation_mae_list
            }

    @staticmethod
    def ModelInitialize(inputSize, outputSize, hyperparameters = None):
        """
        Initializes an ANN model given the input parameter.

        Parameters
        ----------
        inputSize : int
            The number of initial inputs. This is usually computed directly from the dataset.
        outputSize : int
            The number of categories for classification tasks. This is usually computed directly from the dataset.
        hyperparameters : dict
            The hyperparameters dictionary. You can create one using ANN.HyperparametersByInput or, if you are using a sample Dataset, you can get it from ANN.HyperParametersBySampleDatasetName.
       
        Returns
        -------
        _ANNModel
            Returns the trained model.

        """
        
        task_type = hyperparameters['taskType']
        validation_ratio = hyperparameters['validationRatio']
        hidden_layers = hyperparameters['hiddenLayers']
        learning_rate = hyperparameters['learningRate']
        epochs = hyperparameters['epochs']
        activation = hyperparameters['activation']
        batch_size = hyperparameters['batchSize']
        patience = hyperparameters['patience']
        early_stopping = hyperparameters['earlyStopping']
        random_state = hyperparameters['randomState']
        holdout = hyperparameters['holdout']
        k_folds = hyperparameters['kFolds']

        task_type = task_type.lower()
        if task_type not in ['classification', 'regression']:
            print("ANN.ModelInitialize - Error: The input parameter taskType is not recognized. It must be either 'classification' or 'regression'. Returning None.")
            return None
        
        model = _ANNModel(inputSize=inputSize,
                    outputSize=outputSize,
                    taskType=task_type,
                    validationRatio=validation_ratio,
                    hiddenLayers=hidden_layers,
                    learningRate=learning_rate,
                    epochs=epochs,
                    activation=activation,
                    batchSize=batch_size,
                    patience=patience,
                    earlyStopping = early_stopping,
                    randomState = random_state,
                    holdout=holdout,
                    kFolds=k_folds
                    )
        return model

    @staticmethod
    def ModelTrain(model, X, y):
        """
        Trains the input model given the input features (X), and target (y).

        Parameters
        ----------
        model : ANN Model
            The input model.
        X : list
            The input list of features.
        y : list
            The input list of targets
       
        Returns
        -------
        _ANNModel
            Returns the trained model.

        """
        model.train(X, y)
        return model

    @staticmethod
    def ModelEvaluate(model, X, y):
        """
        Returns the labels (actual values) and predictions (predicted values) given the input model, features (X), and target (y).

        Parameters
        ----------
        model : ANN Model
            The input model.
        X : list
            The input list of features.
        y : list
            The input list of targets
       
        Returns
        -------
        list, list
            Returns two lists: labels, and predictions.

        """
        labels, predictions = model.evaluate(X, y)
        return labels, predictions

    @staticmethod
    def ModelFigures(model, width=900, height=600, template="plotly", colorScale='viridis', colorSamples=10):
        """
        Creates Plotly Figures from the model data. For classification tasks this includes
        a confusion matrix, loss, and accuracy figures. For regression tasks this includes
        loss and MAE figures.

        Parameters
        ----------
        model : ANN Model
            The input model.
        width : int , optional
            The desired figure width in pixels. The default is 900.
        height : int , optional
            The desired figure height in pixels. The default is 900.
        template : str , optional
            The desired Plotly template to use for the scatter plot.
            This can be one of ['ggplot2', 'seaborn', 'simple_white', 'plotly',
            'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
            'ygridoff', 'gridon', 'none']. The default is "plotly".
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "viridis", "plasma"). The default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        colorSamples : int , optional
            The number of discrete color samples to use for displaying the data. The default is 10.
       
        Returns
        -------
        list
            Returns a list of Plotly figures and a corresponding list of file names.

        """
        import plotly.graph_objects as go
        from topologicpy.Plotly import Plotly
        figures = []
        filenames = []
        if model.task_type.lower() == 'classification':
            data_lists = [[model.training_loss_list, model.validation_loss_list], [model.training_accuracy_list, model.validation_accuracy_list]]
            label_lists = [['Training Loss', 'Validation Loss'], ['Training Accuracy', 'Validation Accuracy']]
            titles = ['Training and Validation Loss', 'Training and Validation Accuracy']
            legend_titles = ['Loss Type', 'Accuracy Type']
            xaxis_titles = ['Epoch', 'Epoch']
            yaxis_titles = ['Loss', 'Accuracy']
            filenames = yaxis_titles
            if len(model.labels) > 0 and len(model.labels) == len(model.predictions):
                confusion_matrix = ANN.ModelMetrics(model, labels = model.labels, predictions = model.predictions)['Confusion Matrix']
                confusion_matrix_figure = Plotly.FigureByConfusionMatrix(confusion_matrix, width=width, height=height, colorScale=colorScale, colorSamples=colorSamples)
                figures.append(confusion_matrix_figure)
                filenames.append("ConfusionMatrix")
        elif model.task_type.lower() == 'regression':
            data_lists = [[model.training_loss_list, model.validation_loss_list], [model.training_mae_list, model.validation_mae_list]]
            label_lists = [['Training Loss', 'Validation Loss'], ['Training MAE', 'Validation MAE']]
            titles = ['Training and Validation Loss', 'Training and Validation MAE']
            legend_titles = ['Loss Type', 'MAE Type']
            xaxis_titles = ['Epoch', 'Epoch']
            yaxis_titles = ['Loss', 'MAE']
            filenames = yaxis_titles
        else:
            print("ANN.ModelFigures - Error: Could not recognize model task type. Returning None.")
            return None
        for i in range(2):
            data = data_lists[i]
            labels = label_lists[i]
            title = titles[i]
            legend_title = legend_titles[i]
            xaxis_title = xaxis_titles[i]
            yaxis_title = yaxis_titles[i]
            lengths = [len(d) for d in data]

            max_length = max(lengths)
            x_ticks = list(range(1, max_length + 1))

            figure = go.Figure()
            for j in range(len(data)):
                figure.add_trace(go.Scatter(x=x_ticks, y=data[j], mode='lines+markers', name=labels[j]))


            figure.update_layout(
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                legend_title= legend_title,
                template=template,
                width=width,
                height=height
            )
            figures.append(figure)
        return figures, filenames
    
    @staticmethod
    def ModelMetrics(model, labels, predictions):
        """
        Returns the model performance metrics given the input labels and predictions, and the model's task type.

        Parameters
        ----------
        model : ANN Model
            The input model.
        labels : list
            The input list of labels (actual values).
        predictions : list
            The input list of predictions (predicted values).
       
        Returns
        -------
        dict
            if the task type is 'classification', this methods return a dictionary with the following keys:
                "Accuracy"
                "Precision"
                "Recall"
                "F1 Score"
                "Confusion Matrix"
            else if the task type is 'regression', this method returns:
                "Mean Squared Error"
                "Mean Absolute Error"
                "R-squared"

        """
        metrics = model.metrics(labels, predictions)
        return metrics

    @staticmethod
    def ModelSave(model, path, overwrite=False):
        """
        Saves the model.

        Parameters
        ----------
        model : Model
            The input model.
        path : str
            The file path at which to save the model.
        overwrite : bool, optional
            If set to True, any existing file will be overwritten. Otherwise, it won't. The default is False.

        Returns
        -------
        bool
            True if the model is saved correctly. False otherwise.

        """
        import os

        if model == None:
            print("DGL.ModelSave - Error: The input model parameter is invalid. Returning None.")
            return None
        if path == None:
            print("DGL.ModelSave - Error: The input path parameter is invalid. Returning None.")
            return None
        if not overwrite and os.path.exists(path):
            print("DGL.ModelSave - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        if overwrite and os.path.exists(path):
            os.remove(path)
        # Make sure the file extension is .pt
        ext = path[len(path)-3:len(path)]
        if ext.lower() != ".pt":
            path = path+".pt"
        # Save the trained model
        torch.save(model.state_dict(), path)
        return True

    @staticmethod
    def ModelLoad(model, path):
        """
        Loads the model state dictionary found at the input file path. The model input parameter must be pre-initialized using the ModelInitialize method.

        Parameters
        ----------
        model : ANN object
            The input ANN model. The model must be pre-initialized using the ModelInitialize method.
        path : str
            File path for the saved model state dictionary.

        Returns
        -------
        ANN model
            The neural network class.

        """
        from os.path import exists
        
        if not exists(path):
            print("ANN.ModelLoad - Error: The specified path does not exist. Returning None.")
            return None
        model.load(path)
        return model

    @staticmethod
    def ConfusionMatrix(actual, predicted, normalize=False):
        """
        Returns the confusion matrix for the input actual and predicted labels. This is to be used with classification tasks only not regression.

        Parameters
        ----------
        actual : list
            The input list of actual labels.
        predicted : list
            The input list of predicts labels.
        normalized : bool , optional
            If set to True, the returned data will be normalized (proportion of 1). Otherwise, actual numbers are returned. The default is False.

        Returns
        -------
        list
            The created confusion matrix.

        """
        import os
        import warnings

        try:
            from sklearn import metrics
            from sklearn.metrics import accuracy_score
        except:
            print("ANN.ConfusionMatrix - Installing required scikit-learn (sklearn) library.")
            try:
                os.system("pip install scikit-learn")
            except:
                os.system("pip install scikit-learn --user")
            try:
                from sklearn import metrics
                from sklearn.metrics import accuracy_score
                print("ANN.ConfusionMatrix - scikit-learn (sklearn) library installed correctly.")
            except:
                warnings.warn("ANN.ConfusionMatrix - Error: Could not import scikit-learn (sklearn). Please try to install scikit-learn manually. Returning None.")
                return None

        if not isinstance(actual, list):
            print("ANN.ConfusionMatrix - ERROR: The actual input is not a list. Returning None")
            return None
        if not isinstance(predicted, list):
            print("ANN.ConfusionMatrix - ERROR: The predicted input is not a list. Returning None")
            return None
        if len(actual) != len(predicted):
            print("ANN.ConfusionMatrix - ERROR: The two input lists do not have the same length. Returning None")
            return None
        if normalize:
            cm = np.transpose(metrics.confusion_matrix(y_true=actual, y_pred=predicted, normalize="true"))
        else:
            cm = np.transpose(metrics.confusion_matrix(y_true=actual, y_pred=predicted))
        return cm

    @staticmethod
    def Show(data,
             labels,
             title="Training/Validation",
             xTitle="Epochs",
             xSpacing=1,
             yTitle="Accuracy and Loss",
             ySpacing=0.1,
             useMarkers=False,
             chartType="Line",
             width=950,
             height=500,
             backgroundColor='rgba(0,0,0,0)',
             gridColor='lightgray',
             marginLeft=0,
             marginRight=0,
             marginTop=40,
             marginBottom=0,
             renderer = "notebook"):
        """
        Shows the data in a plolty graph.

        Parameters
        ----------
        data : list
            The data to display.
        labels : list
            The labels to use for the data.
        width : int , optional
            The desired width of the figure. The default is 950.
        height : int , optional
            The desired height of the figure. The default is 500.
        title : str , optional
            The chart title. The default is "Training and Testing Results".
        xTitle : str , optional
            The X-axis title. The default is "Epochs".
        xSpacing : float , optional
            The X-axis spacing. The default is 1.0.
        yTitle : str , optional
            The Y-axis title. The default is "Accuracy and Loss".
        ySpacing : float , optional
            The Y-axis spacing. The default is 0.1.
        useMarkers : bool , optional
            If set to True, markers will be displayed. The default is False.
        chartType : str , optional
            The desired type of chart. The options are "Line", "Bar", or "Scatter". It is case insensitive. The default is "Line".
        backgroundColor : str , optional
            The desired background color. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        gridColor : str , optional
            The desired grid color. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'lightgray'.
        marginLeft : int , optional
            The desired left margin in pixels. The default is 0.
        marginRight : int , optional
            The desired right margin in pixels. The default is 0.
        marginTop : int , optional
            The desired top margin in pixels. The default is 40.
        marginBottom : int , optional
            The desired bottom margin in pixels. The default is 0.
        renderer : str , optional
            The desired plotly renderer. The default is "notebook".

        Returns
        -------
        None.

        """
        from topologicpy.Plotly import Plotly

        dataFrame = Plotly.DataByDGL(data, labels)
        fig = Plotly.FigureByDataFrame(dataFrame,
                                       labels=labels,
                                       title=title,
                                       xTitle=xTitle,
                                       xSpacing=xSpacing,
                                       yTitle=yTitle,
                                       ySpacing=ySpacing,
                                       useMarkers=useMarkers,
                                       chartType=chartType,
                                       width=width,
                                       height=height,
                                       backgroundColor=backgroundColor,
                                       gridColor=gridColor,
                                       marginRight=marginRight,
                                       marginLeft=marginLeft,
                                       marginTop=marginTop,
                                       marginBottom=marginBottom
                                       )
        Plotly.Show(fig, renderer=renderer)
    
    