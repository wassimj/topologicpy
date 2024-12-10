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

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class _ANN(nn.Module):
    def __init__(self, input_size, hyperparameters, dataset=None):
        super(_ANN, self).__init__()
        self.title =  hyperparameters.get('title', 'Untitled')
        self.task_type = hyperparameters['task_type']
        self.cross_val_type = hyperparameters['cross_val_type']
        self.k_folds = hyperparameters.get('k_folds', 5)
        self.test_size = hyperparameters.get('test_size', 0.3)
        self.validation_ratio = hyperparameters.get('validation_ratio', 0.1)
        self.random_state = hyperparameters.get('random_state', 42)
        self.batch_size = hyperparameters.get('batch_size', 32)
        self.learning_rate = hyperparameters.get('learning_rate', 0.001)
        self.epochs = hyperparameters.get('epochs', 100)
        self.early_stopping = hyperparameters.get('early_stopping', False)
        self.patience = hyperparameters.get('patience', 10)
        self.interval = hyperparameters.get('interval',1)
        self.mantissa = hyperparameters.get('mantissa', 4)
        
        self.train_loss_list = []
        self.val_loss_list = []
        
        self.train_accuracy_list = []
        self.val_accuracy_list = []
        
        self.train_mse_list = []
        self.val_mse_list = []
        
        self.train_mae_list = []
        self.val_mae_list = []
        
        self.train_r2_list = []
        self.val_r2_list = []
        self.epoch_list = []

        self.metrics = {}
        
        layers = []
        hidden_layers = hyperparameters['hidden_layers']
        
        # Compute output_size based on task type and dataset
        if self.task_type == 'regression':
            output_size = 1
        elif self.task_type == 'binary_classification':
            output_size = 1
        elif self.task_type == 'classification' and dataset is not None:
            output_size = len(np.unique(dataset.target))
        else:
            print("ANN - Error: Invalid task type or dataset not provided for classification. Returning None.")
            return None
        
        # Create hidden layers
        in_features = input_size
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            in_features = hidden_units
        
        # Output layer
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)
        
        # Loss function based on task type
        if self.task_type == 'regression':
            self.loss_fn = nn.MSELoss()
        elif self.task_type == 'binary_classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:  # multi-category classification
            self.loss_fn = nn.CrossEntropyLoss()
        
       
        
        # Initialize best model variables
        self.best_model_state = None
        self.best_val_loss = np.inf
    
    def forward(self, x):
        return self.model(x)

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        self.train_loss_list = []
        self.val_loss_list = []
        
        self.train_accuracy_list = []
        self.val_accuracy_list = []
        
        self.train_mse_list = []
        self.val_mse_list = []
        
        self.train_mae_list = []
        self.val_mae_list = []
        
        self.train_r2_list = []
        self.val_r2_list = []
        self.epoch_list = []
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # Reinitialize optimizer for each fold
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        current_patience = self.patience if self.early_stopping else self.epochs

        # Convert to DataLoader for batching
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)

                if self.task_type == 'binary_classification':
                    outputs = outputs.squeeze()
                    targets = targets.float()
                elif self.task_type == 'regression':
                    outputs = outputs.squeeze()

                loss = self.loss_fn(outputs, targets)
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                # Calculate metrics for training set
                if self.task_type == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    correct_train += (predicted == targets).sum().item()
                    total_train += targets.size(0)
                elif self.task_type == 'binary_classification':
                    predicted = torch.round(torch.sigmoid(outputs))
                    correct_train += (predicted == targets).sum().item()
                    total_train += targets.size(0)

            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self(X_val)
                    if self.task_type == 'binary_classification':
                        val_outputs = val_outputs.squeeze()
                        y_val = y_val.float()
                    elif self.task_type == 'regression':
                        val_outputs = val_outputs.squeeze()

                    val_loss = self.loss_fn(val_outputs, y_val)
                    val_loss_item = val_loss.item()

                    # Track the best model state
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_model_state = self.state_dict()
                        current_patience = self.patience if self.early_stopping else self.epochs
                    else:
                        if self.early_stopping:
                            current_patience -= 1

                    if self.early_stopping and current_patience == 0:
                        print(f'ANN - Information: Early stopping after epoch {epoch + 1}')
                        break

            if (epoch + 1) % self.interval == 0:
                self.epoch_list.append(epoch + 1)
                avg_epoch_loss = epoch_loss / len(train_loader)
                self.train_loss_list.append(round(avg_epoch_loss, self.mantissa))

                if self.task_type == 'classification' or self.task_type == 'binary_classification':
                    train_accuracy = round(correct_train / total_train, self.mantissa)
                    self.train_accuracy_list.append(train_accuracy)
                    if X_val is not None and y_val is not None:
                        val_accuracy = (torch.round(torch.sigmoid(val_outputs)) if self.task_type == 'binary_classification' else torch.max(val_outputs, 1)[1] == y_val).float().mean().item()
                        val_accuracy = round(val_accuracy, self.mantissa)
                        self.val_accuracy_list.append(val_accuracy)
                elif self.task_type == 'regression':
                    train_preds = self(X_train).detach().numpy().squeeze()
                    train_mse = round(mean_squared_error(y_train.numpy(), train_preds), self.mantissa)
                    train_mae = round(mean_absolute_error(y_train.numpy(), train_preds), self.mantissa)
                    train_r2 = round(r2_score(y_train.numpy(), train_preds), self.mantissa)
                    self.train_mse_list.append(train_mse)
                    self.train_mae_list.append(train_mae)
                    self.train_r2_list.append(train_r2)
                    if X_val is not None and y_val is not None:
                        val_preds = val_outputs.numpy().squeeze()
                        val_mse = round(mean_squared_error(y_val.numpy(), val_preds), self.mantissa)
                        val_mae = round(mean_absolute_error(y_val.numpy(), val_preds), self.mantissa)
                        val_r2 = round(r2_score(y_val.numpy(), val_preds), self.mantissa)
                        self.val_mse_list.append(val_mse)
                        self.val_mae_list.append(val_mae)
                        self.val_r2_list.append(val_r2)

                if X_val is not None and y_val is not None:
                    self.val_loss_list.append(round(val_loss_item, self.mantissa))
                    print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss_item:.4f}')
                else:
                    print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}')
    
    def evaluate_model(self, X_test, y_test):
        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            
            if self.task_type == 'regression':
                outputs = outputs.squeeze()
                predictions = outputs.numpy()
                mse = mean_squared_error(y_test.numpy(), outputs.numpy())
                mae = mean_absolute_error(y_test.numpy(), outputs.numpy())
                r2 = r2_score(y_test.numpy(), outputs.numpy())
                #print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}')
                metrics = {'mae': round(mae, self.mantissa), 'mse': round(mse, self.mantissa), 'r2': round(r2, self.mantissa)}
            elif self.task_type == 'binary_classification':
                outputs = torch.sigmoid(outputs).squeeze()
                predicted = (outputs > 0.5).int()
                predictions = predicted.numpy()
                accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
                precision = precision_score(y_test.numpy(), predicted.numpy(), zero_division=0)
                recall = recall_score(y_test.numpy(), predicted.numpy(), zero_division=0)
                f1 = f1_score(y_test.numpy(), predicted.numpy(), zero_division=0)
                #print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
                cm = self.confusion_matrix(y_test, predictions)
                metrics = {'accuracy': round(accuracy, self.mantissa), 'precision': round(precision, self.mantissa), 'recall': round(recall, self.mantissa), 'f1': round(f1, self.mantissa), 'confusion_matrix': cm}
            else:  # multi-category classification
                _, predicted = torch.max(outputs, 1)            
                predictions = predicted.numpy()
                accuracy = accuracy_score(y_test.numpy(), predictions)
                precision = precision_score(y_test.numpy(), predictions, average='macro', zero_division=0)
                recall = recall_score(y_test.numpy(), predictions, average='macro', zero_division=0)
                f1 = f1_score(y_test.numpy(), predictions, average='macro', zero_division=0)
                cm = self.confusion_matrix(y_test, predicted.numpy())
                #print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
                metrics = {'accuracy': round(accuracy, self.mantissa), 'precision': round(precision, self.mantissa), 'recall': round(recall, self.mantissa), 'f1': round(f1, self.mantissa), 'confusion_matrix': cm}
            self.metrics = metrics
            
            return metrics, predictions
    
    def confusion_matrix(self, y_test, predictions):
        if self.task_type != 'regression':
            cm = confusion_matrix(y_test.numpy(), predictions)
            return cm.tolist()
        else:
            print("ANN - Error: Confusion matrix is not applicable for regression tasks. Returning None")
            return None
    
    def reset_parameters(self):
        for layer in self.model:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def cross_validate(self, X, y):
        if 'hold' in self.cross_val_type:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.validation_ratio, random_state=self.random_state)
            self.train_model(X_train, y_train, X_val=X_val, y_val=y_val)
            metrics, predictions = self.evaluate_model(X_test, y_test)
            if self.task_type != 'regression':
                self.confusion_matrix(y_test, predictions)
            return metrics
        
        elif 'fold' in self.cross_val_type:
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
            best_fold_index = -1
            best_val_loss = np.inf
            best_model_state = None
            
            for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
                # Reinitialize model parameters
                self.reset_parameters()
                print("Fold:", fold_idx+1)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.validation_ratio, random_state=self.random_state)
                
                self.train_model(X_train, y_train, X_val=X_val, y_val=y_val)
                
                print(f'Self Best Val Loss: {self.best_val_loss.item():.4f}')
                if self.best_val_loss < best_val_loss:
                    best_val_loss = self.best_val_loss
                    best_fold_index = fold_idx
                    best_model_state = self.best_model_state
            
            if best_fold_index == -1:
                raise ValueError("No best fold found. Check early stopping and validation handling.")
            
            print(f'Selecting best fold: {best_fold_index + 1}')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.validation_ratio, random_state=self.random_state)
            self.load_state_dict(best_model_state)
            #print("Training on Best fold.")
            #self.train_model(X_train, y_train, X_val=X_val, y_val=y_val)
            
            metrics, predictions = self.evaluate_model(X_val, y_val)
            if self.task_type != 'regression':
                self.confusion_matrix(y_val, predictions)
            
            return metrics
    
    def save(self, path):
        if path:
            ext = path[-3:]
            if ext.lower() != ".pt":
                path = path + ".pt"
            torch.save(self.state_dict(), path)
    
    def load(self, path):
        if path:
            self.load_state_dict(torch.load(path))

class ANN():
    @staticmethod
    def DatasetByCSVPath(path, taskType='classification', description=""):
        """
        Returns a dataset according to the input CSV file path.

        Parameters
        ----------
        path : str
            The path to the folder containing the necessary CSV and YML files.
        taskType : str , optional
            The type of evaluation task. This can be 'classification' or 'regression'. The default is 'classification'.
        description : str , optional
            The description of the dataset. In keeping with the scikit BUNCH class, this will be saved in the DESCR parameter.
        
        Returns
        -------
        sklearn.utils._bunch.Bunch
            The created dataset.

        """
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.utils import Bunch

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(path)

        # Assume the last column is the target
        features = df.iloc[:, :-1].values
        target = df.iloc[:, -1].values

        # Set target_names based on the name of the target column
        target_names = [df.columns[-1]]

        # Create a Bunch object
        dataset = Bunch(
            data=features,
            target=target,
            feature_names=df.columns[:-1].tolist(),
            target_names=target_names,
            frame=df,
            DESCR=description,
        )
        return dataset

    @staticmethod
    def DatasetBySampleName(name):
        """
        Returns a dataset from the scikit-learn dataset samples.

        Parameters
        ----------
        name : str
            The  name of the dataset. This can be one of ['breast_cancer', 'california_housing', 'digits', 'iris', 'wine']
        
        Returns
        -------
        sklearn.utils._bunch.Bunch
            The created dataset.
        """
        # Load dataset
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
        return dataset
        
    @staticmethod
    def DatasetSampleNames():
        """
        Returns the names of the available sample datasets from sci-kit learn.

        Parameters
        ----------

        Returns
        -------
        list
            The list of names of available sample datasets
        """
        return ['breast_cancer', 'california_housing', 'digits', 'iris', 'wine']
    
    @staticmethod
    def DatasetSplit(X, y, testRatio=0.3, randomState=42):
        """
        Splits the input dataset according to the input ratios.

        Parameters
        ----------
        X : list
            The list of features.
        y : list
            The list of targets.
        testRatio : float , optional
            The ratio of the dataset to reserve as unseen data for testing. The default is 0.3
        randomState : int , optional
            The randomState parameter is used to ensure reproducibility of the results. When you set the randomState parameter to a specific integer value,
            it controls the shuffling of the data before splitting it into training and testing sets.
            This means that every time you run your code with the same randomState value and the same dataset, you will get the same split of the data.
            The default is 42 which is just a randomly picked integer number. Specify None for random sampling.

        Returns
        -------
        list
            Returns the following list : [X_train, X_test, y_train,y_test]
            X_train is the list of features used for training
            X_test is the list of features used for testing
            y_train is the list of targets used for training
            y_test is the list of targets used for testing

        """
        if testRatio < 0 or testRatio > 1:
            print("ANN.DatasetSplit - Error: testRatio parameter cannot be outside the range [0,1]. Returning None.")
            return None
        # First split: train and temp (remaining)
        return train_test_split(X, y, test_size=testRatio, random_state=randomState)

    @staticmethod
    def Hyperparameters(title='Untitled',
                        taskType='classification',
                        testRatio = 0.3,
                        validationRatio = 0.2,
                        hiddenLayers = [12,12,12],
                        learningRate = 0.001,
                        epochs = 10,
                        batchSize = 1,
                        patience = 5,
                        earlyStopping = True,
                        randomState = 42,
                        crossValidationType = "holdout",
                        kFolds = 3,
                        interval = 1,
                        mantissa = 6):
        """
        Returns a Hyperparameters dictionary based on the input parameters.

        Parameters
        ----------
        title : str , optional
            The desired title for the dataset. The default is "Untitled".
        taskType : str , optional
            The desired task type. This can be either 'classification' or 'regression' (case insensitive).
            Classification is a type of supervised learning where the model is trained to predict categorical labels (classes) from input data.
            Regression is a type of supervised learning where the model is trained to predict continuous numerical values from input data.
        testRatio : float , optional
            The split ratio between training and testing. The default is 0.3. This means that
            70% of the data will be used for training/validation and 30% will be reserved for testing as unseen data.
        validationRatio : float , optional
            The split ratio between training and validation. The default is 0.2. This means that
            80% of the validation data (left over after reserving test data) will be used for training and 20% will be used for validation.
        hiddenLayers : list , optional
            The number of hidden layers and the number of nodes in each layer.
            If you wish to have 3hidden layers with 8 nodes in the first
            16 nodes in the second, and 4 nodes in the last layer, you specify [8,16,4].
            The default is [12,12,12]
        learningRate : float, optional
            The desired learning rate. The default is 0.001. See https://en.wikipedia.org/wiki/Learning_rate
        epochs : int , optional
            The desired number of epochs. The default is 10. See https://en.wikipedia.org/wiki/Neural_network_(machine_learning)
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
        crossValidationType : str , optional
            The desired type of cross-validation. This can be one of 'holdout' or 'k-fold'. The default is 'holdout'
        kFolds : int , optional
            The number of splits (folds) to use if K-Fold cross validation is selected. The default is 5.
        interval : int , optional
            The desired epoch interval at which to report and save metrics data. This must be less than the total number of epochs. The default is 1.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
        Returns
        -------
        dict
            Returns a dictionary with the following keys:
            'title'
            'task_type'
            'test_ratio'
            'validation_ratio'
            'hidden_layers'
            'learning_rate'
            'epochs'
            'batch_size'
            'early_stopping'
            'patience'
            'random_state'
            'cross_val_type'
            'kFolds'
            'interval'
            'mantissa'
        """
        return {
                'title': title,
                'task_type': taskType,
                'test_ratio': testRatio,
                'validation_ratio': validationRatio,
                'hidden_layers': hiddenLayers,
                'learning_rate': learningRate,
                'epochs': epochs,
                'batch_size': batchSize,
                'early_stopping': earlyStopping,
                'patience': patience,
                'random_state': randomState,
                'cross_val_type': crossValidationType,
                'k_folds': kFolds,
                'interval': interval,
                'mantissa': mantissa}
        
    @staticmethod
    def HyperparametersBySampleName(name):
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
            'title'
            'task_type'
            'test_ratio'
            'validation_ratio'
            'hidden_layers'
            'learning_rate'
            'epochs'
            'batch_size'
            'early_stopping'
            'patience'
            'random_state'
            'cross_val_type'
            'k_folds'
            'interval'
            'mantissa'

        """
        hyperparameters = {
            'breast_cancer': {
                'title': 'Breast Cancer',
                'task_type': 'classification',
                'test_ratio': 0.3,
                'validation_ratio': 0.2,
                'hidden_layers': [30, 15],
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'early_stopping': True,
                'patience': 10,
                'random_state': 42,
                'cross_val_type': "holdout",
                'k_folds': 3,
                'interval': 10,
                'mantissa': 6
            },
            'california_housing': {
                'title': 'California Housing',
                'task_type': 'regression',
                'test_ratio': 0.3,
                'validation_atio': 0.2,
                'hidden_layers': [50, 25],
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 16,
                'early_stopping': False,
                'patience': 10,
                'random_state': 42,
                'cross_val_type': "k-fold",
                'k_folds': 3,
                'interval': 5,
                'mantissa': 6
            },
            'digits': {
                'title': 'Digits',
                'task_type': 'classification',
                'test_ratio': 0.3,
                'validation_ratio': 0.2,
                'hidden_layers': [64, 32],
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 32,
                'early_stopping': True,
                'patience': 10,
                'random_state': 42,
                'cross_val_type': "holdout",
                'kFolds': 3,
                'interval': 5,
                'mantissa': 6
            },
            'iris': {
                'title': 'Iris',
                'task_type': 'classification',
                'test_ratio': 0.3,
                'validation_ratio': 0.2,
                'hidden_layers': [10, 5],
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 16,
                'early_stopping': False,
                'patience': 10,
                'random_state': 42,
                'cross_val_type': "holdout",
                'k_folds': 3,
                'interval': 2,
                'mantissa': 6
            },
            'wine': {
                'title': 'Wine',
                'task_type': 'classification',
                'test_ratio': 0.3,
                'validation_ratio': 0.2,
                'hidden_layers': [50, 25],
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 16,
                'early_stopping': False,
                'patience': 10,
                'random_state': 42,
                'cross_val_type': "holdout",
                'k_folds': 3,
                'interval': 2,
                'mantissa': 6
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
            'epochs' (list of epoch numbers at which metrics data was collected)
            'training_loss' (LOSS)
            'validation_loss' (VALIDATION LOSS)
            'training_accuracy' (ACCURACY for classification tasks only)
            'validation_accuracy' (ACCURACYfor classification tasks only)
            'training_mae' (MAE for regression tasks only)
            'validation_mae' (MAE for regression tasks only)
            'training_mse' (MSE for regression tasks only)
            'validation_mse' (MSE for regression tasks only)
            'training_r2' (R^2 for regression tasks only)
            'validation_r2' (R^2 for regression tasks only)

        """

        return {
                'epochs': model.epoch_list,
                'training_loss': model.training_loss_list,
                'validation_loss': model.validation_loss_list,
                'training_accuracy': model.training_accuracy_list,
                'validation_accuracy': model.validation_accuracy_list,
                'training_mae': model.training_mae_list,
                'validation_mae': model.validation_mae_list,
                'training_mse': model.training_mse_list,
                'validation_mse': model.validation_mse_list,
                'training_r2': model.training_r2_list,
                'validation_r2': model.validation_r2_list
            }

    @staticmethod
    def Initialize(hyperparameters, dataset):
        """
        Initializes an ANN model with the input dataset and hyperparameters.

        Parameters
        ----------
        hyperparameters : dict
            The hyperparameters dictionary. You can create one using ANN.Hyperparameters() or, if you are using a sample Dataset, you can get it from ANN.HyperParametersBySampleName.
        dataset : sklearn.utils._bunch.Bunch
            The input dataset.
               
        Returns
        -------
        _ANNModel
            Returns the initialized model.

        """
        def prepare_data(dataset, task_type='classification'):
            X, y = dataset.data, dataset.target
            
            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long if task_type != 'regression' else torch.float32)
            return X, y
        
        task_type = hyperparameters['task_type']
        if task_type not in ['classification', 'regression']:
            print("ANN.ModelInitialize - Error: The task type in the input hyperparameters parameter is not recognized. It must be either 'classification' or 'regression'. Returning None.")
            return None
        X, y = prepare_data(dataset, task_type=task_type)
        model = _ANN(input_size=X.shape[1], hyperparameters=hyperparameters, dataset=dataset)
        return model

    @staticmethod
    def Train(hyperparameters, dataset):
        """
        Trains the input model given the input features (X), and target (y).

        Parameters
        ----------
        hyperparameters : dict
            The hyperparameters dictionary. You can create one using ANN.Hyperparameters() or, if you are using a sample Dataset, you can get it from ANN.HyperParametersBySampleName.
        dataset : sklearn.utils._bunch.Bunch
            The input dataset.
       
        Returns
        -------
        _ANNModel
            Returns the trained model.

        """
        def prepare_data(dataset, task_type='classification'):
            X, y = dataset.data, dataset.target
            
            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long if task_type != 'regression' else torch.float32)
            return X, y
        
        X, y = prepare_data(dataset, task_type=hyperparameters['task_type'])
        model = _ANN(input_size=X.shape[1], hyperparameters=hyperparameters, dataset=dataset)
        model.cross_validate(X, y)
        return model

    @staticmethod
    def Test(model, hyperparameters, dataset):
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
            Returns four lists: y_test, predictions, metrics, and confusion_matrix.

        """
        def prepare_data(dataset, task_type='classification'):
            X, y = dataset.data, dataset.target
            
            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long if task_type != 'regression' else torch.float32)
            return X, y
        
        X, y = prepare_data(dataset, task_type=hyperparameters['task_type'])
        X_train, X_test, y_train,y_test = ANN.DatasetSplit(X, y, testRatio=hyperparameters['test_ratio'], randomState=hyperparameters['random_state'])
        metrics, predictions = model.evaluate_model(X_test, y_test)
        confusion_matrix = None
        if hyperparameters['task_type'] != 'regression':
            confusion_matrix = model.confusion_matrix(y_test, predictions)
        return y_test, predictions, metrics, confusion_matrix

    @staticmethod
    def Figures(model, width=900, height=600, template="plotly", colorScale='viridis', colorSamples=10):
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
        import numpy as np
        figures = []
        filenames = []
        if model.task_type == 'classification':
            confusion_matrix = model.metrics['confusion_matrix']
            confusion_matrix_figure = Plotly.FigureByConfusionMatrix(confusion_matrix, width=width, height=height, colorScale=colorScale, colorSamples=colorSamples)
            confusion_matrix_figure.update_layout(title=model.title+"<BR>Confusion Matrix")
            figures.append(confusion_matrix_figure)
            filenames.append("ConfusionMatrix")
            data_lists = [[model.train_loss_list, model.val_loss_list], [model.train_accuracy_list, model.val_accuracy_list]]
            label_lists = [['Training Loss', 'Validation Loss'], ['Training Accuracy', 'Validation Accuracy']]
            titles = ['Training and Validation Loss', 'Training and Validation Accuracy']
            titles = [model.title+"<BR>"+t for t in titles]
            legend_titles = ['Loss Type', 'Accuracy Type']
            xaxis_titles = ['Epoch', 'Epoch']
            yaxis_titles = ['Loss', 'Accuracy']
            filenames = yaxis_titles
            
        elif model.task_type.lower() == 'regression':
            data_lists = [[model.train_loss_list, model.val_loss_list], [model.train_mae_list, model.val_mae_list], [model.train_mse_list, model.val_mse_list], [model.train_r2_list, model.val_r2_list]]
            label_lists = [['Training Loss', 'Validation Loss'], ['Training MAE', 'Validation MAE'], ['Training MSE', 'Validation MSE'],['Training R^2', 'Validation R^2']]
            titles = ['Training and Validation Loss', 'Training and Validation MAE', 'Training and Validation MSE', 'Training and Validation R^2']
            titles = [model.title+"<BR>"+t for t in titles]
            legend_titles = ['Loss Type', 'MAE Type', 'MSE Type', 'R^2 Type']
            xaxis_titles = ['Epoch', 'Epoch', 'Epoch', 'Epoch']
            yaxis_titles = ['Loss', 'MAE', 'MSE', 'R^2']
            filenames = yaxis_titles
        else:
            print("ANN.ModelFigures - Error: Could not recognize model task type. Returning None.")
            return None
        for i in range(len(data_lists)):
            data = data_lists[i]
            labels = label_lists[i]
            title = titles[i]
            legend_title = legend_titles[i]
            xaxis_title = xaxis_titles[i]
            yaxis_title = yaxis_titles[i]
            x = model.epoch_list
            

            figure = go.Figure()
            min_x = np.inf
            max_x = -np.inf
            min_y = np.inf
            max_y = -np.inf
            for j in range(len(data)):
                y = data[j]
                figure.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=labels[j]))
                min_x = min(min_x, min(x))
                max_x = max(max_x, max(x))
                min_y = min(min_y, min(y))
                max_y = max(max_y, max(y))

            figure.update_layout(
                xaxis=dict(range=[0, max_x+max_x*0.01]),
                yaxis=dict(range=[min_y-min_y*0.01, max_y+max_y*0.01]),
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
    def Metrics(model):
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
        metrics = model.metrics
        return metrics

    @staticmethod
    def Save(model, path, overwrite=False):
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
            print("ANN.Save - Error: The input model parameter is invalid. Returning None.")
            return None
        if path == None:
            print("ANN.Save - Error: The input path parameter is invalid. Returning None.")
            return None
        if not overwrite and os.path.exists(path):
            print("ANN.Save - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
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
    def Load(model, path):
        """
        Loads the model state dictionary found at the input file path. The model input parameter must be pre-initialized using the ANN.Initialize() method.

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
            print("ANN.Load - Error: The specified path does not exist. Returning None.")
            return None
        model.load(path)
        return model 
    