import sys, subprocess
try:
    import numpy as np
except:
    call = [sys.executable, '-m', 'pip', 'install', 'numpy', '-t', sys.path[0]]
    subprocess.run(call)
    import numpy as np
try:
    import pandas as pd
except:
    call = [sys.executable, '-m', 'pip', 'install', 'pandas', '-t', sys.path[0]]
    subprocess.run(call)
    import pandas as pd
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except:
    call = [sys.executable, '-m', 'pip', 'install', 'torch', '-t', sys.path[0]]
    subprocess.run(call)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.utils.data import DataLoader, ConcatDataset
try:
    import dgl
    from dgl.data import DGLDataset
    from dgl.dataloading import GraphDataLoader
    from dgl.nn import GINConv, GraphConv, SAGEConv, TAGConv
except:
    call = [sys.executable, '-m', 'pip', 'install', 'dgl', 'dglgo', '-f', 'https://data.dgl.ai/wheels/repo.html', '--upgrade', '-t', sys.path[0]]
    subprocess.run(call)
    import dgl
    from dgl.data import DGLDataset
    from dgl.nn import GraphConv
try:
    import sklearn
    from sklearn.model_selection import KFold
except:
    call = [sys.executable, '-m', 'pip', 'install', 'sklearn', '-t', sys.path[0]]
    subprocess.run(call)
    import sklearn
    from sklearn.model_selection import KFold


import topologicpy
import topologic
from topologicpy.Dictionary import Dictionary
import os
import plotly.express as px

import random
import time
from datetime import datetime

checkpoint_path = os.path.join(os.path.expanduser('~'), "dgl_classifier.pt")
results_path = os.path.join(os.path.expanduser('~'), "dgl_results.csv")

class GraphDGL(DGLDataset):
    def __init__(self, graphs, labels, node_attr_keys):
        super().__init__(name='GraphDGL')
        self.graphs = graphs
        self.labels = torch.LongTensor(labels)
        # as all graphs have same length of node features then we get dim_nfeats from first graph in the list
        self.dim_nfeats = graphs[0].ndata[node_attr_keys[0]].shape[1]
                # to get the number of classes for graphs
        self.gclasses = len(set(labels))
        self.node_attr_key = node_attr_keys[0]

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

class Hparams:
    def __init__(self, optimizer_str="Adam", amsgrad=False, betas=(0.9, 0.999), eps=1e-6, lr=0.001, lr_decay= 0, maximize=False, rho=0.9, weight_decay=0, cv_type="Holdout", split=0.2, k_folds=5, hidden_layers=[32], conv_layer_type='SAGEConv', pooling="AvgPooling", batch_size=32, epochs=1, 
                 use_gpu=False, loss_function="Cross-Entropy", checkpoint_path=checkpoint_path, results_path=results_path):
        """
        Parameters
        ----------
        cv : str
            An int value in the range of 0 to X to define the method of cross-validation
            "Holdout": Holdout
            "K-Fold": K-Fold cross validation
        k_folds : int
            An int value in the range of 2 to X to define the number of k-folds for cross-validation. Default is 5.
        split : float
            A float value in the range of 0 to 1 to define the split of train
            and test data. A default value of 0.2 means 20% of data will be
            used for testing and remaining 80% for training
        hidden_layers : list
            List of hidden neurons for each layer such as [32] will mean
            that there is one hidden layers in the network with 32 neurons
        optimizer : torch.optim object
            This will be the selected optimizer from torch.optim package. By
            default, torch.optim.Adam is selected
        learning_rate : float
            a step value to be used to apply the gradients by optimizer
        batch_size : int
            to define a set of samples to be used for training and testing in 
            each step of an epoch
        epochs : int
            An epoch means training the neural network with all the training data for one cycle. In an epoch, we use all of the data exactly once. A forward pass and a backward pass together are counted as one pass
        checkpoint_path: str
            Path to save the classifier after training. It is preferred for 
            the path to have .pt extension
        use_GPU : use the GPU. Otherwise, use the CPU

        Returns
        -------
        None

        """
        
        self.optimizer_str = optimizer_str
        self.amsgrad = amsgrad
        self.betas = betas
        self.eps = eps
        self.lr = lr
        self.lr_decay = lr_decay
        self.maximize = maximize
        self.rho = rho
        self.weight_decay = weight_decay
        self.cv_type = cv_type
        self.split = split
        self.k_folds = k_folds
        self.hidden_layers = hidden_layers
        self.conv_layer_type = conv_layer_type
        self.pooling = pooling
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_gpu = use_gpu
        self.loss_function = loss_function
        self.checkpoint_path = checkpoint_path
        self.results_path = results_path

class GCN_Classic(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        """

        Parameters
        ----------
        in_feats : int
            Input dimension in the form of integer
        h_feats : list
            List of hidden neurons for each hidden layer
        num_classes : int
            Number of output classes

        Returns
        -------
        None.

        """
        super(GCN_Classic, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = []
        dim = [in_feats] + h_feats
        for i in range(1, len(dim)):
            self.list_of_layers.append(GraphConv(dim[i-1], dim[i]))
        self.final = GraphConv(dim[-1], num_classes)

    def forward(self, g, in_feat):
        h = in_feat
        for i in range(len(self.list_of_layers)):
            h = self.list_of_layers[i](g, h)
            h = F.relu(h)
        h = self.final(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

class GCN_GINConv(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, pooling):
        super(GCN_GINConv, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = []
        dim = [in_feats] + h_feats

        # Convolution (Hidden) Layers
        for i in range(1, len(dim)):
            lin = nn.Linear(dim[i-1], dim[i])
            self.list_of_layers.append(GINConv(lin, 'sum'))

        # Final Layer
        self.final = nn.Linear(dim[-1], num_classes)

        # Pooling layer
        if pooling == "AvgPooling":
            self.pooling_layer = dgl.nn.AvgPooling()
        elif pooling == "MaxPooling":
            self.pooling_layer = dgl.nn.MaxPooling()
        elif pooling == "SumPooling":
            self.pooling_layer = dgl.nn.SumPooling()
        else:
            raise NotImplementedError

    def forward(self, g, in_feat):
        h = in_feat
        # Generate node features
        for i in range(len(self.list_of_layers)): # Aim for 2 about 3 layers
            h = self.list_of_layers[i](g, h)
            h = F.relu(h)
        # h will now be matrix of dimension num_nodes by h_feats[-1]
        h = self.final(h)
        g.ndata['h'] = h
        # Go from node level features to graph level features by pooling
        h = self.pooling_layer(g, h)
        # h will now be vector of dimension num_classes
        return h

class GCN_GraphConv(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, pooling):
        super(GCN_GraphConv, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = []
        dim = [in_feats] + h_feats

        # Convolution (Hidden) Layers
        for i in range(1, len(dim)):
            self.list_of_layers.append(GraphConv(dim[i-1], dim[i]))

        # Final Layer
        # Followed example at: https://docs.dgl.ai/tutorials/blitz/5_graph_classification.html#sphx-glr-tutorials-blitz-5-graph-classification-py
        self.final = GraphConv(dim[-1], num_classes)

        # Pooling layer
        if pooling == "AvgPooling":
            self.pooling_layer = dgl.nn.AvgPooling()
        elif pooling == "MaxPooling":
            self.pooling_layer = dgl.nn.MaxPooling()
        elif pooling == "SumPooling":
            self.pooling_layer = dgl.nn.SumPooling()
        else:
            raise NotImplementedError

    def forward(self, g, in_feat):
        h = in_feat
        # Generate node features
        for i in range(len(self.list_of_layers)): # Aim for 2 about 3 layers
            h = self.list_of_layers[i](g, h)
            h = F.relu(h)
        # h will now be matrix of dimension num_nodes by h_feats[-1]
        h = self.final(g,h)
        g.ndata['h'] = h
        # Go from node level features to graph level features by pooling
        h = self.pooling_layer(g, h)
        # h will now be vector of dimension num_classes
        return h

class GCN_SAGEConv(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, pooling):
        super(GCN_SAGEConv, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = []
        dim = [in_feats] + h_feats

        # Convolution (Hidden) Layers
        for i in range(1, len(dim)):
            self.list_of_layers.append(SAGEConv(dim[i-1], dim[i], aggregator_type='pool'))

        # Final Layer
        self.final = nn.Linear(dim[-1], num_classes)

        # Pooling layer
        if pooling == "AvgPooling":
            self.pooling_layer = dgl.nn.AvgPooling()
        elif pooling == "MaxPooling":
            self.pooling_layer = dgl.nn.MaxPooling()
        elif pooling == "SumPooling":
            self.pooling_layer = dgl.nn.SumPooling()
        else:
            raise NotImplementedError

    def forward(self, g, in_feat):
        h = in_feat
        # Generate node features
        for i in range(len(self.list_of_layers)): # Aim for 2 about 3 layers
            h = self.list_of_layers[i](g, h)
            h = F.relu(h)
        # h will now be matrix of dimension num_nodes by h_feats[-1]
        h = self.final(h)
        g.ndata['h'] = h
        # Go from node level features to graph level features by pooling
        h = self.pooling_layer(g, h)
        # h will now be vector of dimension num_classes
        return h

class GCN_TAGConv(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, pooling):
        super(GCN_TAGConv, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = []
        dim = [in_feats] + h_feats

        # Convolution (Hidden) Layers
        for i in range(1, len(dim)):
            self.list_of_layers.append(TAGConv(dim[i-1], dim[i], k=2))

        # Final Layer
        self.final = nn.Linear(dim[-1], num_classes)

        # Pooling layer
        if pooling == "AvgPooling":
            self.pooling_layer = dgl.nn.AvgPooling()
        elif pooling == "MaxPooling":
            self.pooling_layer = dgl.nn.MaxPooling()
        elif pooling == "SumPooling":
            self.pooling_layer = dgl.nn.SumPooling()
        else:
            raise NotImplementedError

    def forward(self, g, in_feat):
        h = in_feat
        # Generate node features
        for i in range(len(self.list_of_layers)): # Aim for 2 about 3 layers
            h = self.list_of_layers[i](g, h)
            h = F.relu(h)
        # h will now be matrix of dimension num_nodes by h_feats[-1]
        h = self.final(h)
        g.ndata['h'] = h
        # Go from node level features to graph level features by pooling
        h = self.pooling_layer(g, h)
        # h will now be vector of dimension num_classes
        return h


class ClassifierSplit:
    def __init__(self, hparams, trainingDataset):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        self.trainingDataset = trainingDataset
        self.hparams = hparams
        if hparams.conv_layer_type == 'Classic':
            self.model = GCN_Classic(trainingDataset.dim_nfeats, hparams.hidden_layers, 
                            trainingDataset.gclasses).to(device)
        elif hparams.conv_layer_type == 'GINConv':
            self.model = GCN_GINConv(trainingDataset.dim_nfeats, hparams.hidden_layers, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type == 'GraphConv':
            self.model = GCN_GraphConv(trainingDataset.dim_nfeats, hparams.hidden_layers, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type == 'SAGEConv':
            self.model = GCN_SAGEConv(trainingDataset.dim_nfeats, hparams.hidden_layers, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type == 'TAGConv':
            self.model = GCN_TAGConv(trainingDataset.dim_nfeats, hparams.hidden_layers, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type == 'GCN':
            self.model = GCN_Classic(trainingDataset.dim_nfeats, hparams.hidden_layers, 
                            trainingDataset.gclasses).to(device)
        else:
            raise NotImplementedError

        if hparams.optimizer_str == "Adadelta":
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), eps=hparams.eps, 
                                            lr=hparams.lr, rho=hparams.rho, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str == "Adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), eps=hparams.eps, 
                                            lr=hparams.lr, lr_decay=hparams.lr_decay, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=hparams.amsgrad, betas=hparams.betas, eps=hparams.eps, 
                                            lr=hparams.lr, maximize=hparams.maximize, weight_decay=hparams.weight_decay)
        self.use_gpu = hparams.use_gpu
        self.training_loss_list = []
        self.testing_loss_list = []
        self.training_accuracy_list = []
        self.testing_accuracy_list = []
        self.node_attr_key = trainingDataset.node_attr_key
        # train test split
        idx = torch.randperm(len(trainingDataset))
        num_train = int(len(trainingDataset) * hparams.split)
        
        train_sampler = SubsetRandomSampler(idx[:num_train])
        test_sampler = SubsetRandomSampler(idx[num_train:])
        
        self.train_dataloader = GraphDataLoader(trainingDataset, sampler=train_sampler, 
                                                batch_size=hparams.batch_size,
                                                drop_last=False)
        self.test_dataloader = GraphDataLoader(trainingDataset, sampler=test_sampler,
                                                batch_size=hparams.batch_size,
                                                drop_last=False)
    def train(self):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        # Init the loss and accuracy reporting lists
        self.training_accuracy_list = []
        self.training_loss_list = []
        self.testing_accuracy_list = []
        self.testing_loss_list = []

        # Run the training loop for defined number of epochs
        for _ in range(self.hparams.epochs):
            num_correct = 0
            num_tests = 0
            temp_loss_list = []

            # Iterate over the DataLoader for training data
            for batched_graph, labels in self.train_dataloader:
                    
                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
                # Compute loss
                if self.hparams.loss_function == "Negative Log Likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, labels)
                elif self.hparams.loss_function == "Cross Entropy":
                    loss = F.cross_entropy(pred, labels)

                # Save loss information for reporting
                temp_loss_list.append(loss.item())
                num_correct += (pred.argmax(1) == labels).sum().item()
                num_tests += len(labels)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()

            self.training_accuracy = num_correct / num_tests
            self.training_accuracy_list.append(self.training_accuracy)
            self.training_loss_list.append(sum(temp_loss_list) / len(temp_loss_list))
            self.test()
            self.testing_accuracy_list.append(self.testing_accuracy)
            self.testing_loss_list.append(self.testing_loss)
        if self.hparams.checkpoint_path is not None:
            # Save the entire model
            torch.save(self.model, self.hparams.checkpoint_path)

    def test(self):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        num_correct = 0
        num_tests = 0
        temp_testing_loss = []
        for batched_graph, labels in self.test_dataloader:
            pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
            if self.hparams.loss_function == "Negative Log Likelihood":
                logp = F.log_softmax(pred, 1)
                loss = F.nll_loss(logp, labels)
            elif self.hparams.loss_function == "Cross Entropy":
                loss = F.cross_entropy(pred, labels)
            temp_testing_loss.append(loss.item())
            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
        self.testing_loss = (sum(temp_testing_loss) / len(temp_testing_loss))
        self.testing_accuracy = num_correct / num_tests
        return self.testing_accuracy

class ClassifierKFold:
    def __init__(self, hparams, trainingDataset, validationDataset):
        self.trainingDataset = trainingDataset
        self.validationDataset = validationDataset
        self.hparams = hparams
        # at beginning of the script
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        if hparams.conv_layer_type == 'Classic':
            self.model = GCN_Classic(trainingDataset.dim_nfeats, hparams.hidden_layers, 
                            trainingDataset.gclasses).to(device)
        elif hparams.conv_layer_type == 'GINConv':
            self.model = GCN_GINConv(trainingDataset.dim_nfeats, hparams.hidden_layers, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type == 'GraphConv':
            self.model = GCN_GraphConv(trainingDataset.dim_nfeats, hparams.hidden_layers, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type == 'SAGEConv':
            self.model = GCN_SAGEConv(trainingDataset.dim_nfeats, hparams.hidden_layers, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type == 'TAGConv':
            self.model = GCN_TAGConv(trainingDataset.dim_nfeats, hparams.hidden_layers, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        else:
            raise NotImplementedError

        if hparams.optimizer_str == "Adadelta":
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), eps=hparams.eps, 
                                            lr=hparams.lr, rho=hparams.rho, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str == "Adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), eps=hparams.eps, 
                                            lr=hparams.lr, lr_decay=hparams.lr_decay, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=hparams.amsgrad, betas=hparams.betas, eps=hparams.eps, 
                                            lr=hparams.lr, maximize=hparams.maximize, weight_decay=hparams.weight_decay)
        self.use_gpu = hparams.use_gpu
        self.training_loss_list = []
        self.testing_loss_list = []
        self.training_accuracy_list = []
        self.testing_accuracy_list = []
        self.node_attr_key = trainingDataset.node_attr_key

    def reset_weights(self):
        '''
        Try resetting model weights to avoid
        weight leakage.
        '''
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def train(self):
        # The number of folds (This should come from the hparams)
        k_folds = self.hparams.k_folds

        # Init the loss and accuracy reporting lists
        self.training_accuracy_list = []
        self.training_loss_list = []
        self.testing_accuracy_list = []
        self.testing_loss_list = []

        # Set fixed random number seed
        torch.manual_seed(42)
        
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=k_folds, shuffle=True)

        # K-fold Cross-validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.trainingDataset)):
            epoch_training_loss_list = []
            epoch_training_accuracy_list = []
            epoch_testing_loss_list = []
            epoch_testing_accuracy_list = []
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            self.train_dataloader = GraphDataLoader(self.trainingDataset, sampler=train_subsampler, 
                                                batch_size=self.hparams.batch_size,
                                                drop_last=False)
            self.test_dataloader = GraphDataLoader(self.trainingDataset, sampler=test_subsampler,
                                                batch_size=self.hparams.batch_size,
                                                drop_last=False)
            # Init the neural network
            self.model.apply(self.reset_weights())

            # Run the training loop for defined number of epochs
            for _ in range(self.hparams.epochs):
                num_correct = 0
                num_tests = 0
                training_temp_loss_list = []

                # Iterate over the DataLoader for training data
                for batched_graph, labels in self.train_dataloader:
                    
                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Perform forward pass
                    pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float())

                    # Compute loss
                    if self.hparams.loss_function == "Negative Log Likelihood":
                        logp = F.log_softmax(pred, 1)
                        loss = F.nll_loss(logp, labels)
                    elif self.hparams.loss_function == "Cross Entropy":
                        loss = F.cross_entropy(pred, labels)

                    # Save loss information for reporting
                    training_temp_loss_list.append(loss.item())
                    num_correct += (pred.argmax(1) == labels).sum().item()
                    num_tests += len(labels)

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    self.optimizer.step()

                self.training_accuracy = num_correct / num_tests
                epoch_training_accuracy_list.append(self.training_accuracy)
                epoch_training_loss_list.append(sum(training_temp_loss_list) / len(training_temp_loss_list))
                self.test()
                epoch_testing_accuracy_list.append(self.testing_accuracy)
                epoch_testing_loss_list.append(self.testing_loss)
            if self.hparams.checkpoint_path is not None:
                # Save the entire model
                torch.save(self.model, self.hparams.checkpoint_path+"-fold_"+str(fold))
            self.training_accuracy_list.append(epoch_training_accuracy_list)
            self.training_loss_list.append(epoch_training_loss_list)
            self.testing_accuracy_list.append(epoch_testing_accuracy_list)
            self.testing_loss_list.append(epoch_testing_loss_list)

    def test(self):
        num_correct = 0
        num_tests = 0
        temp_testing_loss = []
        for batched_graph, labels in self.test_dataloader:
            pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float())
            if self.hparams.loss_function == "Negative Log Likelihood":
                logp = F.log_softmax(pred, 1)
                loss = F.nll_loss(logp, labels)
            elif self.hparams.loss_function == "Cross Entropy":
                loss = F.cross_entropy(pred, labels)
            temp_testing_loss.append(loss.item())
            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
        self.testing_loss = (sum(temp_testing_loss) / len(temp_testing_loss))
        self.testing_accuracy = num_correct / num_tests
        return self.testing_accuracy

    def accuracy(self, item):
        dgl_labels, dgl_predictions = item
    
        num_correct = 0
        for i in range(len(dgl_predictions)):
            if dgl_predictions[i] == dgl_labels[i]:
                num_correct = num_correct + 1
        return (num_correct / len(dgl_predictions))

    def predict(self):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        predicted_labels = []
        idx = torch.randperm(len(self.validationDataset))
        num_train = int(len(self.validationDataset))
        sampler = SubsetRandomSampler(idx[:num_train])
        dataloader = GraphDataLoader(self.validationDataset, sampler=sampler, 
                                                batch_size=1,
                                                drop_last=False)
        num_correct = 0
        num_tests = 0
        for batched_graph, labels in dataloader:
            pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
        accuracy = num_correct / num_tests
        return accuracy

    def validate(self):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        # Set training to 100% of the data, validate, and save a final model
        idx = torch.randperm(len(self.trainingDataset))
        num_train = int(len(self.trainingDataset))
        sampler = SubsetRandomSampler(idx[:num_train])
        dataloader = GraphDataLoader(self.trainingDataset, sampler=sampler, 
                                                batch_size=self.hparams.batch_size,
                                                drop_last=False)
        # Once a model is chosen, train on all the data and save
        for e in range(self.hparams.epochs):
            num_correct = 0
            num_tests = 0
            for batched_graph, labels in dataloader:
                #pred = self.model(batched_graph, batched_graph.ndata['attr'].float()).to(device)
                pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
                if self.hparams.loss_function == "Negative Log Likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, labels)
                elif self.hparams.loss_function == "Cross Entropy":
                    loss = F.cross_entropy(pred, labels)
                num_correct += (pred.argmax(1) == labels).sum().item()
                num_tests += len(labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            training_accuracy = num_correct / num_tests
            validation_accuracy = self.predict()
            if validation_accuracy >= training_accuracy and validation_accuracy > 0.6:
                break
        print("Validation - Stopped at Epoch:", e+1)
        if self.hparams.checkpoint_path is not None:
            # Save the entire model
            torch.save(self.model, self.hparams.checkpoint_path)







class DGL:
    @staticmethod
    def Accuracy(dgl_labels, dgl_predictions):
        """
        Parameters
        ----------
        dgl_labels : TYPE
            DESCRIPTION.
        dgl_predictions : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        # dgl_labels, dgl_predictions = item
        num_correct = 0
        mask = []
        for i in range(len(dgl_predictions)):
            if dgl_predictions[i] == dgl_labels[i]:
                num_correct = num_correct + 1
                mask.append(True)
            else:
                mask.append(False)
        size = len(dgl_predictions)
        return [size, num_correct, len(dgl_predictions)- num_correct, mask, num_correct / len(dgl_predictions)]
    
    @staticmethod
    def ClassifierByFilePath(item):
        """
        Parameters
        ----------
        item : str
            Path for the saved checkpoint of the model

        Returns
        -------
        Object saved with torch.save
            The classifier model

        """
        return torch.load(item)
    
    @staticmethod
    def DatasetByDGLGraph(dgl_graphs, dgl_labels, node_attr_key):
        """
        Parameters
        ----------
        dgl_graphs : TYPE
            DESCRIPTION.
        dgl_labels : TYPE
            DESCRIPTION.
        node_attr_key : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # dgl_graphs, dgl_labels, node_attr_key = item
        if isinstance(dgl_graphs, list) == False:
            dgl_graphs = [dgl_graphs]
        if isinstance(dgl_labels, list) == False:
            dgl_labels = [dgl_labels]
        return GraphDGL(dgl_graphs, dgl_labels, node_attr_key)
    
    @staticmethod
    def DatasetByImportedCSV_NC(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        graphs_folder_path = item
        return dgl.data.CSVDataset(graphs_folder_path, force_reload=True)
    
    @staticmethod
    def DatasetBySample(sample):
        """
        Parameters
        ----------
        sample : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        dataset = dgl.data.TUDataset(sample)
        dgl_graphs, dgl_labels = zip(*[dataset[i] for i in range(len(dataset.graph_lists))])
        if sample == 'ENZYMES':
            node_attr_key = 'node_attr'
        elif sample == 'DD':
            node_attr_key = 'node_labels'
        elif sample == 'COLLAB':
            node_attr_key = '_ID'
        elif sample == 'MUTAG':
            node_attr_key = 'node_labels'
        else:
            raise NotImplementedError
        return GraphDGL(dgl_graphs, dgl_labels, node_attr_key)
    
    @staticmethod
    def DatasetBySample_NC(sample):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        if sample == 'Cora':
            return [dgl.data.CoraGraphDataset(), 7]
        elif sample == 'Citeseer':
            return [dgl.data.CiteseerGraphDataset(), 6]
        elif sample == 'Pubmed':
            return [dgl.data.PubmedGraphDataset(), 3]
        else:
            raise NotImplementedError
    
    @staticmethod
    def DatasetGraphs_NC(dataset):
        """
        Parameters
        ----------
        dataset : TYPE
            DESCRIPTION.

        Returns
        -------
        graphs : TYPE
            DESCRIPTION.

        """
        try:
            _ = dataset[1]
        except:
            dataset = [dataset[0]]
        graphs = []
        for aGraph in dataset:
            if isinstance(aGraph, tuple):
                aGraph = aGraph[0]
            graphs.append(aGraph)
        return graphs
    
    @staticmethod
    def OneHotEncode(item, categories):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.
        categories : TYPE
            DESCRIPTION.

        Returns
        -------
        returnList : TYPE
            DESCRIPTION.

        """
        returnList = []
        for i in range(len(categories)):
            if item == categories[i]:
                returnList.append(1)
            else:
                returnList.append(0)
        return returnList
    
    @staticmethod
    def ByGraph(graph, bidirectional, key, categories, node_attr_key, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        bidirectional : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.
        categories : TYPE
            DESCRIPTION.
        node_attr_key : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # graph, bidirectional, key, categories, node_attr_key, tolerance = item
        
        def vertexIndex(v, vertexList, tolerance):
            for i in range(len(vertexList)):
                d = topologic.VertexUtility.Distance(v, vertexList[i])
                if d < tolerance:
                    return i
            return None

        def graphVertices(graph):
            vertices = []
            if graph:
                try:
                    _ = graph.Vertices(vertices)
                except:
                    print("ERROR: (Topologic>Graph.Vertices) operation failed.")
                    vertices = None
            if vertices:
                return vertices
            else:
                return []

        def graphEdges(graph):
            if graph:
                try:
                    vertices = []
                    edges = []
                    _ = graph.Vertices(vertices)
                    _ = graph.Edges(vertices, 0.001, edges)
                except:
                    print("ERROR: (Topologic>Graph.Edges) operation failed.")
                    edges = None
            if edges:
                return edges
            else:
                return []
        
        graph_dict = {}
        vertices = graphVertices(graph)
        edges = graphEdges(graph)
        graph_dict["num_nodes"] = len(vertices)
        graph_dict["src"] = []
        graph_dict["dst"] = []
        graph_dict["node_labels"] = {}
        graph_dict["node_features"] = []
        nodes = []
        graph_edges = []

        
        for i in range(len(vertices)):
            vDict = vertices[i].GetDictionary()
            vLabel = Dictionary.ValueAtKey(vDict, key)
            graph_dict["node_labels"][i] = vLabel
            # appending tensor of onehotencoded feature for each node following index i
            graph_dict["node_features"].append(torch.tensor(DGL.OneHotEncode(vLabel, categories)))
            nodes.append(i)


        for i in range(len(edges)):
            e = edges[i]
            sv = e.StartVertex()
            ev = e.EndVertex()
            sn = nodes[vertexIndex(sv, vertices, tolerance)]
            en = nodes[vertexIndex(ev, vertices, tolerance)]
            if (([sn,en] in graph_edges) == False) and (([en,sn] in graph_edges) == False):
                graph_edges.append([sn,en])

        for anEdge in graph_edges:
            graph_dict["src"].append(anEdge[0])
            graph_dict["dst"].append(anEdge[1])

        # Create DDGL graph
        src = np.array(graph_dict["src"])
        dst = np.array(graph_dict["dst"])
        num_nodes = graph_dict["num_nodes"]
        # Create a graph
        dgl_graph = dgl.graph((src, dst), num_nodes=num_nodes)
        
        # Setting the node features as node_attr_key using onehotencoding of vlabel
        dgl_graph.ndata[node_attr_key] = torch.stack(graph_dict["node_features"])
        
        if bidirectional:
            dgl_graph = dgl.add_reverse_edges(dgl_graph)
        return dgl_graph
    
    @staticmethod
    def ByImportedCSV(graphs_file_path, edges_file_path,
                              nodes_file_path, graph_id_header,
                              graph_label_header, num_nodes_header, src_header,
                              dst_header, node_label_header, node_attr_key,
                              categories, bidirectional):
        """
        Parameters
        ----------
        graphs_file_path : TYPE
            DESCRIPTION.
        edges_file_path : TYPE
            DESCRIPTION.
        nodes_file_path : TYPE
            DESCRIPTION.
        graph_id_header : TYPE
            DESCRIPTION.
        graph_label_header : TYPE
            DESCRIPTION.
        num_nodes_header : TYPE
            DESCRIPTION.
        src_header : TYPE
            DESCRIPTION.
        dst_header : TYPE
            DESCRIPTION.
        node_label_header : TYPE
            DESCRIPTION.
        node_attr_key : TYPE
            DESCRIPTION.
        categories : TYPE
            DESCRIPTION.
        bidirectional : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        # graphs_file_path, edges_file_path, nodes_file_path, graph_id_header, graph_label_header, num_nodes_header, src_header, dst_header, node_label_header, node_attr_key, categories, bidirectional = item

        graphs = pd.read_csv(graphs_file_path)
        edges = pd.read_csv(edges_file_path)
        nodes = pd.read_csv(nodes_file_path)
        dgl_graphs = []
        labels = []

        # Create a graph for each graph ID from the edges table.
        # First process the graphs table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in graphs.iterrows():
            label_dict[row[graph_id_header]] = row[graph_label_header]
            num_nodes_dict[row[graph_id_header]] = row[num_nodes_header]
        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby(graph_id_header)
        # For the nodes, first group the table by graph IDs.
        nodes_group = nodes.groupby(graph_id_header)
        # For each graph ID...
        for graph_id in edges_group.groups:
            graph_dict = {}
            graph_dict[src_header] = []
            graph_dict[dst_header] = []
            graph_dict[node_label_header] = {}
            graph_dict["node_features"] = []
            num_nodes = num_nodes_dict[graph_id]
            graph_label = label_dict[graph_id]
            labels.append(graph_label)

            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id[src_header].to_numpy()
            dst = edges_of_id[dst_header].to_numpy()

            # Find the nodes and their labels and features
            nodes_of_id = nodes_group.get_group(graph_id)
            node_labels = nodes_of_id[node_label_header]
            #graph_dict["node_labels"][graph_id] = node_labels

            for node_label in node_labels:
                graph_dict["node_features"].append(torch.tensor(DGL.OneHotEncode(node_label, categories)))
            # Create a graph and add it to the list of graphs and labels.
            dgl_graph = dgl.graph((src, dst), num_nodes=num_nodes)
            # Setting the node features as node_attr_key using onehotencoding of node_label
            dgl_graph.ndata[node_attr_key] = torch.stack(graph_dict["node_features"])
            if bidirectional:
                dgl_graph = dgl.add_reverse_edges(dgl_graph)        
            dgl_graphs.append(dgl_graph)
        return [dgl_graphs, labels]

    @staticmethod
    def ByImportedDGCNN(file_path, categories, bidirectional):
        """
        Parameters
        ----------
        file_path : TYPE
            DESCRIPTION.
        categories : TYPE
            DESCRIPTION.
        bidirectional : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        # file_path, categories, bidirectional = item
        graphs = []
        labels = []
        file = open(file_path)
        if file:
            lines = file.readlines()
            n_graphs = int(lines[0])
            index = 1
            for i in range(n_graphs):
                graph_dict = {}
                graph_dict["src"] = []
                graph_dict["dst"] = []
                graph_dict["node_labels"] = {}
                graph_dict["node_features"] = []
                line = lines[index].split()
                n_nodes = int(line[0])
                graph_dict["num_nodes"] = n_nodes
                graph_label = int(line[1])
                labels.append(graph_label)
                index+=1
                for j in range(n_nodes):
                    line = lines[index+j].split()
                    node_label = int(line[0])
                    graph_dict["node_labels"][j] = node_label
                    graph_dict["node_features"].append(torch.tensor(DGL.OneHotEncode(node_label, categories)))
                    adj_vertices = line[2:]
                    for adj_vertex in adj_vertices:
                        graph_dict["src"].append(j)
                        graph_dict["dst"].append(int(adj_vertex))

                # Create DDGL graph
                src = np.array(graph_dict["src"])
                dst = np.array(graph_dict["dst"])
                # Create a graph
                dgl_graph = dgl.graph((src, dst), num_nodes=graph_dict["num_nodes"])
                # Setting the node features as 'node_attr' using onehotencoding of vlabel
                dgl_graph.ndata['node_attr'] = torch.stack(graph_dict["node_features"])
                if bidirectional:
                    dgl_graph = dgl.add_reverse_edges(dgl_graph)        
                graphs.append(dgl_graph)
                index+=n_nodes
            file.close()
        return [graphs, labels]
    
    @staticmethod
    def EdgeData_NC(dglGraph):
        """
        Parameters
        ----------
        dglGraph : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return dglGraph.edata
    
    @staticmethod
    def NodeData_NC(dglGraph):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return dglGraph.ndata
    
    @staticmethod
    def Hyperparameters(optimizer, cv_type, split, k_folds,
                           hidden_layers_str, conv_layer_type, pooling,
                           batch_size, epochs, use_gpu, loss_function,
                           checkpoint_path, results_path):
        """
        Parameters
        ----------
        optimizer : TYPE
            DESCRIPTION.
        cv_type : TYPE
            DESCRIPTION.
        split : TYPE
            DESCRIPTION.
        k_folds : TYPE
            DESCRIPTION.
        hidden_layers_str : TYPE
            DESCRIPTION.
        conv_layer_type : TYPE
            DESCRIPTION.
        pooling : TYPE
            DESCRIPTION.
        batch_size : TYPE
            DESCRIPTION.
        epochs : TYPE
            DESCRIPTION.
        use_gpu : TYPE
            DESCRIPTION.
        loss_function : TYPE
            DESCRIPTION.
        checkpoint_path : TYPE
            DESCRIPTION.
        results_path : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        # optimizer, cv_type, split, k_folds, hidden_layers_str, conv_layer_type, pooling, batch_size, epochs, use_gpu, loss_function, checkpoint_path, results_path = item
        amsgrad = False
        betas=(0.9, 0.999)
        eps=1e-6
        lr=0.001
        lr_decay= 0
        maximize=False
        rho=0.9
        weight_decay=0
        if optimizer[0] == "Adadelta":
            optimizer_str, eps, lr, rho, weight_decay = optimizer
        elif optimizer[0] == "Adagrad":
            optimizer_str, eps, lr, lr_decay, weight_decay = optimizer
        elif optimizer[0] == "Adam":
            optimizer_str, amsgrad, betas, eps, lr, maximize, weight_decay = optimizer
        hl_str_list = hidden_layers_str.split()
        hidden_layers = []
        for hl in hl_str_list:
            if hl != None and hl.isnumeric():
                hidden_layers.append(int(hl))
        # Classifier: Make sure the file extension is .pt
        ext = checkpoint_path[len(checkpoint_path)-3:len(checkpoint_path)]
        if ext.lower() != ".pt":
            checkpoint_path = checkpoint_path+".pt"
        # Results: Make sure the file extension is .csv
        ext = results_path[len(results_path)-4:len(results_path)]
        if ext.lower() != ".csv":
            results_path = results_path+".csv"
        return Hparams(optimizer_str, amsgrad, betas, eps, lr, lr_decay, maximize, rho, weight_decay, cv_type, split, k_folds, hidden_layers, conv_layer_type, pooling, batch_size, epochs, use_gpu, loss_function, checkpoint_path, results_path)

    @staticmethod
    def Plot(data, data_labels, chart_title, x_title, x_spacing, y_title,
                y_spacing, use_markers, chart_type):
        """
        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        data_labels : TYPE
            DESCRIPTION.
        chart_title : TYPE
            DESCRIPTION.
        x_title : TYPE
            DESCRIPTION.
        x_spacing : TYPE
            DESCRIPTION.
        y_title : TYPE
            DESCRIPTION.
        y_spacing : TYPE
            DESCRIPTION.
        use_markers : TYPE
            DESCRIPTION.
        chart_type : str
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # data, data_labels, chart_title, x_title, x_spacing, y_title, y_spacing, use_markers, chart_type = item
        dlist = list(map(list, zip(*data)))
        df = pd.DataFrame(dlist, columns=data_labels)
        if chart_type.lower() == "line":
            fig = px.line(df, x = data_labels[0], y=data_labels[1:], title=chart_title, markers=use_markers)
        elif chart_type.lower() == "bar":
            fig = px.bar(df, x = data_labels[0], y=data_labels[1:], title=chart_title)
        elif chart_type.lower() == "scatter":
            fig = px.scatter(df, x = data_labels[0], y=data_labels[1:], title=chart_title)
        else:
            raise NotImplementedError
        fig.layout.xaxis.title=x_title
        fig.layout.xaxis.dtick=x_spacing
        fig.layout.yaxis.title=y_title
        fig.layout.yaxis.dtick= y_spacing
        #fig.show()
        import os
        from os.path import expanduser
        home = expanduser("~")
        filePath = os.path.join(home, "dgl_result.html")
        html = fig.to_html(full_html=True, include_plotlyjs=True)
        # save html file
        with open(filePath, "w") as f:
            f.write(html)
        os.system("start "+filePath)
        
    @staticmethod
    def Predict(test_dataset, classifier, node_attr_key):
        """
        Parameters
        ----------
        test_dataset : list
            A list containing several dgl graphs for prediction.
        classifier : TYPE
            DESCRIPTION.
        node_attr_key : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            list containing labels and probabilities

        """
        
        # test_dataset, classifier, node_attr_key  = item
        labels = []
        probabilities = []
        for item in test_dataset:
            graph = item[0]
            print("Node Label:", graph.ndata[node_attr_key].float())
            pred = classifier(graph, graph.ndata[node_attr_key].float())
            labels.append(pred.argmax(1).item())
            probability = (torch.nn.functional.softmax(pred, dim=1).tolist())
            probability = probability[0]
            temp_probability = []
            for p in probability:
                temp_probability.append(round(p, 3))
            probabilities.append(temp_probability)
        return [labels, probabilities]
    
    @staticmethod
    def Predict_NC(classifier, dataset):
        """
        Parameters
        ----------
        classifier : TYPE
            DESCRIPTION.
        dataset : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        from topologicpy.Helper import Helper

        # classifier, dataset  = item
        allLabels = []
        allPredictions = []
        trainLabels = []
        trainPredictions = []
        valLabels = []
        valPredictions = []
        testLabels = []
        testPredictions = []
        
        graphs = DGL.DGLDatasetGraphs_NC(dataset)
        for g in graphs:
            if not g.ndata:
                continue
            train_mask = g.ndata['train_mask']
            val_mask = g.ndata['val_mask']
            test_mask = g.ndata['test_mask']
            features = g.ndata['feat']
            labels = g.ndata['label']
            train_labels = labels[train_mask]
            val_labels = labels[val_mask]
            test_labels = labels[test_mask]
            allLabels.append(labels.tolist())
            trainLabels.append(train_labels.tolist())
            valLabels.append(val_labels.tolist())
            testLabels.append(test_labels.tolist())
            
            # Forward
            logits = classifier(g, features)
            train_logits = logits[train_mask]
            val_logits = logits[val_mask]
            test_logits = logits[test_mask]
            
            # Compute prediction
            predictions = logits.argmax(1)
            train_predictions = train_logits.argmax(1)
            val_predictions = val_logits.argmax(1)
            test_predictions = test_logits.argmax(1)
            allPredictions.append(predictions.tolist())
            trainPredictions.append(train_predictions.tolist())
            valPredictions.append(val_predictions.tolist())
            testPredictions.append(test_predictions.tolist())
            
        return [Helper.Flatten(allLabels), Helper.Flatten(allPredictions),Helper.Flatten(trainLabels), Helper.Flatten(trainPredictions), Helper.Flatten(valLabels), Helper.Flatten(valPredictions), Helper.Flatten(testLabels), Helper.Flatten(testPredictions)]
    




    @staticmethod
    def Train(i, item):
        from topologicpy.Helper import Helper
        import time
        import datetime
        start = time.time()
        hparams, trainingDataset, validationDataset = item
        if hparams.cv_type == "Holdout":
            classifier = ClassifierSplit(hparams, trainingDataset)
            classifier.train()
            accuracy = classifier.test()
        elif hparams.cv_type == "K-Fold":
            classifier = ClassifierKFold(hparams, trainingDataset, validationDataset)
            classifier.train()
            final_epochs = classifier.validate()

            # Transpose the fold data
            temp_list = Helper.Transpose(classifier.training_accuracy_list)
            tr_a_l = []
            for l in temp_list:
                tr_a_l.append((sum(l) / len(l)))
            temp_list = Helper.Transpose(classifier.training_loss_list)
            tr_l_l = []
            for l in temp_list:
                tr_l_l.append((sum(l) / len(l)))
            temp_list = Helper.Transpose(classifier.testing_accuracy_list)
            te_a_l = []
            for l in temp_list:
                te_a_l.append((sum(l) / len(l)))
            temp_list = Helper.Transpose(classifier.testing_loss_list)
            te_l_l = []
            for l in temp_list:
                te_l_l.append((sum(l) / len(l)))

            classifier.training_accuracy_list = tr_a_l
            classifier.training_loss_list = tr_l_l
            classifier.testing_accuracy_list = te_a_l
            classifier.testing_loss_list = te_l_l
    
        end = time.time()
        duration = round(end - start,3)
        utcnow = datetime.utcnow()
        timestamp_str = "UTC-"+str(utcnow.year)+"-"+str(utcnow.month)+"-"+str(utcnow.day)+"-"+str(utcnow.hour)+"-"+str(utcnow.minute)+"-"+str(utcnow.second)
        epoch_list = list(range(1,classifier.hparams.epochs+1))
        data_list = [timestamp_str, duration, classifier.model, classifier.hparams.optimizer_str, classifier.hparams.cv_type, classifier.hparams.split, classifier.hparams.k_folds, classifier.hparams.hidden_layers, classifier.hparams.conv_layer_type, classifier.hparams.pooling, classifier.hparams.lr, classifier.hparams.batch_size, list(range(1,classifier.hparams.epochs+1)), classifier.training_accuracy_list, classifier.testing_accuracy_list, classifier.training_loss_list, classifier.testing_loss_list]
        d2 = [[timestamp_str], [duration], [classifier.hparams.optimizer_str], [classifier.hparams.cv_type], [classifier.hparams.split], [classifier.hparams.k_folds], classifier.hparams.hidden_layers, [classifier.hparams.conv_layer_type], [classifier.hparams.pooling], [classifier.hparams.lr], [classifier.hparams.batch_size], epoch_list, classifier.training_accuracy_list, classifier.testing_accuracy_list, classifier.training_loss_list, classifier.testing_loss_list]
        d2 = Replication.iterate(d2)
        d2 = Helper.Transpose(d2)
    
        data = {'TimeStamp': "UTC-"+str(utcnow.year)+"-"+str(utcnow.month)+"-"+str(utcnow.day)+"-"+str(utcnow.hour)+"-"+str(utcnow.minute)+"-"+str(utcnow.second),
                'Duration': [duration],
                'Optimizer': [classifier.hparams.optimizer_str],
                'CV Type': [classifier.hparams.cv_type],
                'Split': [classifier.hparams.split],
                'K-Folds': [classifier.hparams.k_folds],
                'HL Widths': [classifier.hparams.hidden_layers],
                'Conv Layer Type': [classifier.hparams.conv_layer_type],
                'Pooling': [classifier.hparams.pooling],
                'Learning Rate': [classifier.hparams.lr],
                'Batch Size': [classifier.hparams.batch_size],
                'Epochs': [classifier.hparams.epochs],
                'Training Accuracy': [classifier.training_accuracy_list],
                'Testing Accuracy': [classifier.testing_accuracy_list],
                'Training Loss': [classifier.training_loss_list],
                'Testing Loss': [classifier.testing_loss_list]
            }
        if classifier.hparams.results_path:
            df = pd.DataFrame(d2, columns= ['TimeStamp', 'Duration', 'Optimizer', 'CV Type', 'Split', 'K-Folds', 'HL Widths', 'Conv Layer Type', 'Pooling', 'Learning Rate', 'Batch Size', 'Epochs', 'Training Accuracy', 'Testing Accuracy', 'Training Loss', 'Testing Loss'])
            if i == 0:
                df.to_csv(classifier.hparams.results_path, mode='w+', index = False, header=True)
            else:
                df.to_csv(classifier.hparams.results_path, mode='a', index = False, header=False)


        return data_list

























    @staticmethod
    def Train_NC(graphs, model, hparams):
        """
        Parameters
        ----------
        graphs : TYPE
            DESCRIPTION.
        model : TYPE
            DESCRIPTION.
        hparams : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        # Default optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        if hparams.optimizer_str == "Adadelta":
            optimizer = torch.optim.Adadelta(model.parameters(), eps=hparams.eps, 
                                                lr=hparams.lr, rho=hparams.rho, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str == "Adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), eps=hparams.eps, 
                                                lr=hparams.lr, lr_decay=hparams.lr_decay, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), amsgrad=hparams.amsgrad, betas=hparams.betas, eps=hparams.eps, 
                                                lr=hparams.lr, maximize=hparams.maximize, weight_decay=hparams.weight_decay)
        

        
        for e in range(hparams.epochs):
            best_val_acc = 0
            best_test_acc = 0
            for i in range(len(graphs)):
                g = graphs[i]
                if not g.ndata:
                    continue
                features = g.ndata['feat']
                labels = g.ndata['label']
                train_mask = g.ndata['train_mask']
                val_mask = g.ndata['val_mask']
                test_mask = g.ndata['test_mask']
                # Forward
                logits = model(g, features)
                
                # Compute prediction
                pred = logits.argmax(1)
                
                # Compute loss
                # Note that you should only compute the losses of the nodes in the training set.
                # Compute loss
                if hparams.loss_function == "Negative Log Likelihood":
                    logp = F.log_softmax(logits[train_mask], 1)
                    loss = F.nll_loss(logp, labels[train_mask])
                elif hparams.loss_function == "Cross Entropy":
                    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
                # Compute accuracy on training/validation/test
                train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
                val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
                test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

                # Save the best validation accuracy and the corresponding test accuracy.
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                if test_acc > best_test_acc:
                    best_test_acc = test_acc

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if e % 1 == 0:
                print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                    e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
        return [model, pred]

    @staticmethod
    def TrainClassifier_NC(hparams, dataset, numLabels, sample):
        """
        Parameters
        ----------
        hparams : TYPE
            DESCRIPTION.
        dataset : TYPE
            DESCRIPTION.
        numLabels : TYPE
            DESCRIPTION.
        sample : TYPE
            DESCRIPTION.

        Returns
        -------
        final_model : TYPE
            DESCRIPTION.

        """
        
        # hparams, dataset, numLabels, sample = item
        # We will consider only the first graph in the dataset.
        #graphs = DGLDatasetGraphs_NC.processItem(dataset)
        graphs = DGL.DGLDatasetGraphs_NC(dataset)
        # Sample a random list from the graphs
        if sample < len(graphs) and sample > 0:
            graphs = random.sample(graphs, sample)
        if len(graphs) == 1:
            i = 0
        elif len(graphs) > 1:
            i = random.randrange(0, len(graphs)-1)
        else: # There are no gaphs in the dataset, return None
            return None
        model = GCN(graphs[i].ndata['feat'].shape[1], hparams.hidden_layers, numLabels)
        final_model, predictions = DGL.train2(graphs, model, hparams)
        # Save the entire model
        if hparams.checkpoint_path is not None:
            torch.save(final_model, hparams.checkpoint_path)
        return final_model
    
    
    
    