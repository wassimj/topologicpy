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
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.utils.data import DataLoader, ConcatDataset
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
    call = [sys.executable, '-m', 'pip', 'install', 'scikit-learn', '-t', sys.path[0]]
    subprocess.run(call)
    import sklearn
    from sklearn.model_selection import KFold


import topologicpy
import topologic
from topologicpy.Dictionary import Dictionary
import os


import random
import time
from datetime import datetime

checkpoint_path = os.path.join(os.path.expanduser('~'), "dgl_classifier.pt")
results_path = os.path.join(os.path.expanduser('~'), "dgl_results.csv")

class GraphDGL(DGLDataset):
    def __init__(self, graphs, labels, node_attr_key):
        super().__init__(name='GraphDGL')
        self.graphs = graphs
        self.labels = torch.LongTensor(labels)
        self.node_attr_key = node_attr_key
        # as all graphs have same length of node features then we get dim_nfeats from first graph in the list
        self.dim_nfeats = graphs[0].ndata[node_attr_key].shape[1]
        # to get the number of classes for graphs
        self.gclasses = len(set(labels))

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

class Hparams:
    def __init__(self, optimizer_str="Adam", amsgrad=False, betas=(0.9, 0.999), eps=1e-6, lr=0.001, lr_decay= 0, maximize=False, rho=0.9, weight_decay=0, cv_type="Holdout", split=0.2, k_folds=5, hl_widths=[32], conv_layer_type='SAGEConv', pooling="AvgPooling", batch_size=32, epochs=1, 
                 use_gpu=False, loss_function="Cross Entropy", checkpoint_path=checkpoint_path, results_path=results_path):
        """
        Parameters
        ----------
        cv : str
            A string to define the method of cross-validation
            "Holdout": Holdout
            "K-Fold": K-Fold cross validation
        k_folds : int
            An int value in the range of 2 to X to define the number of k-folds for cross-validation. Default is 5.
        split : float
            A float value in the range of 0 to 1 to define the split of train
            and test data. A default value of 0.2 means 20% of data will be
            used for testing and remaining 80% for training
        hl_widths : list
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
        self.hl_widths = hl_widths
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
        if pooling.lower() == "avgpooling":
            self.pooling_layer = dgl.nn.AvgPooling()
        elif pooling.lower() == "maxpooling":
            self.pooling_layer = dgl.nn.MaxPooling()
        elif pooling.lower() == "sumpooling":
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
        if pooling.lower() == "avgpooling":
            self.pooling_layer = dgl.nn.AvgPooling()
        elif pooling.lower() == "maxpooling":
            self.pooling_layer = dgl.nn.MaxPooling()
        elif pooling.lower() == "sumpooling":
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
        if pooling.lower() == "avgpooling":
            self.pooling_layer = dgl.nn.AvgPooling()
        elif pooling.lower() == "maxpooling":
            self.pooling_layer = dgl.nn.MaxPooling()
        elif pooling.lower() == "sumpooling":
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
        if pooling.lower() == "avgpooling":
            self.pooling_layer = dgl.nn.AvgPooling()
        elif pooling.lower() == "maxpooling":
            self.pooling_layer = dgl.nn.MaxPooling()
        elif pooling.lower() == "sumpooling":
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
        if hparams.conv_layer_type.lower() == 'classic':
            self.model = GCN_Classic(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses).to(device)
        elif hparams.conv_layer_type.lower() == 'ginconv':
            self.model = GCN_GINConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'graphconv':
            self.model = GCN_GraphConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'sageconv':
            self.model = GCN_SAGEConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'tagconv':
            self.model = GCN_TAGConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'gcn':
            self.model = GCN_Classic(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses).to(device)
        else:
            raise NotImplementedError

        if hparams.optimizer_str.lower() == "adadelta":
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), eps=hparams.eps, 
                                            lr=hparams.lr, rho=hparams.rho, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str.lower() == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), eps=hparams.eps, 
                                            lr=hparams.lr, lr_decay=hparams.lr_decay, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str.lower() == "adam":
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
                if self.hparams.loss_function.lower() == "negative log likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, labels)
                elif self.hparams.loss_function.lower() == "cross entropy":
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
            if self.hparams.loss_function.lower() == "negative log likelihood":
                logp = F.log_softmax(pred, 1)
                loss = F.nll_loss(logp, labels)
            elif self.hparams.loss_function.lower() == "cross entropy":
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
        if hparams.conv_layer_type.lower() == 'classic':
            self.model = GCN_Classic(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses).to(device)
        elif hparams.conv_layer_type.lower() == 'ginconv':
            self.model = GCN_GINConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'graphconv':
            self.model = GCN_GraphConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'sageconv':
            self.model = GCN_SAGEConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'tagconv':
            self.model = GCN_TAGConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        else:
            raise NotImplementedError

        if hparams.optimizer_str.lower() == "adadelta":
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), eps=hparams.eps, 
                                            lr=hparams.lr, rho=hparams.rho, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str.lower() == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), eps=hparams.eps, 
                                            lr=hparams.lr, lr_decay=hparams.lr_decay, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str.lower() == "adam":
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
                    if self.hparams.loss_function.lower() == "negative log likelihood":
                        logp = F.log_softmax(pred, 1)
                        loss = F.nll_loss(logp, labels)
                    elif self.hparams.loss_function.lower() == "cross entropy":
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
            if self.hparams.loss_function.lower() == "negative log likelihood":
                logp = F.log_softmax(pred, 1)
                loss = F.nll_loss(logp, labels)
            elif self.hparams.loss_function.lower() == "cross entropy":
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
                pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
                if self.hparams.loss_function.lower() == "negative log likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, labels)
                elif self.hparams.loss_function.lower() == "cross entropy":
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
    def Accuracy(actual, predicted, mantissa=4):
        """
        Computes the accuracy of the input predictions based on the input labels
        Parameters
        ----------
        predictions : list
            The input list of predictions.
        labels : list
            The input list of labels.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        dict
            A dictionary returning the accuracy information. This contains the following keys and values:
            - "accuracy" (float): The number of correct predictions divided by the length of the list.
            - "correct" (int): The number of correct predictions
            - "mask" (list): A boolean mask for correct vs. wrong predictions which can be used to filter the list of predictions
            - "size" (int): The size of the predictions list
            - "wrong" (int): The number of wrong predictions

        """
        if len(predicted) < 1 or len(actual) < 1 or not len(predicted) == len(actual):
            return None
        correct = 0
        mask = []
        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                correct = correct + 1
                mask.append(True)
            else:
                mask.append(False)
        size = len(predicted)
        wrong = len(predicted)- correct
        accuracy = round(float(correct) / float(len(predicted)), mantissa)
        return {"accuracy":accuracy, "correct":correct, "mask":mask, "size":size, "wrong":wrong}
    
    def ConfusionMatrix(actual, predicted, categories, renderer="notebook"):
        from sklearn import metrics
        return metrics.confusion_matrix(actual, predicted)

    @staticmethod
    def ClassifierBypath(path):
        """
        Returns the classifier found at the input file path.
        Parameters
        ----------
        path : str
            File path for the saved classifier.

        Returns
        -------
        DGL Classifier
            The classifier.

        """
        return torch.load(path)
    
    @staticmethod
    def DatasetByDGLGraphs(DGLGraphs, labels, key="node_attr"):
        """
        Returns a DGL Dataset from the input DGL graphs.

        Parameters
        ----------
        DGLGraphs : list
            The input list dgl graphs.
        labels : list
            The list of labels.
        key : str
            THe key used for the node attributes.

        Returns
        -------
        DGL.Dataset
            The creatred DGL dataset.

        """
        if isinstance(DGLGraphs, list) == False:
            DGLGraphs = [DGLGraphs]
        if isinstance(labels, list) == False:
            labels = [labels]
        return GraphDGL(DGLGraphs, labels, key)
    
    @staticmethod
    def DatasetByImportedCSV_NC(folderPath):
        """
        UNDER CONSTRUCTION. DO NOT USE.

        Parameters
        ----------
        folderPath : str
            The path to folder containing the input CSV files. In that folder there should be graphs.csv, edges.csv, and vertices.csv

        Returns
        -------
        DGLDataset
            The returns DGL dataset.

        """
        return dgl.data.CSVDataset(folderPath, force_reload=True)
    
    @staticmethod
    def DatasetBySample(name="ENZYMES"):
        """
        Returns a dataset from the samples database.

        Parameters
        ----------
        name : str
            The name of the sample dataset. This can be "ENZYMES", "DD", "COLLAB", or "MUTAG". It is case insensitive. The default is "ENZYMES".

        Returns
        -------
        GraphDGL
            The created DGL dataset.

        """
        name = name.upper()
        dataset = dgl.data.TUDataset(name)
        dgl_graphs, dgl_labels = zip(*[dataset[i] for i in range(len(dataset.graph_lists))])
        if name == 'ENZYMES':
            node_attr_key = 'node_attr'
        elif name == 'DD':
            node_attr_key = 'node_labels'
        elif name == 'COLLAB':
            node_attr_key = '_ID'
        elif name == 'MUTAG':
            node_attr_key = 'node_labels'
        else:
            raise NotImplementedError
        return GraphDGL(dgl_graphs, dgl_labels, node_attr_key)
    
    @staticmethod
    def DatasetBySample_NC(name="Cora"):
        """
        Returns the sample dataset as specified by the input sample name

        Parameters
        ----------
        name : str
            The name of the sample dataset to load. This can be "Cora", "Citeseer", or "Pubmed". It is case insensitive. The default is "Cora".

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        if name.lower() == 'cora':
            return [dgl.data.CoraGraphDataset(), 7]
        elif name.lower() == 'citeseer':
            return [dgl.data.CiteseerGraphDataset(), 6]
        elif name.lower() == 'pubmed':
            return [dgl.data.PubmedGraphDataset(), 3]
        else:
            raise NotImplementedError
    
    @staticmethod
    def DatasetGraphs_NC(dataset):
        """
        Return the DGL graphs found the in the input dataset.

        Parameters
        ----------
        dataset : DGLDataset
            The input dataset.

        Returns
        -------
        list
            The list of DGL graphs found in the input dataset.

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
        One-hot encodes the input item according to the input categories. One-Hot Encoding is a method to encode categorical variables to numerical data that Machine Learning algorithms can deal with. One-Hot encoding is most used during feature engineering for a ML Model. It converts categorical values into a new categorical column and assign a binary value of 1 or 0 to those columns. 
        
        Parameters
        ----------
        item : any
            The input item.
        categories : list
            The input list of categories.

        Returns
        -------
        list
            A one-hot encoded list of the input item according to the input categories.

        """
        returnList = []
        for i in range(len(categories)):
            if item == categories[i]:
                returnList.append(1)
            else:
                returnList.append(0)
        return returnList
    
    @staticmethod
    def ByGraph(graph, bidirectional=True, key=None, categories=[], node_attr_key="node_attr", tolerance=0.0001):
        """
        Returns a DGL graph by the input topologic graph.

        Parameters
        ----------
        graph : topologic.Graph
            The input topologic graph.
        bidirectional : bool , optional
            If set to True, the output DGL graph is forced to be bidirectional. The defaul is True.
        key : str
            The dictionary key where the node label is stored.
        categories : list
            The list of categories of node features.
        node_attr_key : str
            The dictionary key of the node attributes.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        DGL Graph
            The created DGL graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        
        graph_dict = {}
        vertices = Graph.Vertices(graph)
        edges = Graph.Edges(graph)
        graph_dict["num_nodes"] = len(vertices)
        graph_dict["src"] = []
        graph_dict["dst"] = []
        graph_dict["node_labels"] = {}
        graph_dict["node_features"] = []
        nodes = []
        graph_edges = []

        for i in range(len(vertices)):
            vDict = Topology.Dictionary(vertices[i])
            if key:
                vLabel = Dictionary.ValueAtKey(vDict, key)
            else:
                vLabel = ""
            graph_dict["node_labels"][i] = vLabel
            # appending tensor of onehotencoded feature for each node following index i
            graph_dict["node_features"].append(torch.tensor(DGL.OneHotEncode(vLabel, categories)))
            nodes.append(i)

        for i in range(len(edges)):
            e = edges[i]
            sv = e.StartVertex()
            ev = e.EndVertex()
            sn = nodes[Vertex.Index(vertex=sv, vertices=vertices, strict=False, tolerance=tolerance)]
            en = nodes[Vertex.Index(vertex=ev, vertices=vertices, strict=False, tolerance=tolerance)]
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
                              nodes_file_path, graph_id_header="graph_id",
                              graph_label_header="label", num_nodes_header="num_nodes", src_header="src",
                              dst_header="dst", node_label_header="label", node_attr_key="node_attr",
                              categories=[], bidirectional=True):
        """
        Returns DGL graphs according to the input CSV file paths.

        Parameters
        ----------
        graphs_file_path : str
            The file path to the grpahs CSV file.
        edges_file_path : str
            The file path to the edges CSV file.
        nodes_file_path : str
            The file path to the nodes CSV file.
        graph_id_header : str , optional
            The header string used to specify the graph id. The default is "graph_id".
        graph_label_header : str , optional
            The header string used to specify the graph label. The default is "label".
        num_nodes_header : str , optional
            The header string used to specify the number of nodes. The default is "num_nodes".
        src_header : str , optional
            The header string used to specify the source of edges. The default is "src".
        dst_header : str , optional
            The header string used to specify the destination of edges. The default is "dst".
        node_label_header : str , optional
            The header string used to specify the node label. The default is "label".
        node_attr_key : str , optional
            The key string used to specify the node attributes. The default is "node_attr".
        categories : list
            The list of categories.
        bidirectional : bool , optional
            If set to True, the output DGL graph is forced to be bi-directional. The default is True.

        Returns
        -------
        list
            The list of DGL graphs found in the input CSV files.

        """

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
        return {"graphs":dgl_graphs, "labels":labels}

    @staticmethod
    def CategoryDistribution(labels, categories=None, mantissa=4):
        """
        Returns the category distribution in the input list of labels. This is useful to determine if the dataset is balanced or not.

        Parameters
        ----------
        labels : list
            The input list of labels.
        categories : list , optional
            The list of node categories expected in the imported DGCNN file. If not specified, the categories are computed directly from the labels. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        dict
            A dictionary object that contains the categories and their corresponding ratios. The dictionary has the following keys and values:
            - "categories" (list): The list of categories.
            - "ratios" (list): The list of ratios of each category as found in the input list of labels.

        """
        if not categories:
            categories = list(set(labels))
        ratios = []
        for category in categories:
            ratios.append(round(float(labels.count(category))/float(len(labels)), mantissa))
        return {"categories":[categories], "ratios":[ratios]}

    @staticmethod
    def ByImportedDGCNN(file_path, categories=[], bidirectional=True):
        """
        Returns the Graphs from the imported DGCNN file.

        Parameters
        ----------
        file_path : str
            The file path to the DGCNN text file.
        categories : list
            The list of node categories expected in the imported DGCNN file. This is used to one-hot-encode the node features.
        bidirectional : bool , optional
            If set to True, the output DGL graph is forced to be bi-directional. The defaults is True.

        Returns
        -------
        dict
            A dictionary object that contains the imported graphs and their corresponding labels. The dictionary has the following keys and values:
            - "graphs" (list): The list of DGL graphs
            - "labels" (list): The list of graph labels

        """
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
        return {"graphs":graphs, "labels":labels}
    
    @staticmethod
    def EdgeData_NC(dgl_graph):
        """
        Returns the edge data found in the input DGL graph
        Parameters
        ----------
        dgl_graph : DGL Graph
            The input DGL graph.

        Returns
        -------
        edge data
            The edge data.

        """
        return dgl_graph.edata
    
    @staticmethod
    def NodeData_NC(dgl_graph):
        """
        Returns the node data found in the input dgl_graph

        Parameters
        ----------
        dgl_graph : DGL graph
            The input DGL graph.

        Returns
        -------
        node data
            The node data.

        """
        return dgl_graph.ndata
    
    @staticmethod
    def Hyperparameters(optimizer, cv_type="Holdout", split=0.2, k_folds=5,
                           hl_widths=[32], conv_layer_type="SAGEConv", pooling="AvgPooling",
                           batch_size=1, epochs=1, use_gpu=False, loss_function="Cross Entropy",
                           classifier_path="", results_path=""):
        """
        Creates a hyperparameters object based on the input settings.

        Parameters
        ----------
        optimizer : Optimizer
            The desired optimizer.
        cv_type : str , optional
            The desired cross-validation method. This can be "Holdout" or "K-Fold". It is case-insensitive. The default is "Holdout".
        split : float , optional
            The desired split between training and testing. 0.2 means that 80% of the data is used for training and 20% of the data is used for testing. The default is 0.20.
        k_folds : int , optional
            The desired number of k-folds. The default is 5.
        hl_widths : list , optional
            The list of hidden layer widths. A list of [16, 32, 16] means that the model will have 3 hidden layers with number of neurons in each being 16, 32, 16 respectively from input to output. The default is [32].
        conv_layer_type : str , optional
            THe desired type of the convultion layer. The options are "Classic", "GraphConv", "GINConv", "SAGEConv", "TAGConv", "DGN". It is case insensitive. The default is "SAGEConv".
        pooling : str , optional
            The desired type of pooling. The options are "AvgPooling", "MaxPooling", or "SumPooling". It is case insensitive. The default is "AvgPooling".
        batch_size : int , optional
            The desired batch size. The default is 1.
        epochs : int , optional
            The desired number of epochs. The default is 1.
        use_gpu : bool , optional
            If set to True, the model will attempt to use the GPU. The default is False.
        loss_function : str , optional
            The desired loss function. The optionals are "Cross-Entropy" or "Negative Log Likelihood". It is case insensitive. The default is "Cross-Entropy".
        classifier_path : str
            The file path at which to save the trained classifier.
        results_path : str
            The file path at which to save the training and testing results.

        Returns
        -------
        Hyperparameters
            The created hyperparameters object.

        """
        
        if optimizer['name'].lower() == "adadelta":
            name = "Adadelta"
        elif optimizer['name'].lower() == "adagrad":
            name = "Adagrad"
        elif optimizer['name'].lower() == "adam":
            name = "Adam"
        # Classifier: Make sure the file extension is .pt
        ext = classifier_path[len(classifier_path)-3:len(classifier_path)]
        if ext.lower() != ".pt":
            classifier_path = classifier_path+".pt"
        # Results: Make sure the file extension is .csv
        ext = results_path[len(results_path)-4:len(results_path)]
        if ext.lower() != ".csv":
            results_path = results_path+".csv"
        return Hparams(name,
                       optimizer['amsgrad'],
                       optimizer['betas'],
                       optimizer['eps'],
                       optimizer['lr'],
                       optimizer['lr_decay'],
                       optimizer['maximize'],
                       optimizer['rho'],
                       optimizer['weight_decay'],
                       cv_type,
                       split,
                       k_folds,
                       hl_widths,
                       conv_layer_type,
                       pooling,
                       batch_size,
                       epochs,
                       use_gpu,
                       loss_function,
                       classifier_path,
                       results_path)


    @staticmethod
    def Optimizer(name="Adam", amsgrad=True, betas=(0.9,0.999), eps=0.000001, lr=0.001, maximize=False, weightDecay=0.0, rho=0.9, lr_decay=0.0):
        """
        Returns the parameters of the optimizer

        Parameters
        ----------
        amsgrad : bool . optional.
            DESCRIPTION. The default is True.
        betas : tuple . optional
            DESCRIPTION. The default is (0.9, 0.999)
        eps : float . optional.
            DESCRIPTION. The default is 0.000001
        lr : float
            DESCRIPTION. The default is 0.001
        maximize : float . optional
            DESCRIPTION. The default is False.
        weightDecay : float . optional
            DESCRIPTION. The default is 0.0.

        Returns
        -------
        dict
            The dictionary of the optimizer parameters. The dictionary contains the following keys and values:
            - "name" (str): The name of the optimizer
            - "amsgrad" (bool):
            - "betas" (tuple):
            - "eps" (float):
            - "lr" (float):
            - "maximize" (bool):
            - weightDecay (float):

        """
        return {"name":name, "amsgrad":amsgrad, "betas":betas, "eps":eps, "lr": lr, "maximize":maximize, "weight_decay":weightDecay, "rho":rho, "lr_decay":lr_decay}

    @staticmethod
    def Show(data,
             labels,
             title="Untitled",
             x_title="X Axis",
             x_spacing=1.0,
             y_title="Y Axis",
             y_spacing=0.1,
             use_markers=False,
             chart_type="Line",
             renderer = "notebook"):
        """
        Shows the data in a plolty graph.

        Parameters
        ----------
        data : list
            The data to display.
        data_labels : list
            The labels to use for the data.
        title : str , optional
            The chart title. The default is "Untitled".
        x_title : str , optional
            The X-axis title. The default is "Epochs".
        x_spacing : float , optional
            The X-axis spacing. The default is 1.0.
        y_title : str , optional
            The Y-axis title. The default is "Accuracy and Loss".
        y_spacing : float , optional
            The Y-axis spacing. The default is 0.1.
        use_markers : bool , optional
            If set to True, markers will be displayed. The default is False.
        chart_type : str , optional
            The desired type of chart. The options are "Line", "Bar", or "Scatter". It is case insensitive. The default is "Line".
        renderer : str , optional
            The desired plotly renderer. The default is "notebook".

        Returns
        -------
        None.

        """
        from topologicpy.Plotly import Plotly
        if isinstance(data[labels[0]][0], int):
            xAxis_list = list(range(1,data[labels[0]][0]+1))
        else:
            xAxis_list = data[labels[0]][0]
        plot_data = [xAxis_list]
        for i in range(1,len(labels)):
            plot_data.append(data[labels[i]][0][:len(xAxis_list)])

        dlist = list(map(list, zip(*plot_data)))
        df = pd.DataFrame(dlist, columns=labels)
        fig = Plotly.FigureByDataFrame(df,
                                       labels=labels,
                                       title=title,
                                       x_title=x_title,
                                       x_spacing=x_spacing,
                                       y_title=y_title,
                                       y_spacing=y_spacing,
                                       use_markers=use_markers,
                                       chart_type=chart_type)
        Plotly.Show(fig, renderer=renderer)
        
    @staticmethod
    def Predict(dataset, classifier, node_attr_key="node_attr"):
        """
        Predicts the label of the input dataset.

        Parameters
        ----------
        dataset : DGLDataset
            The input DGL dataset.
        classifier : Classifier
            The input trained classifier.
        node_attr_key : str , optional
            The key used for node attributes. The default is "node_attr".

        Returns
        -------
        dict
            Dictionary containing labels and probabilities. The included keys and values are:
            - "labels" (list): the list of predicted labels
            - "probabilities" (list): the list of probabilities that the label is one of the categories.

        """
        labels = []
        probabilities = []
        for item in dataset:
            graph = item[0]
            pred = classifier(graph, graph.ndata[node_attr_key].float())
            labels.append(pred.argmax(1).item())
            probability = (torch.nn.functional.softmax(pred, dim=1).tolist())
            probability = probability[0]
            temp_probability = []
            for p in probability:
                temp_probability.append(round(p, 3))
            probabilities.append(temp_probability)
        return {"labels":labels, "probabilities":probabilities}
    
    @staticmethod
    def Predict_NC(dataset, classifier):
        """
        Predicts the node labels found in the input dataset using the input classifier.

        Parameters
        ----------
        dataset : DGLDataset
            The input DGL Dataset.

        classifier : Classifier
            The input classifier.
        
        Returns
        -------
        dict
            A dictionary containing all the results.

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
    def Train(hparams, trainingDataset, validationDataset=None, overwrite=True):
        """
        Trains a neural network classifier.

        Parameters
        ----------
        hparams : HParams
            The input hyperparameters 
        trainingDataset : DGLDataset
            The input training dataset.
        validationDataset : DGLDataset
            The input validation dataset. This is required only if the cross validation type (cv_type) is "K-Fold"

        classifier : Classifier
            The input classifier.
        
        Returns
        -------
        dict
            A dictionary containing all the results.

        """
        from topologicpy.Helper import Helper
        import time
        import datetime
        start = time.time()
        if hparams.cv_type.lower() == "holdout":
            classifier = ClassifierSplit(hparams, trainingDataset)
            classifier.train()
            accuracy = classifier.test()
        elif hparams.cv_type.lower() == "k-fold":
            classifier = ClassifierKFold(hparams, trainingDataset, validationDataset)
            classifier.train()
            classifier.validate()

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
        utcnow = datetime.datetime.utcnow()
        timestamp_str = "UTC-"+str(utcnow.year)+"-"+str(utcnow.month)+"-"+str(utcnow.day)+"-"+str(utcnow.hour)+"-"+str(utcnow.minute)+"-"+str(utcnow.second)
        epoch_list = list(range(1,classifier.hparams.epochs+1))
        d2 = [[timestamp_str], [duration], [classifier.hparams.optimizer_str], [classifier.hparams.cv_type], [classifier.hparams.split], [classifier.hparams.k_folds], classifier.hparams.hl_widths, [classifier.hparams.conv_layer_type], [classifier.hparams.pooling], [classifier.hparams.lr], [classifier.hparams.batch_size], epoch_list, classifier.training_accuracy_list, classifier.testing_accuracy_list, classifier.training_loss_list, classifier.testing_loss_list]
        d2 = Helper.Iterate(d2)
        d2 = Helper.Transpose(d2)
    
        data = {'TimeStamp': "UTC-"+str(utcnow.year)+"-"+str(utcnow.month)+"-"+str(utcnow.day)+"-"+str(utcnow.hour)+"-"+str(utcnow.minute)+"-"+str(utcnow.second),
                'Duration': [duration],
                'Optimizer': [classifier.hparams.optimizer_str],
                'CV Type': [classifier.hparams.cv_type],
                'Split': [classifier.hparams.split],
                'K-Folds': [classifier.hparams.k_folds],
                'HL Widths': [classifier.hparams.hl_widths],
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

        df = pd.DataFrame(d2, columns= ['TimeStamp', 'Duration', 'Optimizer', 'CV Type', 'Split', 'K-Folds', 'HL Widths', 'Conv Layer Type', 'Pooling', 'Learning Rate', 'Batch Size', 'Epochs', 'Training Accuracy', 'Testing Accuracy', 'Training Loss', 'Testing Loss'])
        if classifier.hparams.results_path:
            if overwrite:
                df.to_csv(classifier.hparams.results_path, mode='w+', index = False, header=True)
            else:
                df.to_csv(classifier.hparams.results_path, mode='a', index = False, header=False)
        return data

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
        if hparams.optimizer_str.lower() == "adadelta":
            optimizer = torch.optim.Adadelta(model.parameters(), eps=hparams.eps, 
                                                lr=hparams.lr, rho=hparams.rho, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str.lower() == "adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), eps=hparams.eps, 
                                                lr=hparams.lr, lr_decay=hparams.lr_decay, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str.lower() == "adam":
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
                if hparams.loss_function.lower() == "negative log likelihood":
                    logp = F.log_softmax(logits[train_mask], 1)
                    loss = F.nll_loss(logp, labels[train_mask])
                elif hparams.loss_function.lower() == "cross entropy":
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
        model = GCN_Classic(graphs[i].ndata['feat'].shape[1], hparams.hl_widths, numLabels)
        final_model, predictions = DGL.Train_NC(graphs, model, hparams)
        # Save the entire model
        if hparams.checkpoint_path is not None:
            torch.save(final_model, hparams.checkpoint_path)
        return final_model