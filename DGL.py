
import topologicpy
import topologic
from topologicpy.Dictionary import Dictionary
import os
import random
import time
from datetime import datetime
import copy
import sys
import subprocess

try:
    import numpy as np
except:
    call = [sys.executable, '-m', 'pip', 'install', 'numpy', '-t', sys.path[0]]
    subprocess.run(call)
    try:
        import numpy as np
    except:
        print("DGL - Error: Could not import numpy.")
try:
    import pandas as pd
except:
    call = [sys.executable, '-m', 'pip', 'install', 'pandas', '-t', sys.path[0]]
    subprocess.run(call)
    try:
        import pandas as pd
    except:
        print("DGL - Error: Could not import pandas")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.utils.data import DataLoader, ConcatDataset
except:
    call = [sys.executable, '-m', 'pip', 'install', 'torch', '-t', sys.path[0]]
    subprocess.run(call)
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data.sampler import SubsetRandomSampler
        from torch.utils.data import DataLoader, ConcatDataset
    except:
        print("DGL - Error: Could not import torch")
try:
    import dgl
    from dgl.data import DGLDataset
    from dgl.dataloading import GraphDataLoader
    from dgl.nn import GINConv, GraphConv, SAGEConv, TAGConv
    from dgl import save_graphs, load_graphs
except:
    call = [sys.executable, '-m', 'pip', 'install', 'dgl', 'dglgo', '-f', 'https://data.dgl.ai/wheels/repo.html', '--upgrade', '-t', sys.path[0]]
    subprocess.run(call)
    try:
        import dgl
        from dgl.data import DGLDataset
        from dgl.nn import GraphConv
        from dgl import save_graphs, load_graphs
    except:
        print("DGL - Error: Could not import dgl")
try:
    import sklearn
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
except:
    call = [sys.executable, '-m', 'pip', 'install', 'scikit-learn', '-t', sys.path[0]]
    subprocess.run(call)
    try:
        import sklearn
        from sklearn.model_selection import KFold
        from sklearn.metrics import accuracy_score
    except:
        print("DGL - Error: Could not import sklearn")
try:
    from tqdm.auto import tqdm
except:
    call = [sys.executable, '-m', 'pip', 'install', 'tqdm', '-t', sys.path[0]]
    subprocess.run(call)
    try:
        from tqdm.auto import tqdm
    except:
        print("DGL - Error: Could not import tqdm")

class _Dataset(DGLDataset):
    def __init__(self, graphs, labels, node_attr_key):
        super().__init__(name='GraphDGL')
        self.graphs = graphs
        if isinstance(labels[0], int):
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = torch.DoubleTensor(labels)
        self.node_attr_key = node_attr_key
        # as all graphs have same length of node features then we get dim_nfeats from first graph in the list
        self.dim_nfeats = graphs[0].ndata[node_attr_key].shape[1]
        # to get the number of classes for graphs
        self.gclasses = len(set(labels))

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

class _Hparams:
    def __init__(self, model_type="ClassifierHoldout", optimizer_str="Adam", amsgrad=False, betas=(0.9, 0.999), eps=1e-6, lr=0.001, lr_decay= 0, maximize=False, rho=0.9, weight_decay=0, cv_type="Holdout", split=[0.8,0.1, 0.1], k_folds=5, hl_widths=[32], conv_layer_type='SAGEConv', pooling="AvgPooling", batch_size=32, epochs=1, 
                 use_gpu=False, loss_function="Cross Entropy"):
        """
        Parameters
        ----------
        cv : str
            A string to define the method of cross-validation
            "Holdout": Holdout
            "K-Fold": K-Fold cross validation
        k_folds : int
            An int value in the range of 2 to X to define the number of k-folds for cross-validation. Default is 5.
        split : list
            A list of three item in the range of 0 to 1 to define the split of train,
            validate, and test data. A default value of [0.8,0.1,0.1] means 80% of data will be
            used for training, 10% will be used for validation, and the remaining 10% will be used for training
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
        use_GPU : use the GPU. Otherwise, use the CPU

        Returns
        -------
        None

        """
        
        self.model_type = model_type
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

class _Classic(nn.Module):
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
        super(_Classic, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = nn.ModuleList()
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

class _ClassicReg(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(_ClassicReg, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = nn.ModuleList()
        dim = [in_feats] + h_feats
        for i in range(1, len(dim)):
            self.list_of_layers.append(GraphConv(dim[i-1], dim[i]))
        self.final = nn.Linear(dim[-1], 1)

    def forward(self, g, in_feat):
        h = in_feat
        for i in range(len(self.list_of_layers)):
            h = self.list_of_layers[i](g, h)
            h = F.relu(h)
        h = self.final(h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')
    
class _GINConv(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, pooling):
        super(_GINConv, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = nn.ModuleList()
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

class _GraphConv(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, pooling):
        super(_GraphConv, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = nn.ModuleList()
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

class _SAGEConv(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, pooling):
        super(_SAGEConv, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = nn.ModuleList()
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

class _TAGConv(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, pooling):
        super(_TAGConv, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = nn.ModuleList()
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


class _GraphConvReg(nn.Module):
    def __init__(self, in_feats, h_feats, pooling):
        super(_GraphConvReg, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) !=0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = nn.ModuleList()
        dim = [in_feats] + h_feats

        # Convolution (Hidden) Layers
        for i in range(1, len(dim)):
            self.list_of_layers.append(GraphConv(dim[i-1], dim[i]))

        # Final Layer
        self.final = nn.Linear(dim[-1], 1)

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


class _RegressorHoldout:
    def __init__(self, hparams, trainingDataset, validationDataset=None, testingDataset=None):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        self.trainingDataset = trainingDataset
        self.validationDataset = validationDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        if hparams.conv_layer_type.lower() == 'classic':
            self.model = _ClassicReg(trainingDataset.dim_nfeats, hparams.hl_widths).to(device)
        elif hparams.conv_layer_type.lower() == 'ginconv':
            self.model = _GINConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            1, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'graphconv':
            self.model = _GraphConvReg(trainingDataset.dim_nfeats, hparams.hl_widths, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'sageconv':
            self.model = _SAGEConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            1, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'tagconv':
            self.model = _TAGConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            1, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'gcn':
            self.model = _ClassicReg(trainingDataset.dim_nfeats, hparams.hl_widths).to(device)
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
        self.validation_loss_list = []
        self.node_attr_key = trainingDataset.node_attr_key

        # train, validate, test split
        num_train = int(len(trainingDataset) * (hparams.split[0]))
        num_validate = int(len(trainingDataset) * (hparams.split[1]))
        num_test = len(trainingDataset) - num_train - num_validate
        idx = torch.randperm(len(trainingDataset))
        train_sampler = SubsetRandomSampler(idx[:num_train])
        validate_sampler = SubsetRandomSampler(idx[num_train:num_train+num_validate])
        test_sampler = SubsetRandomSampler(idx[num_train+num_validate:num_train+num_validate+num_test])
        
        if validationDataset:
            self.train_dataloader = GraphDataLoader(trainingDataset, 
                                                    batch_size=hparams.batch_size,
                                                    drop_last=False)
            self.validate_dataloader = GraphDataLoader(validationDataset,
                                                    batch_size=hparams.batch_size,
                                                    drop_last=False)
        else:
            self.train_dataloader = GraphDataLoader(trainingDataset, sampler=train_sampler, 
                                                    batch_size=hparams.batch_size,
                                                    drop_last=False)
            self.validate_dataloader = GraphDataLoader(trainingDataset, sampler=validate_sampler,
                                                    batch_size=hparams.batch_size,
                                                    drop_last=False)
        
        if testingDataset:
            self.test_dataloader = GraphDataLoader(testingDataset,
                                                    batch_size=len(testingDataset),
                                                    drop_last=False)
        else:
            self.test_dataloader = GraphDataLoader(trainingDataset, sampler=test_sampler,
                                                    batch_size=hparams.batch_size,
                                                    drop_last=False)

    def train(self):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        # Init the loss and accuracy reporting lists
        self.training_loss_list = []
        self.validation_loss_list = []
        

        # Run the training loop for defined number of epochs
        for _ in tqdm(range(self.hparams.epochs), desc='Epochs', total=self.hparams.epochs, leave=False):
            # Iterate over the DataLoader for training data
            for batched_graph, labels in tqdm(self.train_dataloader, desc='Training', leave=False):
                # Make sure the model is in training mode
                self.model.train()
                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
                # Compute loss
                loss = F.mse_loss(torch.flatten(pred), labels.float())

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()

            self.training_loss_list.append(torch.sqrt(loss).item())
            self.validate()
            self.validation_loss_list.append(torch.sqrt(self.validation_loss).item())


    def validate(self):
        device = torch.device("cpu")
        self.model.eval()
        for batched_graph, labels in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
            loss = F.mse_loss(torch.flatten(pred), labels.float())
        self.validation_loss = loss
    
    def test(self):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        self.model.eval()
        for batched_graph, labels in tqdm(self.test_dataloader, desc='Testing', leave=False):
            pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
            loss = F.mse_loss(torch.flatten(pred), labels.float())
        self.testing_loss = torch.sqrt(loss).item()
    
    def save(self, path):
        if path:
            # Make sure the file extension is .pt
            ext = path[len(path)-3:len(path)]
            if ext.lower() != ".pt":
                path = path+".pt"
            torch.save(self.model, path)

class _RegressorKFold:
    def __init__(self, hparams, trainingDataset, testingDataset=None):
        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        self.losses = []
        self.min_loss = 0
        # at beginning of the script
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        if hparams.conv_layer_type.lower() == 'classic':
            self.model = _ClassicReg(trainingDataset.dim_nfeats, hparams.hl_widths).to(device)
        elif hparams.conv_layer_type.lower() == 'ginconv':
            self.model = _GINConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            1, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'graphconv':
            self.model = _GraphConvReg(trainingDataset.dim_nfeats, hparams.hl_widths, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'sageconv':
            self.model = _SAGEConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            1, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'tagconv':
            self.model = _TAGConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            1, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'gcn':
            self.model = _ClassicReg(trainingDataset.dim_nfeats, hparams.hl_widths).to(device)
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
        self.validation_loss_list = []
        self.node_attr_key = trainingDataset.node_attr_key

        # train, validate, test split
        num_train = int(len(trainingDataset) * (hparams.split[0]))
        num_validate = int(len(trainingDataset) * (hparams.split[1]))
        num_test = len(trainingDataset) - num_train - num_validate
        idx = torch.randperm(len(trainingDataset))
        test_sampler = SubsetRandomSampler(idx[num_train+num_validate:num_train+num_validate+num_test])
        
        if testingDataset:
            self.test_dataloader = GraphDataLoader(testingDataset,
                                                    batch_size=len(testingDataset),
                                                    drop_last=False)
        else:
            self.test_dataloader = GraphDataLoader(trainingDataset, sampler=test_sampler,
                                                    batch_size=hparams.batch_size,
                                                    drop_last=False)
    
    def reset_weights(self):
        '''
        Try resetting model weights to avoid
        weight leakage.
        '''
        device = torch.device("cpu")
        if self.hparams.conv_layer_type.lower() == 'classic':
            self.model = _ClassicReg(self.trainingDataset.dim_nfeats, self.hparams.hl_widths).to(device)
        elif self.hparams.conv_layer_type.lower() == 'ginconv':
            self.model = _GINConv(self.trainingDataset.dim_nfeats, self.hparams.hl_widths, 
                            1, self.hparams.pooling).to(device)
        elif self.hparams.conv_layer_type.lower() == 'graphconv':
            self.model = _GraphConvReg(self.trainingDataset.dim_nfeats, self.hparams.hl_widths, self.hparams.pooling).to(device)
        elif self.hparams.conv_layer_type.lower() == 'sageconv':
            self.model = _SAGEConv(self.trainingDataset.dim_nfeats, self.hparams.hl_widths, 
                            1, self.hparams.pooling).to(device)
        elif self.hparams.conv_layer_type.lower() == 'tagconv':
            self.model = _TAGConv(self.trainingDataset.dim_nfeats, self.hparams.hl_widths, 
                            1, self.hparams.pooling).to(device)
        elif self.hparams.conv_layer_type.lower() == 'gcn':
            self.model = _ClassicReg(self.trainingDataset.dim_nfeats, self.hparams.hl_widths).to(device)
        else:
            raise NotImplementedError

        if self.hparams.optimizer_str.lower() == "adadelta":
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), eps=self.hparams.eps, 
                                            lr=self.hparams.lr, rho=self.hparams.rho, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_str.lower() == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), eps=self.hparams.eps, 
                                            lr=self.hparams.lr, lr_decay=self.hparams.lr_decay, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_str.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=self.hparams.amsgrad, betas=self.hparams.betas, eps=self.hparams.eps, 
                                            lr=self.hparams.lr, maximize=self.hparams.maximize, weight_decay=self.hparams.weight_decay)

    
    
    
    def train(self):
        device = torch.device("cpu")

        # The number of folds (This should come from the hparams)
        k_folds = self.hparams.k_folds

        # Init the loss and accuracy reporting lists
        self.training_loss_list = []
        self.validation_loss_list = []

        # Set fixed random number seed
        torch.manual_seed(42)
        
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=k_folds, shuffle=True)

        models = []
        weights = []
        losses = []
        train_dataloaders = []
        validate_dataloaders = []

        # K-fold Cross-validation model evaluation
        for fold, (train_ids, validate_ids) in tqdm(enumerate(kfold.split(self.trainingDataset)), desc="Fold", initial=1, total=k_folds, leave=False):
            epoch_training_loss_list = []
            epoch_validation_loss_list = []
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            validate_subsampler = torch.utils.data.SubsetRandomSampler(validate_ids)

            # Define data loaders for training and testing data in this fold
            self.train_dataloader = GraphDataLoader(self.trainingDataset, sampler=train_subsampler, 
                                                batch_size=self.hparams.batch_size,
                                                drop_last=False)
            self.validate_dataloader = GraphDataLoader(self.trainingDataset, sampler=validate_subsampler,
                                                batch_size=self.hparams.batch_size,
                                                drop_last=False)
            # Init the neural network
            self.reset_weights()

            # Run the training loop for defined number of epochs
            best_rmse = np.inf
            # Run the training loop for defined number of epochs
            for _ in tqdm(range(self.hparams.epochs), desc='Epochs', total=self.hparams.epochs, initial=1, leave=False):
                # Iterate over the DataLoader for training data
                for batched_graph, labels in tqdm(self.train_dataloader, desc='Training', leave=False):
                    # Make sure the model is in training mode
                    self.model.train()
                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Perform forward pass
                    pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
                    # Compute loss
                    loss = F.mse_loss(torch.flatten(pred), labels.float())

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    self.optimizer.step()


                epoch_training_loss_list.append(torch.sqrt(loss).item())
                self.validate()
                epoch_validation_loss_list.append(torch.sqrt(self.validation_loss).item())

            models.append(self.model)
            weights.append(copy.deepcopy(self.model.state_dict()))
            losses.append(torch.sqrt(self.validation_loss).item())
            train_dataloaders.append(self.train_dataloader)
            validate_dataloaders.append(self.validate_dataloader)
            self.training_loss_list.append(epoch_training_loss_list)
            self.validation_loss_list.append(epoch_validation_loss_list)
        self.losses = losses
        min_loss = min(losses)
        self.min_loss = min_loss
        ind = losses.index(min_loss)
        self.model = models[ind]
        self.model.load_state_dict(weights[ind])
        self.model.eval()
        self.training_loss_list = self.training_loss_list[ind]
        self.validation_loss_list = self.validation_loss_list[ind]

    def validate(self):
        device = torch.device("cpu")
        self.model.eval()
        for batched_graph, labels in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
            loss = F.mse_loss(torch.flatten(pred), labels.float())
        self.validation_loss = loss
    
    def test(self):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        #self.model.eval()
        for batched_graph, labels in tqdm(self.test_dataloader, desc='Testing', leave=False):
            pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
            loss = F.mse_loss(torch.flatten(pred), labels.float())
        self.testing_loss = torch.sqrt(loss).item()
    
    def save(self, path):
        if path:
            # Make sure the file extension is .pt
            ext = path[len(path)-3:len(path)]
            if ext.lower() != ".pt":
                path = path+".pt"
            torch.save(self.model, path)

class _ClassifierHoldout:
    def __init__(self, hparams, trainingDataset, validationDataset=None, testingDataset=None):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        self.trainingDataset = trainingDataset
        self.validationDataset = validationDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        if hparams.conv_layer_type.lower() == 'classic':
            self.model = _Classic(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses).to(device)
        elif hparams.conv_layer_type.lower() == 'ginconv':
            self.model = _GINConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'graphconv':
            self.model = _GraphConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'sageconv':
            self.model = _SAGEConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'tagconv':
            self.model = _TAGConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'gcn':
            self.model = _Classic(trainingDataset.dim_nfeats, hparams.hl_widths, 
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
        self.validation_loss_list = []
        self.training_accuracy_list = []
        self.validation_accuracy_list = []
        self.node_attr_key = trainingDataset.node_attr_key

        # train, validate, test split
        num_train = int(len(trainingDataset) * (hparams.split[0]))
        num_validate = int(len(trainingDataset) * (hparams.split[1]))
        num_test = len(trainingDataset) - num_train - num_validate
        idx = torch.randperm(len(trainingDataset))
        train_sampler = SubsetRandomSampler(idx[:num_train])
        validate_sampler = SubsetRandomSampler(idx[num_train:num_train+num_validate])
        test_sampler = SubsetRandomSampler(idx[num_train+num_validate:num_train+num_validate+num_test])
        
        if validationDataset:
            self.train_dataloader = GraphDataLoader(trainingDataset, 
                                                    batch_size=hparams.batch_size,
                                                    drop_last=False)
            self.validate_dataloader = GraphDataLoader(validationDataset,
                                                    batch_size=hparams.batch_size,
                                                    drop_last=False)
        else:
            self.train_dataloader = GraphDataLoader(trainingDataset, sampler=train_sampler, 
                                                    batch_size=hparams.batch_size,
                                                    drop_last=False)
            self.validate_dataloader = GraphDataLoader(trainingDataset, sampler=validate_sampler,
                                                    batch_size=hparams.batch_size,
                                                    drop_last=False)
        
        if testingDataset:
            self.test_dataloader = GraphDataLoader(testingDataset,
                                                    batch_size=len(testingDataset),
                                                    drop_last=False)
        else:
            self.test_dataloader = GraphDataLoader(trainingDataset, sampler=test_sampler,
                                                    batch_size=hparams.batch_size,
                                                    drop_last=False)
    def train(self):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        # Init the loss and accuracy reporting lists
        self.training_accuracy_list = []
        self.training_loss_list = []
        self.validation_accuracy_list = []
        self.validation_loss_list = []

        # Run the training loop for defined number of epochs
        for _ in tqdm(range(self.hparams.epochs), desc='Epochs', initial=1, leave=False):
            temp_loss_list = []
            temp_acc_list = []
            # Iterate over the DataLoader for training data
            for batched_graph, labels in tqdm(self.train_dataloader, desc='Training', leave=False):
                # Make sure the model is in training mode
                self.model.train()

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
                temp_acc_list.append(accuracy_score(labels, pred.argmax(1)))

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()

            self.training_accuracy_list.append(np.mean(temp_acc_list).item())
            self.training_loss_list.append(np.mean(temp_loss_list).item())
            self.validate()
            self.validation_accuracy_list.append(self.validation_accuracy)
            self.validation_loss_list.append(self.validation_loss)
        
    def validate(self):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        temp_loss_list = []
        temp_acc_list = []
        self.model.eval()
        for batched_graph, labels in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float()).to(device)
            if self.hparams.loss_function.lower() == "negative log likelihood":
                logp = F.log_softmax(pred, 1)
                loss = F.nll_loss(logp, labels)
            elif self.hparams.loss_function.lower() == "cross entropy":
                loss = F.cross_entropy(pred, labels)
            temp_loss_list.append(loss.item())
            temp_acc_list.append(accuracy_score(labels, pred.argmax(1)))
        self.validation_accuracy = np.mean(temp_acc_list).item()
        self.validation_loss = np.mean(temp_loss_list).item()
    
    def test(self):
        if self.test_dataloader:
            temp_loss_list = []
            temp_acc_list = []
            self.model.eval()
            for batched_graph, labels in tqdm(self.test_dataloader, desc='Testing', leave=False):
                pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float())
                if self.hparams.loss_function.lower() == "negative log likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, labels)
                elif self.hparams.loss_function.lower() == "cross entropy":
                    loss = F.cross_entropy(pred, labels)
                temp_loss_list.append(loss.item())
                temp_acc_list.append(accuracy_score(labels, pred.argmax(1)))
            self.testing_accuracy = np.mean(temp_acc_list).item()
            self.testing_loss = np.mean(temp_loss_list).item()
            
    def save(self, path):
        if path:
            # Make sure the file extension is .pt
            ext = path[len(path)-3:len(path)]
            if ext.lower() != ".pt":
                path = path+".pt"
            torch.save(self.model, path)

class _ClassifierKFold:
    def __init__(self, hparams, trainingDataset, testingDataset=None):
        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        self.testing_accuracy = 0
        self.accuracies = []
        self.max_accuracy = 0
        # at beginning of the script
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        if hparams.conv_layer_type.lower() == 'classic':
            self.model = _Classic(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses).to(device)
        elif hparams.conv_layer_type.lower() == 'ginconv':
            self.model = _GINConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'graphconv':
            self.model = _GraphConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'sageconv':
            self.model = _SAGEConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
                            trainingDataset.gclasses, hparams.pooling).to(device)
        elif hparams.conv_layer_type.lower() == 'tagconv':
            self.model = _TAGConv(trainingDataset.dim_nfeats, hparams.hl_widths, 
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
        self.validation_loss_list = []
        self.training_accuracy_list = []
        self.validation_accuracy_list = []
        self.node_attr_key = trainingDataset.node_attr_key

    
    def reset_weights(self):
        '''
        Try resetting model weights to avoid
        weight leakage.
        '''
        device = torch.device("cpu")
        if self.hparams.conv_layer_type.lower() == 'classic':
            self.model = _Classic(self.trainingDataset.dim_nfeats, self.hparams.hl_widths, 
                            self.trainingDataset.gclasses).to(device)
        elif self.hparams.conv_layer_type.lower() == 'ginconv':
            self.model = _GINConv(self.trainingDataset.dim_nfeats, self.hparams.hl_widths, 
                            self.trainingDataset.gclasses, self.hparams.pooling).to(device)
        elif self.hparams.conv_layer_type.lower() == 'graphconv':
            self.model = _GraphConv(self.trainingDataset.dim_nfeats, self.hparams.hl_widths, 
                            self.trainingDataset.gclasses, self.hparams.pooling).to(device)
        elif self.hparams.conv_layer_type.lower() == 'sageconv':
            self.model = _SAGEConv(self.trainingDataset.dim_nfeats, self.hparams.hl_widths, 
                            self.trainingDataset.gclasses, self.hparams.pooling).to(device)
        elif self.hparams.conv_layer_type.lower() == 'tagconv':
            self.model = _TAGConv(self.trainingDataset.dim_nfeats, self.hparams.hl_widths, 
                            self.trainingDataset.gclasses, self.hparams.pooling).to(device)
        else:
            raise NotImplementedError
        if self.hparams.optimizer_str.lower() == "adadelta":
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), eps=self.hparams.eps, 
                                            lr=self.hparams.lr, rho=self.hparams.rho, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_str.lower() == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), eps=self.hparams.eps, 
                                            lr=self.hparams.lr, lr_decay=self.hparams.lr_decay, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_str.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=self.hparams.amsgrad, betas=self.hparams.betas, eps=self.hparams.eps, 
                                            lr=self.hparams.lr, maximize=self.hparams.maximize, weight_decay=self.hparams.weight_decay)

    def train(self):
        # The number of folds (This should come from the hparams)
        k_folds = self.hparams.k_folds

        # Init the loss and accuracy reporting lists
        self.training_accuracy_list = []
        self.training_loss_list = []
        self.validation_accuracy_list = []
        self.validation_loss_list = []

        # Set fixed random number seed
        torch.manual_seed(42)
        
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=k_folds, shuffle=True)

        models = []
        weights = []
        accuracies = []
        train_dataloaders = []
        validate_dataloaders = []

        # K-fold Cross-validation model evaluation
        for fold, (train_ids, validate_ids) in tqdm(enumerate(kfold.split(self.trainingDataset)), desc="Fold", initial=1, total=k_folds, leave=False):
            epoch_training_loss_list = []
            epoch_training_accuracy_list = []
            epoch_validation_loss_list = []
            epoch_validation_accuracy_list = []
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            validate_subsampler = torch.utils.data.SubsetRandomSampler(validate_ids)

            # Define data loaders for training and testing data in this fold
            self.train_dataloader = GraphDataLoader(self.trainingDataset, sampler=train_subsampler, 
                                                batch_size=self.hparams.batch_size,
                                                drop_last=False)
            self.validate_dataloader = GraphDataLoader(self.trainingDataset, sampler=validate_subsampler,
                                                batch_size=self.hparams.batch_size,
                                                drop_last=False)
            # Init the neural network
            self.reset_weights()

            # Run the training loop for defined number of epochs
            for _ in tqdm(range(0,self.hparams.epochs), desc='Epochs', initial=1, total=self.hparams.epochs, leave=False):
                temp_loss_list = []
                temp_acc_list = []

                # Iterate over the DataLoader for training data
                for batched_graph, labels in tqdm(self.train_dataloader, desc='Training', leave=False):

                    # Make sure the model is in training mode
                    self.model.train()
                    
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
                    temp_loss_list.append(loss.item())
                    temp_acc_list.append(accuracy_score(labels, pred.argmax(1)))

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    self.optimizer.step()

                epoch_training_accuracy_list.append(np.mean(temp_acc_list).item())
                epoch_training_loss_list.append(np.mean(temp_loss_list).item())
                self.validate()
                epoch_validation_accuracy_list.append(self.validation_accuracy)
                epoch_validation_loss_list.append(self.validation_loss)
            models.append(self.model)
            weights.append(copy.deepcopy(self.model.state_dict()))
            accuracies.append(self.validation_accuracy)
            train_dataloaders.append(self.train_dataloader)
            validate_dataloaders.append(self.validate_dataloader)
            self.training_accuracy_list.append(epoch_training_accuracy_list)
            self.training_loss_list.append(epoch_training_loss_list)
            self.validation_accuracy_list.append(epoch_validation_accuracy_list)
            self.validation_loss_list.append(epoch_validation_loss_list)
        self.accuracies = accuracies
        max_accuracy = max(accuracies)
        self.max_accuracy = max_accuracy
        ind = accuracies.index(max_accuracy)
        self.model = models[ind]
        self.model.load_state_dict(weights[ind])
        self.model.eval()
        self.training_accuracy_list = self.training_accuracy_list[ind]
        self.training_loss_list = self.training_loss_list[ind]
        self.validation_accuracy_list = self.validation_accuracy_list[ind]
        self.validation_loss_list = self.validation_loss_list[ind]
        
    def validate(self):
        temp_loss_list = []
        temp_acc_list = []
        self.model.eval()
        for batched_graph, labels in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float())
            if self.hparams.loss_function.lower() == "negative log likelihood":
                logp = F.log_softmax(pred, 1)
                loss = F.nll_loss(logp, labels)
            elif self.hparams.loss_function.lower() == "cross entropy":
                loss = F.cross_entropy(pred, labels)
            temp_loss_list.append(loss.item())
            temp_acc_list.append(accuracy_score(labels, pred.argmax(1)))
        self.validation_accuracy = np.mean(temp_acc_list).item()
        self.validation_loss = np.mean(temp_loss_list).item()
    
    def test(self):
        if self.testingDataset:
            self.test_dataloader = GraphDataLoader(self.testingDataset,
                                                    batch_size=len(self.testingDataset),
                                                    drop_last=False)
            temp_loss_list = []
            temp_acc_list = []
            self.model.eval()
            for batched_graph, labels in tqdm(self.test_dataloader, desc='Testing', leave=False):
                pred = self.model(batched_graph, batched_graph.ndata[self.node_attr_key].float())
                if self.hparams.loss_function.lower() == "negative log likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, labels)
                elif self.hparams.loss_function.lower() == "cross entropy":
                    loss = F.cross_entropy(pred, labels)
                temp_loss_list.append(loss.item())
                temp_acc_list.append(accuracy_score(labels, pred.argmax(1)))
            self.testing_accuracy = np.mean(temp_acc_list).item()
            self.testing_loss = np.mean(temp_loss_list).item()
        
    def save(self, path):
        if path:
            # Make sure the file extension is .pt
            ext = path[len(path)-3:len(path)]
            if ext.lower() != ".pt":
                path = path+".pt"
            torch.save(self.model, path)
    
class DGL:
    @staticmethod
    def Accuracy(actual, predicted, mantissa=4):
        """
        Computes the accuracy of the input predictions based on the input labels. This is to be used only with classification not with regression.

        Parameters
        ----------
        actual : list
            The input list of actual values.
        predicted : list
            The input list of predicted values.
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
    
    @staticmethod
    def RMSE(actual, predicted, mantissa=4):
        """
        Computes the accuracy based on the mean squared error of the input predictions based on the input actual values. This is to be used only with regression not with classification.

        Parameters
        ----------
        actual : list
            The input list of actual values.
        predicted : list
            The input list of predicted values.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.
        
        Returns
        -------
        float
            The RMSE value.
        """
        if len(predicted) < 1 or len(actual) < 1 or not len(predicted) == len(actual):
            return None
        size = len(predicted)
        mse = F.mse_loss(torch.tensor(predicted), torch.tensor(actual))
        return round(torch.sqrt(mse).item(), mantissa)
    
    @staticmethod
    def DatasetBalance(dataset, labels=None, method="undersampling", key="node_attr"):
        """
        Balances the input dataset using the specified method.
    
        Parameters
        ----------
        dataset : DGLDataset
            The input dataset.
        labels : list , optional
            The input list of labels. If set to None, all labels in the dataset will be considered and balanced.
        method : str, optional
            The method of sampling. This can be "undersampling" or "oversampling". It is case insensitive. The defaul is "undersampling".
        key : str
            The key used for the node attributes.
        
        Returns
        -------
        DGLDataset
            The balanced dataset.
        
        """
        if labels == None:
            labels = dataset.labels
        df = pd.DataFrame({'graph_index': range(len(labels)), 'label': labels})

        if method.lower() == 'undersampling':
            min_distribution = df['label'].value_counts().min()
            df = df.groupby('label').sample(n=min_distribution)
        elif method.lower() == 'oversampling':
            max_distribution = df['label'].value_counts().max()
            df = df.groupby('label').sample(n=max_distribution, replace=True)
        else:
            raise NotImplementedError

        list_idx = df['graph_index'].tolist()
        graphs = []
        labels = []
        for index in list_idx:
            graph, label = dataset[index]
            graphs.append(graph)
            labels.append(label)
        return DGL.DatasetByGraphs(dictionary={'graphs': graphs, 'labels': labels}, key=key)
    
    @staticmethod
    def GraphByTopologicGraph(topologicGraph, bidirectional=True, key=None, categories=[], node_attr_key="node_attr", tolerance=0.0001):
        """
        Returns a DGL graph by the input topologic graph.

        Parameters
        ----------
        topologicGraph : topologic.Graph
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
        vertices = Graph.Vertices(topologicGraph)
        edges = Graph.Edges(topologicGraph)
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
    def GraphsByImportedCSV(graphs_file_path, edges_file_path,
                              nodes_file_path, graph_id_header="graph_id",
                              graph_label_header="label", num_nodes_header="num_nodes", src_header="src",
                              dst_header="dst", node_label_header="label", node_attr_key="node_attr",
                              categories=[], bidirectional=True):
        """
        DEPRECATED. DO NOT USE. PLEASE USE GraphsByCSVPath
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
        dict
            The dictionary of DGL graphs and labels found in the input CSV files. The keys in the dictionary are "graphs" and "labels"

        """
        print("DGL.ByImportedCSV - WARNING: This method is deprecated. Please do not use. Instead use DGL.ByCSVPath.")
        return DGL.GraphsByCSVPath(graphs_file_path=graphs_file_path, edges_file_path=edges_file_path,
                              nodes_file_path=nodes_file_path, graph_id_header=graph_id_header,
                              graph_label_header=graph_label_header, num_nodes_header=num_nodes_header, src_header=src_header,
                              dst_header=dst_header, node_label_header=node_label_header, node_attr_key=node_attr_key,
                              categories=categories, bidirectional=bidirectional)

    @staticmethod
    def GraphsByCSVPath(graphs_file_path, edges_file_path,
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
        dict
            The dictionary of DGL graphs and labels found in the input CSV files. The keys in the dictionary are "graphs" and "labels"

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
    def GraphsByImportedDGCNN(path, categories=[], bidirectional=True):
        """
        DEPRECATED. DO NOT USE. PLEASE USE GraphsByCSVPath
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
        print("DGL.ByImportedDGCNN - WARNING: This method is deprecated. Please do not use. Instead use DGL.ByDGCNNPath.")
        return DGL.GraphsByDGCNNPath(path=path, categories=categories, bidirectional=bidirectional)

    @staticmethod
    def GraphsByDGCNNPath(path, categories=[], bidirectional=True):
        """
        Returns the Graphs from the input DGCNN file path.

        Parameters
        ----------
        file : file object
            The input DGCNN text file.
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
        if not path:
            return None
        try:
            file = open(path)
        except:
            print("DGL.GraphsByDGCNNPath - Error: the DGCNN file is not a valid file. Returning None.")
            return None
        return DGL.GraphsByDGCNNFile(file=file, categories=categories, bidirectional=bidirectional)

    @staticmethod
    def GraphsByDGCNNFile(file, categories=[], bidirectional=True):
        """
        Returns the Graphs from the input DGCNN file.

        Parameters
        ----------
        file : file object
            The input DGCNN text file.
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

        if not file:
            return None
        graphs = []
        labels = []
        lines = []
        for lineo, line in enumerate(file):
            lines.append(line)
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
    def ModelByFilePath(path):
        """
        DEPRECATED. DO NOT USE. INSTEAD USE ModeLoad.
        Returns the model found at the input PT file path.
        Parameters
        ----------
        path : str
            File path for the saved classifier.

        Returns
        -------
        DGL Classifier
            The classifier.

        """
        print("DGL.ModelByFilePath - WARNING: DEPRECTAED. DO NOT USE. INSTEAD USE DGL.ModelLoad.")
        if not path:
            return None
        return torch.load(path)
    
    @staticmethod
    def ModelLoad(path):
        """
        Returns the model found at the input file path.

        Parameters
        ----------
        path : str
            File path for the saved classifier.

        Returns
        -------
        DGL Classifier
            The classifier.

        """
        if not path:
            return None
        return torch.load(path)
    
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
        from sklearn import metrics
        import numpy
        if not isinstance(actual, list):
            print("DGL.CondfusionMatrix - ERROR: The actual input is not a list. Returning None")
            return None
        if not isinstance(predicted, list):
            print("DGL.CondfusionMatrix - ERROR: The predicted input is not a list. Returning None")
            return None
        if len(actual) != len(predicted):
            print("DGL.CondfusionMatrix - ERROR: The two input lists do not have the same length. Returning None")
            return None
        if normalize:
            cm = numpy.transpose(metrics.confusion_matrix(y_true=actual, y_pred=predicted, normalize="true"))
        else:
            cm = numpy.transpose(metrics.confusion_matrix(y_true=actual, y_pred=predicted))
        return cm
    
    @staticmethod
    def DatasetByGraphs(dictionary, key="node_attr"):
        """
        Returns a DGL Dataset from the input DGL graphs.

        Parameters
        ----------
        dictionary : dict
            The input dictionary of graphs and labels. This dictionary must have the keys "graphs" and "labels"
        key : str
            The key used for the node attributes.

        Returns
        -------
        DGL.Dataset
            The creatred DGL dataset.

        """
        graphs = dictionary['graphs']
        labels = dictionary['labels']
        return _Dataset(graphs, labels, key)
    
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
    def DatasetByCSVPath_NC(folderPath):
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
        return _Dataset(dgl_graphs, dgl_labels, node_attr_key)
    
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
    def DatasetGraphs(dataset):
        """
        Returns the DGL graphs found the in the input dataset.

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
    def GraphEdgeData(graph):
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
        return graph.edata
    
    @staticmethod
    def Hyperparameters(optimizer, model_type="classifier", cv_type="Holdout", split=[0.8,0.1,0.1], k_folds=5,
                           hl_widths=[32], conv_layer_type="SAGEConv", pooling="AvgPooling",
                           batch_size=1, epochs=1, use_gpu=False, loss_function="Cross Entropy"):
        """
        Creates a hyperparameters object based on the input settings.

        Parameters
        ----------
        model_type : str , optional
            The desired type of model. The options are:
            - "Classifier"
            - "Regressor"
            The option is case insensitive. The default is "classifierholdout"
        optimizer : Optimizer
            The desired optimizer.
        cv_type : str , optional
            The desired cross-validation method. This can be "Holdout" or "K-Fold". It is case-insensitive. The default is "Holdout".
        split : list , optional
            The desired split between training validation, and testing. [0.8, 0.1, 0.1] means that 80% of the data is used for training 10% of the data is used for validation, and 10% is used for testing. The default is [0.8, 0.1, 0.1].
        k_folds : int , optional
            The desired number of k-folds. The default is 5.
        hl_widths : list , optional
            The list of hidden layer widths. A list of [16, 32, 16] means that the model will have 3 hidden layers with number of neurons in each being 16, 32, 16 respectively from input to output. The default is [32].
        conv_layer_type : str , optional
            The desired type of the convolution layer. The options are "Classic", "GraphConv", "GINConv", "SAGEConv", "TAGConv", "DGN". It is case insensitive. The default is "SAGEConv".
        pooling : str , optional
            The desired type of pooling. The options are "AvgPooling", "MaxPooling", or "SumPooling". It is case insensitive. The default is "AvgPooling".
        batch_size : int , optional
            The desired batch size. The default is 1.
        epochs : int , optional
            The desired number of epochs. The default is 1.
        use_gpu : bool , optional
            If set to True, the model will attempt to use the GPU. The default is False.
        loss_function : str , optional
            The desired loss function. The options are "Cross-Entropy" or "Negative Log Likelihood". It is case insensitive. The default is "Cross-Entropy".

        Returns
        -------
        Hyperparameters
            The created hyperparameters object.

        """
        
        if optimizer['name'].lower() == "adadelta":
            optimizer_str = "Adadelta"
        elif optimizer['name'].lower() == "adagrad":
            optimizer_str = "Adagrad"
        elif optimizer['name'].lower() == "adam":
            optimizer_str = "Adam"
        return _Hparams(model_type,
                        optimizer_str,
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
                        loss_function)
    
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
    def DatasetLabels(dataset):
        """
        Returns the labels of the graphs in the input dataset

        Parameters
        ----------
        dataset : DGLDataset
            The input dataset
        
        Returns
        -------
        list
            The list of labels.
        """
        labels = dataset[1]
        if isinstance(labels, torch.LongTensor):
            return [int(g[1]) for g in dataset]
        return [float(g[1]) for g in dataset]
    
    @staticmethod
    def DatasetMerge(datasets, key="node_attr"):
        """
        Merges the input list of datasets into one dataset

        Parameters
        ----------
        datasets : list
            The input list of DGLdatasets
        
        Returns
        -------
        DGLDataset
            The merged dataset
        """

        graphs = []
        labels = []
        for ds in datasets:
            graphs += DGL.DatasetGraphs(ds)
            labels += DGL.DatasetLabels(ds)
        return DGL.DatasetByGraphs(graphs, labels, key=key)
    
    @staticmethod
    def GraphNodeData(graph):
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
        return graph.ndata
    
    @staticmethod
    def DatasetRemoveCategory(dataset, label, key="node_attr"):
        """
        Removes graphs from the input dataset that have the input label

        Parameters
        ----------
        dataset : DGLDataset
            The input dataset
        label : int
            The input label
        key : str , optional
            The input node attribute key

        Returns
        -------
        DGLDataset
            The resulting dataset

        """

        graphs = DGL.DatasetGraphs(dataset)
        labels = DGL.DatasetLabels(dataset)
        new_graphs = []
        new_labels = []
        for i in range(len(labels)):
            if not labels[i] == label:
                new_graphs.append(graphs[i])
                new_labels.append(labels[i])
        return DGL.DatasetByGraphs(new_graphs, new_labels, key)
    
    @staticmethod
    def DatasetSplit(dataset, fracList=[0.8, 0.1, 0.1], shuffle=False, randomState=None, key="node_attr"):
        """
        Splits the dataset into training, validation, and testing datasets.

        Parameters
        ----------
        dataset : DGLDataset
            The input dataset
        fracList : list , optional
            A list of length 3 containing the fraction to use for training, validation and test. If None, we will use [0.8, 0.1, 0.1]. The default is [0.8, 0.1, 0.1]
        randomState :  int or array_like , optional
            Random seed used to initialize the pseudo-random number generator. Can be any integer between 0 and 2**32 - 1 inclusive, an array (or other sequence) of such integers, or None (the default). If seed is None, then RandomState will try to read data from /dev/urandom (or the Windows analogue) if available or seed from the clock otherwise.
        Returns
        -------
        dict
            The dictionary of the optimizer parameters. The dictionary contains the following keys and values:
            - "train_ds" (DGLDataset)
            - "validate_ds" (DGLDataset)
            - "test_ds" (DGLDataset)

        """

        if not 0 <= fracList[0] <= 1:
            return None
        if not 0 <= fracList[1] <= 1:
            return None
        if not 0 <= fracList[2] <= 1:
            return None
        if sum(fracList) > 1:
            return None
        datasets = dgl.data.utils.split_dataset(dataset, frac_list=fracList, shuffle=shuffle, random_state=randomState)
        if fracList[0] > 0:
            train_ds = DGL.DatasetByGraphs({'graphs': DGL.DatasetGraphs(datasets[0]), 'labels' :DGL.DatasetLabels(datasets[0])}, key=key)
        else:
            train_ds = None
        if fracList[1] > 0:
            validate_ds = DGL.DatasetByGraphs({'graphs': DGL.DatasetGraphs(datasets[1]), 'labels' :DGL.DatasetLabels(datasets[1])}, key=key)
        else:
            validate_ds = None
        if fracList[2] > 0:
            test_ds = DGL.DatasetByGraphs({'graphs': DGL.DatasetGraphs(datasets[2]), 'labels' :DGL.DatasetLabels(datasets[2])}, key=key)
        else:
            test_ds = None

        return {
            "train_ds" : train_ds,
            "validate_ds" : validate_ds,
            "test_ds" : test_ds
        }
    @staticmethod
    def Optimizer(name="Adam", amsgrad=True, betas=(0.9,0.999), eps=0.000001, lr=0.001, maximize=False, weightDecay=0.0, rho=0.9, lr_decay=0.0):
        """
        Returns the parameters of the optimizer

        Parameters
        ----------
        amsgrad : bool , optional.
            amsgrad is an extension to the Adam version of gradient descent that attempts to improve the convergence properties of the algorithm, avoiding large abrupt changes in the learning rate for each input variable. The default is True.
        betas : tuple , optional
            Betas are used as for smoothing the path to the convergence also providing some momentum to cross a local minima or saddle point. The default is (0.9, 0.999).
        eps : float . optional.
            eps is a term added to the denominator to improve numerical stability. The default is 0.000001.
        lr : float
            The learning rate (lr) defines the adjustment in the weights of our network with respect to the loss gradient descent. The default is 0.001.
        maximize : float , optional
            maximize the params based on the objective, instead of minimizing. The default is False.
        weightDecay : float , optional
            weightDecay (L2 penalty) is a regularization technique applied to the weights of a neural network. The default is 0.0.

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
    def ModelClassify(model, dataset, node_attr_key="node_attr"):
        """
        Predicts the classification the labels of the input dataset.

        Parameters
        ----------
        dataset : DGLDataset
            The input DGL dataset.
        model : Model
            The input trained model.
        node_attr_key : str , optional
            The key used for node attributes. The default is "node_attr".

        Returns
        -------
        dict
            Dictionary containing labels and probabilities. The included keys and values are:
            - "predictions" (list): the list of predicted labels
            - "probabilities" (list): the list of probabilities that the label is one of the categories.

        """
        try:
            model = model.model #The inoput model might be our wrapper model. In that case, get its model attribute to do the prediciton.
        except:
            pass
        labels = []
        probabilities = []
        for item in tqdm(dataset, desc='Classifying', leave=False):
            graph = item[0]
            pred = model(graph, graph.ndata[node_attr_key].float())
            labels.append(pred.argmax(1).item())
            probability = (torch.nn.functional.softmax(pred, dim=1).tolist())
            probability = probability[0]
            temp_probability = []
            for p in probability:
                temp_probability.append(round(p, 3))
            probabilities.append(temp_probability)
        return {"predictions":labels, "probabilities":probabilities}
    
    @staticmethod
    def ModelPredict(model, dataset, node_attr_key="node_attr"):
        """
        Predicts the value of the input dataset.

        Parameters
        ----------
        dataset : DGLDataset
            The input DGL dataset.
        model : Model
            The input trained model.
        node_attr_key : str , optional
            The key used for node attributes. The default is "node_attr".
    
        Returns
        -------
        list
            The list of predictions
        """
        try:
            model = model.model #The inoput model might be our wrapper model. In that case, get its model attribute to do the prediciton.
        except:
            pass
        values = []
        for item in tqdm(dataset, desc='Predicting', leave=False):
            graph = item[0]
            pred = model(graph, graph.ndata[node_attr_key].float())
            values.append(round(pred.item(), 3))
        return values
    
    @staticmethod
    def ModelClassifyNodes(model, dataset):
        """
        Predicts the calssification of the node labels found in the input dataset using the input classifier.

        Parameters
        ----------
        model : Model
            The input model.
        dataset : DGLDataset
            The input DGL Dataset.
        
        Returns
        -------
        dict
            A dictionary containing all the results. The keys in this dictionary are:
            - "alllabels"
            - "allpredictions"
            - "trainlabels"
            - "trainpredictions"
            - "validationlabels"
            - "validationpredictions"
            - "testlabels"
            - "testpredictions"

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
        
        graphs = DGL.DatasetGraphs(dataset)
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
            logits = model(g, features)
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
        
        return {
            "alllabels": Helper.Flatten(allLabels),
            "allpredictions" : Helper.Flatten(allPredictions),
            "trainlabels" : Helper.Flatten(trainLabels),
            "trainpredictions" : Helper.Flatten(trainPredictions),
            "validationlabels" : Helper.Flatten(valLabels),
            "validationpredictions" : Helper.Flatten(valPredictions),
            "testlabels" : Helper.Flatten(testLabels),
            "testpredictions" : Helper.Flatten(testPredictions)
            
        }

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
    
    @staticmethod
    def Model(hparams, trainingDataset, validationDataset=None, testingDataset=None):
        """
        Creates a neural network classifier.

        Parameters
        ----------
        hparams : HParams
            The input hyperparameters 
        trainingDataset : DGLDataset
            The input training dataset.
        validationDataset : DGLDataset
            The input validation dataset. If not specified, a portion of the trainingDataset will be used for validation according the to the split list as specified in the hyper-parameters.
        testingDataset : DGLDataset
            The input testing dataset. If not specified, a portion of the trainingDataset will be used for testing according the to the split list as specified in the hyper-parameters.

        Returns
        -------
        Classifier
            The created classifier

        """

        model = None
        if hparams.model_type.lower() == "classifier":
            if hparams.cv_type.lower() == "holdout":
                model = _ClassifierHoldout(hparams=hparams, trainingDataset=trainingDataset, validationDataset=validationDataset, testingDataset=testingDataset)
            elif "k" in hparams.cv_type.lower():
                model = _ClassifierKFold(hparams=hparams, trainingDataset=trainingDataset, testingDataset=testingDataset)
        elif hparams.model_type.lower() == "regressor":
            if hparams.cv_type.lower() == "holdout":
                model = _RegressorHoldout(hparams=hparams, trainingDataset=trainingDataset, validationDataset=validationDataset, testingDataset=testingDataset)
            elif "k" in hparams.cv_type.lower():
                model = _RegressorKFold(hparams=hparams, trainingDataset=trainingDataset, testingDataset=testingDataset)
        else:
            raise NotImplementedError
        return model

    @staticmethod
    def ModelTrain(model):
        """
        Trains the neural network model.

        Parameters
        ----------
        model : Model
            The input model.

        Returns
        -------
        Model
            The trained model

        """
        if not model:
            return None
        model.train()
        return model
    
    @staticmethod
    def ModelTest(model):
        """
        Tests the neural network model.

        Parameters
        ----------
        model : Model
            The input model.

        Returns
        -------
        Model
            The tested model

        """
        if not model:
            return None
        model.test()
        return model
    
    @staticmethod
    def ModelSave(model, path=None):
        """
        Saves the model.

        Parameters
        ----------
        model : Model
            The input model.

        Returns
        -------
        bool
            True if the model is saved correctly. False otherwise.

        """
        if not model:
            return None
        if path:
            # Make sure the file extension is .pt
            ext = path[len(path)-3:len(path)]
            if ext.lower() != ".pt":
                path = path+".pt"
            return model.save(path)
    
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
            'Model Type'
            'Optimizer'
            'CV Type'
            'Split'
            'K-Folds'
            'HL Widths'
            'Conv Layer Type'
            'Pooling'
            'Learning Rate'
            'Batch Size'
            'Epochs'
            'Training Accuracy'
            'Validation Accuracy'
            'Testing Accuracy'
            'Training Loss'
            'Validation Loss'
            'Testing Loss'
            'Accuracies' (Classifier and K-Fold only)
            'Max Accuracy' (Classifier and K-Fold only)
            'Losses' (Regressor and K-fold only)
            'min Loss' (Regressor and K-fold only)


        """
        from topologicpy.Helper import Helper
        
        data = {'Model Type': [model.hparams.model_type],
                'Optimizer': [model.hparams.optimizer_str],
                'CV Type': [model.hparams.cv_type],
                'Split': model.hparams.split,
                'K-Folds': [model.hparams.k_folds],
                'HL Widths': model.hparams.hl_widths,
                'Conv Layer Type': [model.hparams.conv_layer_type],
                'Pooling': [model.hparams.pooling],
                'Learning Rate': [model.hparams.lr],
                'Batch Size': [model.hparams.batch_size],
                'Epochs': [model.hparams.epochs]
            }
        
        if model.hparams.model_type.lower() == "classifier":
            testing_accuracy_list = [model.testing_accuracy] * model.hparams.epochs
            testing_loss_list = [model.testing_loss] * model.hparams.epochs
            metrics_data = {
                'Training Accuracy': [model.training_accuracy_list],
                'Validation Accuracy': [model.validation_accuracy_list],
                'Testing Accuracy' : [testing_accuracy_list],
                'Training Loss': [model.training_loss_list],
                'Validation Loss': [model.validation_loss_list],
                'Testing Loss' : [testing_loss_list]
            }
            if model.hparams.cv_type.lower() == "k-fold":
                accuracy_data = {
                    'Accuracies' : [model.accuracies],
                    'Max Accuracy' : [model.max_accuracy]
                }
                metrics_data.update(accuracy_data)
            data.update(metrics_data)
        
        elif model.hparams.model_type.lower() == "regressor":
            testing_loss_list = [model.testing_loss] * model.hparams.epochs
            metrics_data = {
                'Training Loss': [model.training_loss_list],
                'Validation Loss': [model.validation_loss_list],
                'Testing Loss' : [testing_loss_list]
            }
            if model.hparams.cv_type.lower() == "k-fold":
                loss_data = {
                    'Losses' : [model.losses],
                    'Min Loss' : [model.min_loss]
                }
                metrics_data.update(loss_data)
            data.update(metrics_data)
        
        return data

    @staticmethod
    def GraphsByBINPath(path, labelKey="value", key='node_attr'):
        """
        Returns the Graphs from the input BIN file path.

        Parameters
        ----------
        path : str
            The input BIN file path.
        labelKey : str , optional
            The label key to use. The default is "value".
        key : str , optional
            The node attributes key. The default is 'node_attr'.

        Returns
        -------
        dict
            A dictionary object that contains the imported graphs and their corresponding labels. The dictionary has the following keys and values:
            - "graphs" (list): The list of DGL graphs
            - "labels" (list): The list of graph labels

        """
        graphs, label_dict = load_graphs(path)
        labels = label_dict[labelKey].tolist()
        return {"graphs" : graphs, "labels": labels}
    
    @staticmethod
    def DataExportToCSV(data, path, overwrite=False):
        """
        Exports the input data to a CSV file

        Parameters
        ----------
        data : dict
            The input data. See Data(model)
        overwrite : bool , optional
            If set to True, previous saved results files are overwritten. Otherwise, the new results are appended to the previously saved files. The default is False.

        Returns
        -------
        bool
            True if the data is saved correctly to a CSV file. False otherwise.

        """
        from topologicpy.Helper import Helper
        from os.path import exists
        
        # Make sure the file extension is .csv
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".csv":
            path = path+".csv"
        
        if not overwrite and exists(path):
            print("DGL.ExportToCSV - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        
        epoch_list = list(range(1, data['Epochs'][0]+1))
        
        d = [data['Model Type'], data['Optimizer'], data['CV Type'], [data['Split']], data['K-Folds'], data['HL Widths'], data['Conv Layer Type'], data['Pooling'], data['Learning Rate'], data['Batch Size'], epoch_list]
        columns = ['Model Type', 'Optimizer', 'CV Type', 'Split', 'K-Folds', 'HL Widths', 'Conv Layer Type', 'Pooling', 'Learning Rate', 'Batch Size', 'Epochs']

        if data['Model Type'][0].lower() == "classifier":
            d.extend([data['Training Accuracy'][0], data['Validation Accuracy'][0], data['Testing Accuracy'][0], data['Training Loss'][0], data['Validation Loss'][0], data['Testing Loss'][0]])
            columns.extend(['Training Accuracy', 'Validation Accuracy', 'Testing Accuracy', 'Training Loss', 'Validation Loss', 'Testing Loss'])
            if data['CV Type'][0].lower() == "k-fold":
                d.extend([data['Accuracies'], data['Max Accuracy']])
                columns.extend(['Accuracies', 'Max Accuracy'])
            
        elif data['Model Type'][0].lower() == "regressor":
            d.extend([data['Training Loss'][0], data['Validation Loss'][0], data['Testing Loss'][0]])
            columns.extend(['Training Loss', 'Validation Loss', 'Testing Loss'])
            if data['CV Type'][0].lower() == "k-fold":
                d.extend([data['Losses'], data['Min Loss']])
                columns.extend(['Losses', 'Min Loss'])

        d = Helper.Iterate(d)
        d = Helper.Transpose(d)
        df = pd.DataFrame(d, columns=columns)
        
        status = False
        if path:
            if overwrite:
                mode = 'w+'
            else:
                mode = 'a'
            try:
                df.to_csv(path, mode=mode, index = False, header=True)
                status = True
            except:
                status = False
        return status
    
    @staticmethod
    def Precision(actual, predicted, mantissa=4):
        """
        Returns the precision of the predicted values vs. the actual values. See https://en.wikipedia.org/wiki/Precision_and_recall

        Parameters
        ----------
        actual : list
            The input list of actual values.
        predicted : list
            The input list of predicted values.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The precision value
        
        """

        categories = set(actual+predicted)
        true_positives = {category: 0 for category in categories}
        false_positives = {category: 0 for category in categories}

        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                true_positives[actual[i]] += 1
            else:
                false_positives[predicted[i]] += 1

        total_true_positives = sum(true_positives.values())
        total_false_positives = sum(false_positives.values())

        if total_true_positives + total_false_positives == 0:
            return 0

        return round(total_true_positives / (total_true_positives + total_false_positives), mantissa)
    
    @staticmethod
    def Recall(actual, predicted, mantissa=4):
        """
        Returns the recall metric of the predicted values vs. the actual values. See https://en.wikipedia.org/wiki/Precision_and_recall

        Parameters
        ----------
        actual : list
            The input list of actual values.
        predicted : list
            The input list of predicted values.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The recall value
        
        """

        categories = set(actual+predicted)
        true_positives = {category: 0 for category in categories}
        false_negatives = {category: 0 for category in categories}

        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                true_positives[actual[i]] += 1
            else:
                false_negatives[actual[i]] += 1

        total_true_positives = sum(true_positives.values())
        total_false_negatives = sum(false_negatives.values())

        if total_true_positives + total_false_negatives == 0:
            return 0

        return round(total_true_positives / (total_true_positives + total_false_negatives), mantissa)



    '''
    @staticmethod
    def TrainRegressor(hparams, trainingDataset, validationDataset=None, testingDataset=None, overwrite=False):
        """
        Trains a neural network regressor.

        Parameters
        ----------
        hparams : HParams
            The input hyperparameters 
        trainingDataset : DGLDataset
            The input training dataset.
        validationDataset : DGLDataset
            The input validation dataset. If not specified, a portion of the trainingDataset will be used for validation according the to the split list as specified in the hyper-parameters.
        testingDataset : DGLDataset
            The input testing dataset. If not specified, a portion of the trainingDataset will be used for testing according the to the split list as specified in the hyper-parameters.
        overwrite : bool , optional
            If set to True, previous saved results files are overwritten. Otherwise, the new results are appended to the previously saved files. The default is False.

        Returns
        -------
        dict
            A dictionary containing all the results.

        """

        from topologicpy.Helper import Helper
        import time
        import datetime
        start = time.time()
        regressor = _RegressorHoldout(hparams, trainingDataset, validationDataset, testingDataset)
        regressor.train()
        accuracy = regressor.validate()
    
        end = time.time()
        duration = round(end - start,3)
        utcnow = datetime.datetime.utcnow()
        timestamp_str = "UTC-"+str(utcnow.year)+"-"+str(utcnow.month)+"-"+str(utcnow.day)+"-"+str(utcnow.hour)+"-"+str(utcnow.minute)+"-"+str(utcnow.second)
        epoch_list = list(range(1,regressor.hparams.epochs+1))
        d2 = [[timestamp_str], [duration], [regressor.hparams.optimizer_str], [regressor.hparams.cv_type], [regressor.hparams.split], [regressor.hparams.k_folds], regressor.hparams.hl_widths, [regressor.hparams.conv_layer_type], [regressor.hparams.pooling], [regressor.hparams.lr], [regressor.hparams.batch_size], epoch_list, regressor.training_accuracy_list, regressor.validation_accuracy_list]
        d2 = Helper.Iterate(d2)
        d2 = Helper.Transpose(d2)
    
        data = {'TimeStamp': "UTC-"+str(utcnow.year)+"-"+str(utcnow.month)+"-"+str(utcnow.day)+"-"+str(utcnow.hour)+"-"+str(utcnow.minute)+"-"+str(utcnow.second),
                'Duration': [duration],
                'Optimizer': [regressor.hparams.optimizer_str],
                'CV Type': [regressor.hparams.cv_type],
                'Split': [regressor.hparams.split],
                'K-Folds': [regressor.hparams.k_folds],
                'HL Widths': [regressor.hparams.hl_widths],
                'Conv Layer Type': [regressor.hparams.conv_layer_type],
                'Pooling': [regressor.hparams.pooling],
                'Learning Rate': [regressor.hparams.lr],
                'Batch Size': [regressor.hparams.batch_size],
                'Epochs': [regressor.hparams.epochs],
                'Training Accuracy': [regressor.training_accuracy_list],
                'Validation Accuracy': [regressor.validation_accuracy_list]
            }

        df = pd.DataFrame(d2, columns= ['TimeStamp', 'Duration', 'Optimizer', 'CV Type', 'Split', 'K-Folds', 'HL Widths', 'Conv Layer Type', 'Pooling', 'Learning Rate', 'Batch Size', 'Epochs', 'Training Accuracy', 'Testing Accuracy'])
        if regressor.hparams.results_path:
            if overwrite:
                df.to_csv(regressor.hparams.results_path, mode='w+', index = False, header=True)
            else:
                df.to_csv(regressor.hparams.results_path, mode='a', index = False, header=False)
        return data
    '''

    @staticmethod
    def _TrainClassifier_NC(graphs, model, hparams):
        """
        Parameters
        ----------
        graphs : list
            The input list of graphs.
        model : GCN Model
            The input classifier model.
        hparams : HParams
            The input hyper-parameters.

        Returns
        -------
        list
            The list of trained model and predictions.

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
    def TrainNodeClassifier(hparams, dataset, numLabels, sample):
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
        graphs = DGL.DatasetGraphs(dataset)
        # Sample a random list from the graphs
        if sample < len(graphs) and sample > 0:
            graphs = random.sample(graphs, sample)
        if len(graphs) == 1:
            i = 0
        elif len(graphs) > 1:
            i = random.randrange(0, len(graphs)-1)
        else: # There are no gaphs in the dataset, return None
            return None
        model = _Classic(graphs[i].ndata['feat'].shape[1], hparams.hl_widths, numLabels)
        final_model, predictions = DGL._TrainNodeClassifier(graphs, model, hparams)
        # Save the entire model
        if hparams.checkpoint_path is not None:
            torch.save(final_model, hparams.checkpoint_path)
        return final_model
