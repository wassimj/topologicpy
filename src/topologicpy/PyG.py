# Copyright (C) 2025
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
import copy
import warnings
import gc

try:
    import numpy as np
except:
    print("PyG - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        print("PyG - numpy library installed successfully.")
    except:
        warnings.warn("PyG - Error: Could not import numpy.")

try:
    import pandas as pd
except:
    print("PyG - Installing required pandas library.")
    try:
        os.system("pip install pandas")
    except:
        os.system("pip install pandas --user")
    try:
        import numpy as np
        print("PyG - pandas library installed successfully.")
    except:
        warnings.warn("PyG - Error: Could not import pandas.")

try:
    from tqdm.auto import tqdm
except:
    print("PyG - Installing required tqdm library.")
    try:
        os.system("pip install tqdm")
    except:
        os.system("pip install tqdm --user")
    try:
        from tqdm.auto import tqdm
        print("PyG - tqdm library installed correctly.")
    except:
        raise Exception("PyG - Error: Could not import tqdm.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data.sampler import SubsetRandomSampler
except:
    print("PyG - Installing required torch library.")
    try:
        os.system("pip install torch")
    except:
        os.system("pip install torch --user")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data.sampler import SubsetRandomSampler
        print("PyG - torch library installed correctly.")
    except:
        warnings.warn("PyG - Error: Could not import torch.")

try:
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool, global_add_pool
except:
    print("PyG - Installing required torch_geometric library.")
    try:
        os.system("pip install torch_geometric")
    except:
        os.system("pip install torch_geometric --user")
    try:
        from torch_geometric.data import Data, Dataset
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool, global_add_pool
        print("PyG - torch_geometric library installed correctly.")
    except:
        warnings.warn("PyG - Error: Could not import torch.")

try:
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
except:
    print("PyG - Installing required scikit-learn library.")
    try:
        os.system("pip install -U scikit-learn")
    except:
        os.system("pip install -U scikit-learn --user")
    try:
        from sklearn.model_selection import KFold
        from sklearn.metrics import accuracy_score
        print("PyG - scikit-learn library installed correctly.")
    except:
        warnings.warn("PyG - Error: Could not import scikit. Please install it manually.")

class CustomGraphDataset(Dataset):
    def __init__(self, root=None, data_list=None, indices=None, node_level=False, graph_level=True, 
                 node_attr_key='feat', edge_attr_key='feat'):
        """
        Initializes the CustomGraphDataset.

        Parameters:
        - root: Root directory of the dataset (used only if data_list is None)
        - data_list: List of preprocessed data objects (used if provided)
        - indices: List of indices to select a subset of the data
        - node_level: Boolean flag indicating if the dataset is node-level
        - graph_level: Boolean flag indicating if the dataset is graph-level
        - node_attr_key: Key for node attributes
        - edge_attr_key: Key for edge attributes
        """
        assert not (node_level and graph_level), "Both node_level and graph_level cannot be True at the same time"
        assert node_level or graph_level, "Both node_level and graph_level cannot be False at the same time"

        self.node_level = node_level
        self.graph_level = graph_level
        self.node_attr_key = node_attr_key
        self.edge_attr_key = edge_attr_key

        if data_list is not None:
            self.data_list = data_list  # Use the provided data list
        elif root is not None:
            # Load and process data from root directory if data_list is not provided
            self.graph_df = pd.read_csv(os.path.join(root, 'graphs.csv'))
            self.nodes_df = pd.read_csv(os.path.join(root, 'nodes.csv'))
            self.edges_df = pd.read_csv(os.path.join(root, 'edges.csv'))
            self.data_list = self.process_all()
        else:
            raise ValueError("Either a root directory or a data_list must be provided.")

        # Filter data_list based on indices if provided
        if indices is not None:
            self.data_list = [self.data_list[i] for i in indices]

    def process_all(self):
        data_list = []
        for graph_id in self.graph_df['graph_id'].unique():
            graph_nodes = self.nodes_df[self.nodes_df['graph_id'] == graph_id]
            graph_edges = self.edges_df[self.edges_df['graph_id'] == graph_id]

            if self.node_attr_key in graph_nodes.columns and not graph_nodes[self.node_attr_key].isnull().all():
                x = torch.tensor(graph_nodes[self.node_attr_key].values.tolist(), dtype=torch.float)
                if x.ndim == 1:
                    x = x.unsqueeze(1)  # Ensure x has shape [num_nodes, *]
            else:
                x = None

            edge_index = torch.tensor(graph_edges[['src_id', 'dst_id']].values.T, dtype=torch.long)

            if self.edge_attr_key in graph_edges.columns and not graph_edges[self.edge_attr_key].isnull().all():
                edge_attr = torch.tensor(graph_edges[self.edge_attr_key].values.tolist(), dtype=torch.float)
            else:
                edge_attr = None

            if self.graph_level:
                label_value = self.graph_df[self.graph_df['graph_id'] == graph_id]['label'].values[0]
                
                if isinstance(label_value, np.int64):
                    label_value = int(label_value)
                if isinstance(label_value, np.float64):
                    label_value = float(label_value)

                if isinstance(label_value, int) or isinstance(label_value, np.int64):
                    y = torch.tensor([label_value], dtype=torch.long)
                elif isinstance(label_value, float):
                    y = torch.tensor([label_value], dtype=torch.float)
                else:
                    raise ValueError(f"Unexpected label type: {type(label_value)}. Expected int or float.")
                    
            elif self.node_level:
                label_values = graph_nodes['label'].values
                
                if issubclass(label_values.dtype.type, int):
                    y = torch.tensor(label_values, dtype=torch.long)
                elif issubclass(label_values.dtype.type, float):
                    y = torch.tensor(label_values, dtype=torch.float)
                else:
                    raise ValueError(f"Unexpected label types: {label_values.dtype}. Expected int or float.")

            data = Data(x=x, edge_index=edge_index, y=y)
            if edge_attr is not None:
                data.edge_attr = edge_attr

            data_list.append(data)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]




# class CustomGraphDataset(Dataset):
#     def __init__(self, root, node_level=False, graph_level=True, node_attr_key='feat', 
#                  edge_attr_key='feat', transform=None, pre_transform=None):
#         super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
#         assert not (node_level and graph_level), "Both node_level and graph_level cannot be True at the same time"
#         assert node_level or graph_level, "Both node_level and graph_level cannot be False at the same time"

#         self.node_level = node_level
#         self.graph_level = graph_level
#         self.node_attr_key = node_attr_key
#         self.edge_attr_key = edge_attr_key

#         self.graph_df = pd.read_csv(os.path.join(root, 'graphs.csv'))
#         self.nodes_df = pd.read_csv(os.path.join(root, 'nodes.csv'))
#         self.edges_df = pd.read_csv(os.path.join(root, 'edges.csv'))

#         self.data_list = self.process_all()

#     @property
#     def raw_file_names(self):
#         return ['graphs.csv', 'nodes.csv', 'edges.csv']

#     def process_all(self):
#         data_list = []
#         for graph_id in self.graph_df['graph_id'].unique():
#             graph_nodes = self.nodes_df[self.nodes_df['graph_id'] == graph_id]
#             graph_edges = self.edges_df[self.edges_df['graph_id'] == graph_id]

#             if self.node_attr_key in graph_nodes.columns and not graph_nodes[self.node_attr_key].isnull().all():
#                 x = torch.tensor(graph_nodes[self.node_attr_key].values.tolist(), dtype=torch.float)
#                 if x.ndim == 1:
#                     x = x.unsqueeze(1)  # Ensure x has shape [num_nodes, *]
#             else:
#                 x = None

#             edge_index = torch.tensor(graph_edges[['src_id', 'dst_id']].values.T, dtype=torch.long)

#             if self.edge_attr_key in graph_edges.columns and not graph_edges[self.edge_attr_key].isnull().all():
#                 edge_attr = torch.tensor(graph_edges[self.edge_attr_key].values.tolist(), dtype=torch.float)
#             else:
#                 edge_attr = None



#             if self.graph_level:
#                 label_value = self.graph_df[self.graph_df['graph_id'] == graph_id]['label'].values[0]
                
#                 # Check if the label is an integer or a float and cast accordingly
#                 if isinstance(label_value, int):
#                     y = torch.tensor([label_value], dtype=torch.long)
#                 elif isinstance(label_value, float):
#                     y = torch.tensor([label_value], dtype=torch.float)
#                 else:
#                     raise ValueError(f"Unexpected label type: {type(label_value)}. Expected int or float.")
                    
#             elif self.node_level:
#                 label_values = graph_nodes['label'].values
                
#                 # Check if the labels are integers or floats and cast accordingly
#                 if issubclass(label_values.dtype.type, int):
#                     y = torch.tensor(label_values, dtype=torch.long)
#                 elif issubclass(label_values.dtype.type, float):
#                     y = torch.tensor(label_values, dtype=torch.float)
#                 else:
#                     raise ValueError(f"Unexpected label types: {label_values.dtype}. Expected int or float.")


#             # if self.graph_level:
#             #     y = torch.tensor([self.graph_df[self.graph_df['graph_id'] == graph_id]['label'].values[0]], dtype=torch.long)
#             # elif self.node_level:
#             #     y = torch.tensor(graph_nodes['label'].values, dtype=torch.long)

#             data = Data(x=x, edge_index=edge_index, y=y)
#             if edge_attr is not None:
#                 data.edge_attr = edge_attr

#             data_list.append(data)

#         return data_list

#     def len(self):
#         return len(self.data_list)

#     def get(self, idx):
#         return self.data_list[idx]

#     def __getitem__(self, idx):
#         return self.get(idx)

class _Hparams:
    def __init__(self, model_type="ClassifierHoldout", optimizer_str="Adam", amsgrad=False, betas=(0.9, 0.999), eps=1e-6, lr=0.001, lr_decay= 0, maximize=False, rho=0.9, weight_decay=0, cv_type="Holdout", split=[0.8,0.1, 0.1], k_folds=5, hl_widths=[32], conv_layer_type='SAGEConv', pooling="AvgPooling", batch_size=32, epochs=1, 
                 use_gpu=False, loss_function="Cross Entropy", input_type="graph"):
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
        input_type : str
            selects the input_type of model such as graph, node or edge

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
        self.input_type = input_type

class _SAGEConv(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, pooling=None):
        super(_SAGEConv, self).__init__()
        assert isinstance(h_feats, list), "h_feats must be a list"
        h_feats = [x for x in h_feats if x is not None]
        assert len(h_feats) != 0, "h_feats is empty. unable to add hidden layers"
        self.list_of_layers = nn.ModuleList()
        dim = [in_feats] + h_feats

        # Convolution (Hidden) Layers
        for i in range(1, len(dim)):
            self.list_of_layers.append(SAGEConv(dim[i-1], dim[i]))

        # Final Layer
        self.final = nn.Linear(dim[-1], num_classes)

        # Pooling layer
        if pooling is None:
            self.pooling_layer = None
        else:
            if "av" in pooling.lower():
                self.pooling_layer = global_mean_pool
            elif "max" in pooling.lower():
                self.pooling_layer = global_max_pool
            elif "sum" in pooling.lower():
                self.pooling_layer = global_add_pool
            else:
                raise NotImplementedError

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = x
        # Generate node features
        for layer in self.list_of_layers:
            h = layer(h, edge_index)
            h = F.relu(h)
        # h will now be a matrix of dimension [num_nodes, h_feats[-1]]
        h = self.final(h)
        # Go from node-level features to graph-level features by pooling
        if self.pooling_layer:
            h = self.pooling_layer(h, batch)
            # h will now be a vector of dimension [num_classes]
        return h

class _GraphRegressorHoldout:
    def __init__(self, hparams, trainingDataset, validationDataset=None, testingDataset=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainingDataset = trainingDataset
        self.validationDataset = validationDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        if hparams.conv_layer_type.lower() == 'sageconv':
            self.model = _SAGEConv(trainingDataset[0].num_node_features, hparams.hl_widths, 1, hparams.pooling).to(self.device)
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
        self.node_attr_key = trainingDataset[0].x.shape[1]

        # Train, validate, test split
        num_train = int(len(trainingDataset) * hparams.split[0])
        num_validate = int(len(trainingDataset) * hparams.split[1])
        num_test = len(trainingDataset) - num_train - num_validate
        idx = torch.randperm(len(trainingDataset))
        train_sampler = SubsetRandomSampler(idx[:num_train])
        validate_sampler = SubsetRandomSampler(idx[num_train:num_train+num_validate])
        test_sampler = SubsetRandomSampler(idx[num_train+num_validate:])

        if validationDataset:
            self.train_dataloader = DataLoader(trainingDataset, 
                                               batch_size=hparams.batch_size,
                                               drop_last=False)
            self.validate_dataloader = DataLoader(validationDataset,
                                                  batch_size=hparams.batch_size,
                                                  drop_last=False)
        else:
            self.train_dataloader = DataLoader(trainingDataset, sampler=train_sampler, 
                                               batch_size=hparams.batch_size,
                                               drop_last=False)
            self.validate_dataloader = DataLoader(trainingDataset, sampler=validate_sampler,
                                                  batch_size=hparams.batch_size,
                                                  drop_last=False)
        
        if testingDataset:
            self.test_dataloader = DataLoader(testingDataset,
                                              batch_size=len(testingDataset),
                                              drop_last=False)
        else:
            self.test_dataloader = DataLoader(trainingDataset, sampler=test_sampler,
                                              batch_size=hparams.batch_size,
                                              drop_last=False)

    def train(self):
        # Init the loss and accuracy reporting lists
        self.training_loss_list = []
        self.validation_loss_list = []

        # Run the training loop for defined number of epochs
        for _ in tqdm(range(self.hparams.epochs), desc='Epochs', total=self.hparams.epochs, leave=False):
            # Iterate over the DataLoader for training data
            for data in tqdm(self.train_dataloader, desc='Training', leave=False):
                data = data.to(self.device)
                # Make sure the model is in training mode
                self.model.train()
                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                pred = self.model(data).to(self.device)
                # Compute loss
                loss = F.mse_loss(torch.flatten(pred), data.y.float())

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()

            self.training_loss_list.append(torch.sqrt(loss).item())
            self.validate()
            self.validation_loss_list.append(torch.sqrt(self.validation_loss).item())
            gc.collect()

    def validate(self):
        self.model.eval()
        for data in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            data = data.to(self.device)
            pred = self.model(data).to(self.device)
            loss = F.mse_loss(torch.flatten(pred), data.y.float())
        self.validation_loss = loss
    
    def test(self):
        self.model.eval()
        for data in tqdm(self.test_dataloader, desc='Testing', leave=False):
            data = data.to(self.device)
            pred = self.model(data).to(self.device)
            loss = F.mse_loss(torch.flatten(pred), data.y.float())
        self.testing_loss = torch.sqrt(loss).item()
    
    def save(self, path):
        if path:
            # Make sure the file extension is .pt
            ext = path[-3:]
            if ext.lower() != ".pt":
                path = path + ".pt"
            torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        #self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

class _GraphRegressorKFold:
    def __init__(self, hparams, trainingDataset, testingDataset=None):
        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        self.losses = []
        self.min_loss = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = self._initialize_model(hparams, trainingDataset)
        self.optimizer = self._initialize_optimizer(hparams)
        
        self.use_gpu = hparams.use_gpu
        self.training_loss_list = []
        self.validation_loss_list = []
        self.node_attr_key = trainingDataset.node_attr_key

        # Train, validate, test split
        num_train = int(len(trainingDataset) * hparams.split[0])
        num_validate = int(len(trainingDataset) * hparams.split[1])
        num_test = len(trainingDataset) - num_train - num_validate
        idx = torch.randperm(len(trainingDataset))
        test_sampler = SubsetRandomSampler(idx[num_train+num_validate:num_train+num_validate+num_test])
        
        if testingDataset:
            self.test_dataloader = DataLoader(testingDataset, batch_size=len(testingDataset), drop_last=False)
        else:
            self.test_dataloader = DataLoader(trainingDataset, sampler=test_sampler, batch_size=hparams.batch_size, drop_last=False)
    
    def _initialize_model(self, hparams, dataset):
        if hparams.conv_layer_type.lower() == 'sageconv':
            return _SAGEConv(dataset[0].num_node_features, hparams.hl_widths, 1, hparams.pooling).to(self.device)
            #return _SAGEConv(dataset.num_node_features, hparams.hl_widths, 1, hparams.pooling).to(self.device)
        else:
            raise NotImplementedError
    
    def _initialize_optimizer(self, hparams):
        if hparams.optimizer_str.lower() == "adadelta":
            return torch.optim.Adadelta(self.model.parameters(), eps=hparams.eps, lr=hparams.lr, rho=hparams.rho, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str.lower() == "adagrad":
            return torch.optim.Adagrad(self.model.parameters(), eps=hparams.eps, lr=hparams.lr, lr_decay=hparams.lr_decay, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), amsgrad=hparams.amsgrad, betas=hparams.betas, eps=hparams.eps, lr=hparams.lr, maximize=hparams.maximize, weight_decay=hparams.weight_decay)
    
    def reset_weights(self):
        self.model = self._initialize_model(self.hparams, self.trainingDataset)
        self.optimizer = self._initialize_optimizer(self.hparams)
    
    def train(self):
        k_folds = self.hparams.k_folds
        torch.manual_seed(42)
        
        kfold = KFold(n_splits=k_folds, shuffle=True)
        models, weights, losses, train_dataloaders, validate_dataloaders = [], [], [], [], []

        for fold, (train_ids, validate_ids) in tqdm(enumerate(kfold.split(self.trainingDataset)), desc="Fold", total=k_folds, leave=False):
            epoch_training_loss_list, epoch_validation_loss_list = [], []
            train_subsampler = SubsetRandomSampler(train_ids)
            validate_subsampler = SubsetRandomSampler(validate_ids)

            self.train_dataloader = DataLoader(self.trainingDataset, sampler=train_subsampler, batch_size=self.hparams.batch_size, drop_last=False)
            self.validate_dataloader = DataLoader(self.trainingDataset, sampler=validate_subsampler, batch_size=self.hparams.batch_size, drop_last=False)

            self.reset_weights()
            best_rmse = np.inf

            for _ in tqdm(range(self.hparams.epochs), desc='Epochs', total=self.hparams.epochs, leave=False):
                for batched_graph in tqdm(self.train_dataloader, desc='Training', leave=False):
                    self.model.train()
                    self.optimizer.zero_grad()

                    batched_graph = batched_graph.to(self.device)
                    pred = self.model(batched_graph)
                    loss = F.mse_loss(torch.flatten(pred), batched_graph.y.float())
                    loss.backward()
                    self.optimizer.step()

                epoch_training_loss_list.append(torch.sqrt(loss).item())
                self.validate()
                epoch_validation_loss_list.append(torch.sqrt(self.validation_loss).item())
                gc.collect()

            models.append(self.model)
            weights.append(copy.deepcopy(self.model.state_dict()))
            losses.append(torch.sqrt(self.validation_loss).item())
            train_dataloaders.append(self.train_dataloader)
            validate_dataloaders.append(self.validate_dataloader)
            self.training_loss_list.append(epoch_training_loss_list)
            self.validation_loss_list.append(epoch_validation_loss_list)

        self.losses = losses
        self.min_loss = min(losses)
        ind = losses.index(self.min_loss)
        self.model = models[ind]
        self.model.load_state_dict(weights[ind])
        self.model.eval()
        self.training_loss_list = self.training_loss_list[ind]
        self.validation_loss_list = self.validation_loss_list[ind]

    def validate(self):
        self.model.eval()
        for batched_graph in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            batched_graph = batched_graph.to(self.device)
            pred = self.model(batched_graph)
            loss = F.mse_loss(torch.flatten(pred), batched_graph.y.float())
        self.validation_loss = loss
    
    def test(self):
        self.model.eval()
        for batched_graph in tqdm(self.test_dataloader, desc='Testing', leave=False):
            batched_graph = batched_graph.to(self.device)
            pred = self.model(batched_graph)
            loss = F.mse_loss(torch.flatten(pred), batched_graph.y.float())
        self.testing_loss = torch.sqrt(loss).item()
    
    def save(self, path):
        if path:
            ext = path[-3:]
            if ext.lower() != ".pt":
                path = path + ".pt"
            torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

class _GraphClassifierKFold:
    def __init__(self, hparams, trainingDataset, testingDataset=None):
        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        self.testing_accuracy = 0
        self.accuracies = []
        self.max_accuracy = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if hparams.conv_layer_type.lower() == 'sageconv':
            self.model = _SAGEConv(trainingDataset.num_node_features, hparams.hl_widths, 
                            trainingDataset.num_classes, hparams.pooling).to(self.device)
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

    def reset_weights(self):
        if self.hparams.conv_layer_type.lower() == 'sageconv':
            self.model = _SAGEConv(self.trainingDataset.num_node_features, self.hparams.hl_widths, 
                            self.trainingDataset.num_classes, self.hparams.pooling).to(self.device)
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
            self.train_dataloader = DataLoader(self.trainingDataset, sampler=train_subsampler, 
                                                batch_size=self.hparams.batch_size,
                                                drop_last=False)
            self.validate_dataloader = DataLoader(self.trainingDataset, sampler=validate_subsampler,
                                                batch_size=self.hparams.batch_size,
                                                drop_last=False)
            # Init the neural network
            self.reset_weights()

            # Run the training loop for defined number of epochs
            for _ in tqdm(range(0,self.hparams.epochs), desc='Epochs', initial=1, total=self.hparams.epochs, leave=False):
                temp_loss_list = []
                temp_acc_list = []

                # Iterate over the DataLoader for training data
                for data in tqdm(self.train_dataloader, desc='Training', leave=False):
                    data = data.to(self.device)
                    # Make sure the model is in training mode
                    self.model.train()
                    
                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Perform forward pass
                    pred = self.model(data)

                    # Compute loss
                    if self.hparams.loss_function.lower() == "negative log likelihood":
                        logp = F.log_softmax(pred, 1)
                        loss = F.nll_loss(logp, data.y)
                    elif self.hparams.loss_function.lower() == "cross entropy":
                        loss = F.cross_entropy(pred, data.y)

                    # Save loss information for reporting
                    temp_loss_list.append(loss.item())
                    temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    self.optimizer.step()

                epoch_training_accuracy_list.append(np.mean(temp_acc_list).item())
                epoch_training_loss_list.append(np.mean(temp_loss_list).item())
                self.validate()
                epoch_validation_accuracy_list.append(self.validation_accuracy)
                epoch_validation_loss_list.append(self.validation_loss)
                gc.collect()
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
        for data in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            data = data.to(self.device)
            pred = self.model(data)
            if self.hparams.loss_function.lower() == "negative log likelihood":
                logp = F.log_softmax(pred, 1)
                loss = F.nll_loss(logp, data.y)
            elif self.hparams.loss_function.lower() == "cross entropy":
                loss = F.cross_entropy(pred, data.y)
            temp_loss_list.append(loss.item())
            temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))
        self.validation_accuracy = np.mean(temp_acc_list).item()
        self.validation_loss = np.mean(temp_loss_list).item()
    
    def test(self):
        if self.testingDataset:
            self.test_dataloader = DataLoader(self.testingDataset,
                                                    batch_size=len(self.testingDataset),
                                                    drop_last=False)
            temp_loss_list = []
            temp_acc_list = []
            self.model.eval()
            for data in tqdm(self.test_dataloader, desc='Testing', leave=False):
                data = data.to(self.device)
                pred = self.model(data)
                if self.hparams.loss_function.lower() == "negative log likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, data.y)
                elif self.hparams.loss_function.lower() == "cross entropy":
                    loss = F.cross_entropy(pred, data.y)
                temp_loss_list.append(loss.item())
                temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))
            self.testing_accuracy = np.mean(temp_acc_list).item()
            self.testing_loss = np.mean(temp_loss_list).item()
        
    def save(self, path):
        if path:
            # Make sure the file extension is .pt
            ext = path[-3:]
            if ext.lower() != ".pt":
                path = path + ".pt"
            torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        #self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

class _GraphClassifierHoldout:
    def __init__(self, hparams, trainingDataset, validationDataset=None, testingDataset=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainingDataset = trainingDataset
        self.validationDataset = validationDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        gclasses = trainingDataset.num_classes
        nfeats = trainingDataset.num_node_features
       
        if hparams.conv_layer_type.lower() == 'sageconv':
            self.model = _SAGEConv(nfeats, hparams.hl_widths, gclasses, hparams.pooling).to(self.device)
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
        self.node_attr_key = trainingDataset[0].x.shape[1]

        # train, validate, test split
        num_train = int(len(trainingDataset) * hparams.split[0])
        num_validate = int(len(trainingDataset) * hparams.split[1])
        num_test = len(trainingDataset) - num_train - num_validate
        idx = torch.randperm(len(trainingDataset))
        train_sampler = SubsetRandomSampler(idx[:num_train])
        validate_sampler = SubsetRandomSampler(idx[num_train:num_train+num_validate])
        test_sampler = SubsetRandomSampler(idx[num_train+num_validate:num_train+num_validate+num_test])

        if validationDataset:
            self.train_dataloader = DataLoader(trainingDataset, batch_size=hparams.batch_size, drop_last=False)
            self.validate_dataloader = DataLoader(validationDataset, batch_size=hparams.batch_size, drop_last=False)
        else:
            self.train_dataloader = DataLoader(trainingDataset, sampler=train_sampler, batch_size=hparams.batch_size, drop_last=False)
            self.validate_dataloader = DataLoader(trainingDataset, sampler=validate_sampler, batch_size=hparams.batch_size, drop_last=False)
        
        if testingDataset:
            self.test_dataloader = DataLoader(testingDataset, batch_size=len(testingDataset), drop_last=False)
        else:
            self.test_dataloader = DataLoader(trainingDataset, sampler=test_sampler, batch_size=hparams.batch_size, drop_last=False)
    
    def train(self):
        # Init the loss and accuracy reporting lists
        self.training_accuracy_list = []
        self.training_loss_list = []
        self.validation_accuracy_list = []
        self.validation_loss_list = []

        # Run the training loop for defined number of epochs
        for _ in tqdm(range(self.hparams.epochs), desc='Epochs', initial=1, leave=False):
            temp_loss_list = []
            temp_acc_list = []
            # Make sure the model is in training mode
            self.model.train()
            # Iterate over the DataLoader for training data
            for data in tqdm(self.train_dataloader, desc='Training', leave=False):
                data = data.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                pred = self.model(data)
                
                # Compute loss
                if self.hparams.loss_function.lower() == "negative log likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, data.y)
                elif self.hparams.loss_function.lower() == "cross entropy":
                    loss = F.cross_entropy(pred, data.y)

                # Save loss information for reporting
                temp_loss_list.append(loss.item())
                temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()

            self.training_accuracy_list.append(np.mean(temp_acc_list).item())
            self.training_loss_list.append(np.mean(temp_loss_list).item())
            self.validate()
            self.validation_accuracy_list.append(self.validation_accuracy)
            self.validation_loss_list.append(self.validation_loss)
            gc.collect()
        
    def validate(self):
        temp_loss_list = []
        temp_acc_list = []
        self.model.eval()
        for data in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            data = data.to(self.device)
            pred = self.model(data)
            if self.hparams.loss_function.lower() == "negative log likelihood":
                logp = F.log_softmax(pred, 1)
                loss = F.nll_loss(logp, data.y)
            elif self.hparams.loss_function.lower() == "cross entropy":
                loss = F.cross_entropy(pred, data.y)
            temp_loss_list.append(loss.item())
            temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))
        self.validation_accuracy = np.mean(temp_acc_list).item()
        self.validation_loss = np.mean(temp_loss_list).item()
    
    def test(self):
        if self.test_dataloader:
            temp_loss_list = []
            temp_acc_list = []
            self.model.eval()
            for data in tqdm(self.test_dataloader, desc='Testing', leave=False):
                data = data.to(self.device)
                pred = self.model(data)
                if self.hparams.loss_function.lower() == "negative log likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, data.y)
                elif self.hparams.loss_function.lower() == "cross entropy":
                    loss = F.cross_entropy(pred, data.y)
                temp_loss_list.append(loss.item())
                temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))
            self.testing_accuracy = np.mean(temp_acc_list).item()
            self.testing_loss = np.mean(temp_loss_list).item()
            
    def save(self, path):
        if path:
            # Make sure the file extension is .pt
            ext = path[-3:]
            if ext.lower() != ".pt":
                path = path + ".pt"
            torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        #self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

class _NodeClassifierHoldout:
    def __init__(self, hparams, trainingDataset, validationDataset=None, testingDataset=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainingDataset = trainingDataset
        self.validationDataset = validationDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        gclasses = trainingDataset.num_classes
        nfeats = trainingDataset.num_node_features
       
        if hparams.conv_layer_type.lower() == 'sageconv':
            # pooling is set None for Node classifier
            self.model = _SAGEConv(nfeats, hparams.hl_widths, gclasses, None).to(self.device)
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
        self.node_attr_key = trainingDataset[0].x.shape[1]

        # train, validate, test split
        num_train = int(len(trainingDataset) * hparams.split[0])
        num_validate = int(len(trainingDataset) * hparams.split[1])
        num_test = len(trainingDataset) - num_train - num_validate
        idx = torch.randperm(len(trainingDataset))
        train_sampler = SubsetRandomSampler(idx[:num_train])
        validate_sampler = SubsetRandomSampler(idx[num_train:num_train+num_validate])
        test_sampler = SubsetRandomSampler(idx[num_train+num_validate:num_train+num_validate+num_test])
        
        if validationDataset:
            self.train_dataloader = DataLoader(trainingDataset, batch_size=hparams.batch_size, drop_last=False)
            self.validate_dataloader = DataLoader(validationDataset, batch_size=hparams.batch_size, drop_last=False)
        else:
            self.train_dataloader = DataLoader(trainingDataset, sampler=train_sampler, batch_size=hparams.batch_size, drop_last=False)
            self.validate_dataloader = DataLoader(trainingDataset, sampler=validate_sampler, batch_size=hparams.batch_size, drop_last=False)
        
        if testingDataset:
            self.test_dataloader = DataLoader(testingDataset, batch_size=len(testingDataset), drop_last=False)
        else:
            self.test_dataloader = DataLoader(trainingDataset, sampler=test_sampler, batch_size=hparams.batch_size, drop_last=False)
    
    def train(self):
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
            for data in tqdm(self.train_dataloader, desc='Training', leave=False):
                data = data.to(self.device)
                # Make sure the model is in training mode
                self.model.train()

                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                pred = self.model(data)
                
                # Compute loss
                if self.hparams.loss_function.lower() == "negative log likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, data.y)
                elif self.hparams.loss_function.lower() == "cross entropy":
                    loss = F.cross_entropy(pred, data.y)

                # Save loss information for reporting
                temp_loss_list.append(loss.item())
                temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()

            self.training_accuracy_list.append(np.mean(temp_acc_list).item())
            self.training_loss_list.append(np.mean(temp_loss_list).item())
            self.validate()
            self.validation_accuracy_list.append(self.validation_accuracy)
            self.validation_loss_list.append(self.validation_loss)
            gc.collect()
        
    def validate(self):
        temp_loss_list = []
        temp_acc_list = []
        self.model.eval()
        for data in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            data = data.to(self.device)
            pred = self.model(data)
            if self.hparams.loss_function.lower() == "negative log likelihood":
                logp = F.log_softmax(pred, 1)
                loss = F.nll_loss(logp, data.y)
            elif self.hparams.loss_function.lower() == "cross entropy":
                loss = F.cross_entropy(pred, data.y)
            temp_loss_list.append(loss.item())
            temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))
        self.validation_accuracy = np.mean(temp_acc_list).item()
        self.validation_loss = np.mean(temp_loss_list).item()
    
    def test(self):
        if self.test_dataloader:
            temp_loss_list = []
            temp_acc_list = []
            self.model.eval()
            for data in tqdm(self.test_dataloader, desc='Testing', leave=False):
                data = data.to(self.device)
                pred = self.model(data)
                if self.hparams.loss_function.lower() == "negative log likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, data.y)
                elif self.hparams.loss_function.lower() == "cross entropy":
                    loss = F.cross_entropy(pred, data.y)
                temp_loss_list.append(loss.item())
                temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))
            self.testing_accuracy = np.mean(temp_acc_list).item()
            self.testing_loss = np.mean(temp_loss_list).item()
            
    def save(self, path):
        if path:
            # Make sure the file extension is .pt
            ext = path[-3:]
            if ext.lower() != ".pt":
                path = path + ".pt"
            torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        #self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

class _NodeRegressorHoldout:
    def __init__(self, hparams, trainingDataset, validationDataset=None, testingDataset=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainingDataset = trainingDataset
        self.validationDataset = validationDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        if hparams.conv_layer_type.lower() == 'sageconv':
            # pooling is set None for Node regressor
            self.model = _SAGEConv(trainingDataset[0].num_node_features, hparams.hl_widths, 1, None).to(self.device)
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
        self.node_attr_key = trainingDataset[0].x.shape[1]

        # Train, validate, test split
        num_train = int(len(trainingDataset) * hparams.split[0])
        num_validate = int(len(trainingDataset) * hparams.split[1])
        num_test = len(trainingDataset) - num_train - num_validate
        idx = torch.randperm(len(trainingDataset))
        train_sampler = SubsetRandomSampler(idx[:num_train])
        validate_sampler = SubsetRandomSampler(idx[num_train:num_train+num_validate])
        test_sampler = SubsetRandomSampler(idx[num_train+num_validate:])

        if validationDataset:
            self.train_dataloader = DataLoader(trainingDataset, 
                                               batch_size=hparams.batch_size,
                                               drop_last=False)
            self.validate_dataloader = DataLoader(validationDataset,
                                                  batch_size=hparams.batch_size,
                                                  drop_last=False)
        else:
            self.train_dataloader = DataLoader(trainingDataset, sampler=train_sampler, 
                                               batch_size=hparams.batch_size,
                                               drop_last=False)
            self.validate_dataloader = DataLoader(trainingDataset, sampler=validate_sampler,
                                                  batch_size=hparams.batch_size,
                                                  drop_last=False)
        
        if testingDataset:
            self.test_dataloader = DataLoader(testingDataset,
                                              batch_size=len(testingDataset),
                                              drop_last=False)
        else:
            self.test_dataloader = DataLoader(trainingDataset, sampler=test_sampler,
                                              batch_size=hparams.batch_size,
                                              drop_last=False)

    def train(self):
        # Init the loss and accuracy reporting lists
        self.training_loss_list = []
        self.validation_loss_list = []

        # Run the training loop for defined number of epochs
        for _ in tqdm(range(self.hparams.epochs), desc='Epochs', total=self.hparams.epochs, leave=False):
            # Iterate over the DataLoader for training data
            for data in tqdm(self.train_dataloader, desc='Training', leave=False):
                data = data.to(self.device)
                # Make sure the model is in training mode
                self.model.train()
                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                pred = self.model(data).to(self.device)
                # Compute loss
                loss = F.mse_loss(torch.flatten(pred), data.y.float())

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()

            self.training_loss_list.append(torch.sqrt(loss).item())
            self.validate()
            self.validation_loss_list.append(torch.sqrt(self.validation_loss).item())
            gc.collect()

    def validate(self):
        self.model.eval()
        for data in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            data = data.to(self.device)
            pred = self.model(data).to(self.device)
            loss = F.mse_loss(torch.flatten(pred), data.y.float())
        self.validation_loss = loss
    
    def test(self):
        self.model.eval()
        for data in tqdm(self.test_dataloader, desc='Testing', leave=False):
            data = data.to(self.device)
            pred = self.model(data).to(self.device)
            loss = F.mse_loss(torch.flatten(pred), data.y.float())
        self.testing_loss = torch.sqrt(loss).item()
    
    def save(self, path):
        if path:
            # Make sure the file extension is .pt
            ext = path[-3:]
            if ext.lower() != ".pt":
                path = path + ".pt"
            torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        #self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

class _NodeClassifierKFold:
    def __init__(self, hparams, trainingDataset, testingDataset=None):
        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        self.testing_accuracy = 0
        self.accuracies = []
        self.max_accuracy = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if hparams.conv_layer_type.lower() == 'sageconv':
            # pooling is set None for Node classifier
            self.model = _SAGEConv(trainingDataset.num_node_features, hparams.hl_widths, 
                            trainingDataset.num_classes, None).to(self.device)
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

    def reset_weights(self):
        if self.hparams.conv_layer_type.lower() == 'sageconv':
            # pooling is set None for Node classifier
            self.model = _SAGEConv(self.trainingDataset.num_node_features, self.hparams.hl_widths, 
                            self.trainingDataset.num_classes, None).to(self.device)
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
            self.train_dataloader = DataLoader(self.trainingDataset, sampler=train_subsampler, 
                                                batch_size=self.hparams.batch_size,
                                                drop_last=False)
            self.validate_dataloader = DataLoader(self.trainingDataset, sampler=validate_subsampler,
                                                batch_size=self.hparams.batch_size,
                                                drop_last=False)
            # Init the neural network
            self.reset_weights()

            # Run the training loop for defined number of epochs
            for _ in tqdm(range(0,self.hparams.epochs), desc='Epochs', initial=1, total=self.hparams.epochs, leave=False):
                temp_loss_list = []
                temp_acc_list = []

                # Iterate over the DataLoader for training data
                for data in tqdm(self.train_dataloader, desc='Training', leave=False):
                    data = data.to(self.device)
                    # Make sure the model is in training mode
                    self.model.train()
                    
                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Perform forward pass
                    pred = self.model(data)

                    # Compute loss
                    if self.hparams.loss_function.lower() == "negative log likelihood":
                        logp = F.log_softmax(pred, 1)
                        loss = F.nll_loss(logp, data.y)
                    elif self.hparams.loss_function.lower() == "cross entropy":
                        loss = F.cross_entropy(pred, data.y)

                    # Save loss information for reporting
                    temp_loss_list.append(loss.item())
                    temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    self.optimizer.step()

                epoch_training_accuracy_list.append(np.mean(temp_acc_list).item())
                epoch_training_loss_list.append(np.mean(temp_loss_list).item())
                self.validate()
                epoch_validation_accuracy_list.append(self.validation_accuracy)
                epoch_validation_loss_list.append(self.validation_loss)
                gc.collect()
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
        for data in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            data = data.to(self.device)
            pred = self.model(data)
            if self.hparams.loss_function.lower() == "negative log likelihood":
                logp = F.log_softmax(pred, 1)
                loss = F.nll_loss(logp, data.y)
            elif self.hparams.loss_function.lower() == "cross entropy":
                loss = F.cross_entropy(pred, data.y)
            temp_loss_list.append(loss.item())
            temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))
        self.validation_accuracy = np.mean(temp_acc_list).item()
        self.validation_loss = np.mean(temp_loss_list).item()
    
    def test(self):
        if self.testingDataset:
            self.test_dataloader = DataLoader(self.testingDataset,
                                                    batch_size=len(self.testingDataset),
                                                    drop_last=False)
            temp_loss_list = []
            temp_acc_list = []
            self.model.eval()
            for data in tqdm(self.test_dataloader, desc='Testing', leave=False):
                data = data.to(self.device)
                pred = self.model(data)
                if self.hparams.loss_function.lower() == "negative log likelihood":
                    logp = F.log_softmax(pred, 1)
                    loss = F.nll_loss(logp, data.y)
                elif self.hparams.loss_function.lower() == "cross entropy":
                    loss = F.cross_entropy(pred, data.y)
                temp_loss_list.append(loss.item())
                temp_acc_list.append(accuracy_score(data.y.cpu(), pred.argmax(1).cpu()))
            self.testing_accuracy = np.mean(temp_acc_list).item()
            self.testing_loss = np.mean(temp_loss_list).item()
        
    def save(self, path):
        if path:
            # Make sure the file extension is .pt
            ext = path[-3:]
            if ext.lower() != ".pt":
                path = path + ".pt"
            torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        #self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

class _NodeRegressorKFold:
    def __init__(self, hparams, trainingDataset, testingDataset=None):
        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset
        self.hparams = hparams
        self.losses = []
        self.min_loss = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = self._initialize_model(hparams, trainingDataset)
        self.optimizer = self._initialize_optimizer(hparams)
        
        self.use_gpu = hparams.use_gpu
        self.training_loss_list = []
        self.validation_loss_list = []
        self.node_attr_key = trainingDataset.node_attr_key

        # Train, validate, test split
        num_train = int(len(trainingDataset) * hparams.split[0])
        num_validate = int(len(trainingDataset) * hparams.split[1])
        num_test = len(trainingDataset) - num_train - num_validate
        idx = torch.randperm(len(trainingDataset))
        test_sampler = SubsetRandomSampler(idx[num_train+num_validate:num_train+num_validate+num_test])
        
        if testingDataset:
            self.test_dataloader = DataLoader(testingDataset, batch_size=len(testingDataset), drop_last=False)
        else:
            self.test_dataloader = DataLoader(trainingDataset, sampler=test_sampler, batch_size=hparams.batch_size, drop_last=False)
    
    def _initialize_model(self, hparams, dataset):
        if hparams.conv_layer_type.lower() == 'sageconv':
            # pooling is set None for Node
            return _SAGEConv(dataset.num_node_features, hparams.hl_widths, 1, None).to(self.device)
        else:
            raise NotImplementedError
    
    def _initialize_optimizer(self, hparams):
        if hparams.optimizer_str.lower() == "adadelta":
            return torch.optim.Adadelta(self.model.parameters(), eps=hparams.eps, lr=hparams.lr, rho=hparams.rho, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str.lower() == "adagrad":
            return torch.optim.Adagrad(self.model.parameters(), eps=hparams.eps, lr=hparams.lr, lr_decay=hparams.lr_decay, weight_decay=hparams.weight_decay)
        elif hparams.optimizer_str.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), amsgrad=hparams.amsgrad, betas=hparams.betas, eps=hparams.eps, lr=hparams.lr, maximize=hparams.maximize, weight_decay=hparams.weight_decay)
    
    def reset_weights(self):
        self.model = self._initialize_model(self.hparams, self.trainingDataset)
        self.optimizer = self._initialize_optimizer(self.hparams)
    
    def train(self):
        k_folds = self.hparams.k_folds
        torch.manual_seed(42)
        
        kfold = KFold(n_splits=k_folds, shuffle=True)
        models, weights, losses, train_dataloaders, validate_dataloaders = [], [], [], [], []

        for fold, (train_ids, validate_ids) in tqdm(enumerate(kfold.split(self.trainingDataset)), desc="Fold", total=k_folds, leave=False):
            epoch_training_loss_list, epoch_validation_loss_list = [], []
            train_subsampler = SubsetRandomSampler(train_ids)
            validate_subsampler = SubsetRandomSampler(validate_ids)

            self.train_dataloader = DataLoader(self.trainingDataset, sampler=train_subsampler, batch_size=self.hparams.batch_size, drop_last=False)
            self.validate_dataloader = DataLoader(self.trainingDataset, sampler=validate_subsampler, batch_size=self.hparams.batch_size, drop_last=False)

            self.reset_weights()
            best_rmse = np.inf

            for _ in tqdm(range(self.hparams.epochs), desc='Epochs', total=self.hparams.epochs, leave=False):
                for batched_graph in tqdm(self.train_dataloader, desc='Training', leave=False):
                    self.model.train()
                    self.optimizer.zero_grad()

                    batched_graph = batched_graph.to(self.device)
                    pred = self.model(batched_graph)
                    loss = F.mse_loss(torch.flatten(pred), batched_graph.y.float())
                    loss.backward()
                    self.optimizer.step()

                epoch_training_loss_list.append(torch.sqrt(loss).item())
                self.validate()
                epoch_validation_loss_list.append(torch.sqrt(self.validation_loss).item())
                gc.collect()

            models.append(self.model)
            weights.append(copy.deepcopy(self.model.state_dict()))
            losses.append(torch.sqrt(self.validation_loss).item())
            train_dataloaders.append(self.train_dataloader)
            validate_dataloaders.append(self.validate_dataloader)
            self.training_loss_list.append(epoch_training_loss_list)
            self.validation_loss_list.append(epoch_validation_loss_list)

        self.losses = losses
        self.min_loss = min(losses)
        ind = losses.index(self.min_loss)
        self.model = models[ind]
        self.model.load_state_dict(weights[ind])
        self.model.eval()
        self.training_loss_list = self.training_loss_list[ind]
        self.validation_loss_list = self.validation_loss_list[ind]

    def validate(self):
        self.model.eval()
        for batched_graph in tqdm(self.validate_dataloader, desc='Validating', leave=False):
            batched_graph = batched_graph.to(self.device)
            pred = self.model(batched_graph)
            loss = F.mse_loss(torch.flatten(pred), batched_graph.y.float())
        self.validation_loss = loss
    
    def test(self):
        self.model.eval()
        for batched_graph in tqdm(self.test_dataloader, desc='Testing', leave=False):
            batched_graph = batched_graph.to(self.device)
            pred = self.model(batched_graph)
            loss = F.mse_loss(torch.flatten(pred), batched_graph.y.float())
        self.testing_loss = torch.sqrt(loss).item()
    
    def save(self, path):
        if path:
            ext = path[-3:]
            if ext.lower() != ".pt":
                path = path + ".pt"
            torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

class PyG:
    @staticmethod
    def DatasetByCSVPath(path, numberOfGraphClasses=0, nodeATTRKey='feat', edgeATTRKey='feat', nodeOneHotEncode=False, 
                         nodeFeaturesCategories=[], edgeOneHotEncode=False, edgeFeaturesCategories=[], addSelfLoop=False, 
                         node_level=False, graph_level=True):
        """
        Returns PyTorch Geometric dataset according to the input CSV folder path. The folder must contain "graphs.csv", 
        "edges.csv", "nodes.csv", and "meta.yml" files according to conventions.

        Parameters
        ----------
        path : str
            The path to the folder containing the necessary CSV and YML files.

        Returns
        -------
        PyG Dataset
            The PyG dataset
        """
        if not isinstance(path, str):
            print("PyG.DatasetByCSVPath - Error: The input path parameter is not a valid string. Returning None.")
            return None
        if not os.path.exists(path):
            print("PyG.DatasetByCSVPath - Error: The input path parameter does not exist. Returning None.")
            return None
        
        return CustomGraphDataset(root=path, node_level=node_level, graph_level=graph_level, node_attr_key=nodeATTRKey, edge_attr_key=edgeATTRKey)
    
    @staticmethod
    def DatasetGraphLabels(dataset, graphLabelHeader="label"):
        """
        Returns the labels of the graphs in the input dataset

        Parameters
        ----------
        dataset : CustomDataset
            The input dataset
        graphLabelHeader: str , optional
            The key string under which the graph labels are stored. The default is "label".
        
        Returns
        -------
        list
            The list of graph labels.
        """
        import torch

        graph_labels = []
        for g in dataset:
            # Get the label of the graph
            label = g.y
            graph_labels.append(label.item())
        return graph_labels

    @staticmethod
    def DatasetSplit(dataset, split=[0.8,0.1,0.1], shuffle=True, randomState=42):
        """
        Splits the dataset into three subsets.

        Parameters
        ----------
        dataset : CustomDataset
            The input dataset
        split: list , optional
            The list of ratios. This list must be made out of three numbers adding to 1.
        shuffle: boolean , optional
            If set to True, the subsets are created from random indices. Otherwise, they are split sequentially. The default is True.
        randomState : int , optional
            The random seed to use for reproducibility. The default is 42.
                
        Returns
        -------
        list
            The list of three subset datasets.
        """

        import torch
        from torch.utils.data import random_split
        train_ratio, val_ratio, test_ratio = split
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must add up to 1."
        
        # Calculate the number of samples for each split
        dataset_len = len(dataset)
        train_len = int(train_ratio * dataset_len)
        val_len = int(val_ratio * dataset_len)
        test_len = dataset_len - train_len - val_len  # Ensure it adds up correctly

        ## Generate indices for the split
        indices = list(range(dataset_len))
        if shuffle:
            torch.manual_seed(randomState)  # For reproducibility
            indices = torch.randperm(dataset_len).tolist()  # Shuffled indices

        # Create splits
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:train_len + val_len + test_len]

        # Create new instances of CustomGraphDataset using the indices
        train_dataset = CustomGraphDataset(data_list=dataset.data_list, indices=train_indices)
        val_dataset = CustomGraphDataset(data_list=dataset.data_list, indices=val_indices)
        test_dataset = CustomGraphDataset(data_list=dataset.data_list, indices=test_indices)

        return train_dataset, val_dataset, test_dataset

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
    def Hyperparameters(optimizer, model_type="classifier", cv_type="Holdout", split=[0.8,0.1,0.1], k_folds=5,
                        hl_widths=[32], conv_layer_type="SAGEConv", pooling="AvgPooling",
                        batch_size=1, epochs=1, use_gpu=False, loss_function="Cross Entropy",
                        input_type="graph"):
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
        input_type : str
            selects the input_type of model such as graph, node or edge
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
                        loss_function,
                        input_type)
    
    @staticmethod
    def Model(hparams, trainingDataset, validationDataset=None, testingDataset=None):
        """
        Creates a neural network classifier.

        Parameters
        ----------
        hparams : HParams
            The input hyperparameters 
        trainingDataset : CustomDataset
            The input training dataset.
        validationDataset : CustomDataset
            The input validation dataset. If not specified, a portion of the trainingDataset will be used for validation according the to the split list as specified in the hyper-parameters.
        testingDataset : CustomDataset
            The input testing dataset. If not specified, a portion of the trainingDataset will be used for testing according the to the split list as specified in the hyper-parameters.

        Returns
        -------
        Classifier
            The created classifier

        """

        model = None
        if hparams.model_type.lower() == "classifier":
            if hparams.input_type == 'graph':
                if hparams.cv_type.lower() == "holdout":
                    model = _GraphClassifierHoldout(hparams=hparams, trainingDataset=trainingDataset, validationDataset=validationDataset, testingDataset=testingDataset)
                elif "k" in hparams.cv_type.lower():
                    model = _GraphClassifierKFold(hparams=hparams, trainingDataset=trainingDataset, testingDataset=testingDataset)
            elif hparams.input_type == 'node':
                if hparams.cv_type.lower() == "holdout":
                    model = _NodeClassifierHoldout(hparams=hparams, trainingDataset=trainingDataset, validationDataset=validationDataset, testingDataset=testingDataset)
                elif "k" in hparams.cv_type.lower():
                    model = _NodeClassifierKFold(hparams=hparams, trainingDataset=trainingDataset, testingDataset=testingDataset)
        elif hparams.model_type.lower() == "regressor":
            if hparams.input_type == 'graph':
                if hparams.cv_type.lower() == "holdout":
                    model = _GraphRegressorHoldout(hparams=hparams, trainingDataset=trainingDataset, validationDataset=validationDataset, testingDataset=testingDataset)
                elif "k" in hparams.cv_type.lower():
                    model = _GraphRegressorKFold(hparams=hparams, trainingDataset=trainingDataset, testingDataset=testingDataset)
            elif hparams.input_type == 'node':
                if hparams.cv_type.lower() == "holdout":
                    model = _NodeRegressorHoldout(hparams=hparams, trainingDataset=trainingDataset, validationDataset=validationDataset, testingDataset=testingDataset)
                elif "k" in hparams.cv_type.lower():
                    model = _NodeRegressorKFold(hparams=hparams, trainingDataset=trainingDataset, testingDataset=testingDataset)
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
            print("PyG.ModelSave - Error: The input model parameter is invalid. Returning None.")
            return None
        if path == None:
            print("PyG.ModelSave - Error: The input path parameter is invalid. Returning None.")
            return None
        if not overwrite and os.path.exists(path):
            print("PyG.ModelSave - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        if overwrite and os.path.exists(path):
            os.remove(path)
        # Make sure the file extension is .pt
        ext = path[len(path)-3:len(path)]
        if ext.lower() != ".pt":
            path = path+".pt"
        model.save(path)
        return True
    
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
            try:
                testing_loss_list = [model.testing_loss] * model.hparams.epochs
            except:
                testing_loss_list = [0.] * model.hparams.epochs
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
    def ModelLoad(path, model):
        """
        Returns the model found at the input file path.

        Parameters
        ----------
        path : str
            File path for the saved classifier.
        model : torch.nn.module
            Initialized instance of model

        Returns
        -------
        PyG Classifier
            The classifier.

        """
        if not path:
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

        try:
            from sklearn import metrics
            from sklearn.metrics import accuracy_score
        except:
            print("PyG - Installing required scikit-learn (sklearn) library.")
            try:
                os.system("pip install scikit-learn")
            except:
                os.system("pip install scikit-learn --user")
            try:
                from sklearn import metrics
                from sklearn.metrics import accuracy_score
                print("PyG - scikit-learn (sklearn) library installed correctly.")
            except:
                warnings.warn("PyG - Error: Could not import scikit-learn (sklearn). Please try to install scikit-learn manually. Returning None.")
                return None
            
        if not isinstance(actual, list):
            print("PyG.ConfusionMatrix - ERROR: The actual input is not a list. Returning None")
            return None
        if not isinstance(predicted, list):
            print("PyG.ConfusionMatrix - ERROR: The predicted input is not a list. Returning None")
            return None
        if len(actual) != len(predicted):
            print("PyG.ConfusionMatrix - ERROR: The two input lists do not have the same length. Returning None")
            return None
        if normalize:
            cm = np.transpose(metrics.confusion_matrix(y_true=actual, y_pred=predicted, normalize="true"))
        else:
            cm = np.transpose(metrics.confusion_matrix(y_true=actual, y_pred=predicted))
        return cm

    @staticmethod
    def ModelPredict(model, dataset, nodeATTRKey="feat"):
        """
        Predicts the value of the input dataset.

        Parameters
        ----------
        dataset : PyGDataset
            The input PyG dataset.
        model : Model
            The input trained model.
        nodeATTRKey : str , optional
            The key used for node attributes. The default is "feat".
    
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
        dataloader = DataLoader(dataset, batch_size=1, drop_last=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()
        for data in tqdm(dataloader, desc='Predicting', leave=False):
            data = data.to(device)
            pred = model(data)
            values.extend(list(np.round(pred.detach().cpu().numpy().flatten(), 3)))
        return values

    @staticmethod
    def ModelClassify(model, dataset, nodeATTRKey="feat"):
        """
        Predicts the classification the labels of the input dataset.

        Parameters
        ----------
        dataset : PyGDataset
            The input PyG dataset.
        model : Model
            The input trained model.
        nodeATTRKey : str , optional
            The key used for node attributes. The default is "feat".

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
        dataloader = DataLoader(dataset, batch_size=1, drop_last=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for data in tqdm(dataloader, desc='Classifying', leave=False):
            data = data.to(device)
            pred = model(data)
            labels.extend(pred.argmax(1).tolist())
            probability = (torch.nn.functional.softmax(pred, dim=1).tolist())
            probability = probability[0]
            temp_probability = []
            for p in probability:
                temp_probability.append(round(p, 3))
            probabilities.extend(temp_probability)
        return {"predictions":labels, "probabilities":probabilities}
    
    @staticmethod
    def Accuracy(actual, predicted, mantissa: int = 6):
        """
        Computes the accuracy of the input predictions based on the input labels. This is to be used only with classification not with regression.

        Parameters
        ----------
        actual : list
            The input list of actual values.
        predicted : list
            The input list of predicted values.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

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
    def MSE(actual, predicted, mantissa: int = 6):
        """
        Computes the Mean Squared Error (MSE) of the input predictions based on the input labels. This is to be used with regression models.

        Parameters
        ----------
        actual : list
            The input list of actual values.
        predicted : list
            The input list of predicted values.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        dict
            A dictionary returning the MSE information. This contains the following keys and values:
            - "mse" (float): The mean squared error rounded to the specified mantissa.
            - "size" (int): The size of the predictions list.
        """
        if len(predicted) < 1 or len(actual) < 1 or not len(predicted) == len(actual):
            return None
        
        mse = np.mean((np.array(predicted) - np.array(actual)) ** 2)
        mse = round(mse, mantissa)
        size = len(predicted)

        return {"mse": mse, "size": size}

    @staticmethod
    def Performance(actual, predicted, mantissa: int = 6):
        """
        Computes regression model performance measures. This is to be used only with regression not with classification.

        Parameters
        ----------
        actual : list
            The input list of actual values.
        predicted : list
            The input list of predicted values.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
        Returns
        -------
        dict
            The dictionary containing the performance measures. The keys in the dictionary are: 'mae', 'mape', 'mse', 'r', 'r2', 'rmse'.
        """
        
        if not isinstance(actual, list):
            print("PyG.Performance - ERROR: The actual input is not a list. Returning None")
            return None
        if not isinstance(predicted, list):
            print("PyG.Performance - ERROR: The predicted input is not a list. Returning None")
            return None
        if not (len(actual) == len(predicted)):
            print("PyG.Performance - ERROR: The actual and predicted input lists have different lengths. Returning None")
            return None
        
        predicted = np.array(predicted)
        actual = np.array(actual)

        mae = np.mean(np.abs(predicted - actual))
        mape = np.mean(np.abs((actual - predicted) / actual))*100
        mse = np.mean((predicted - actual)**2)
        correlation_matrix = np.corrcoef(predicted, actual)
        r = correlation_matrix[0, 1]
        r2 = r**2
        absolute_errors = np.abs(predicted - actual)
        mean_actual = np.mean(actual)
        if mean_actual == 0:
            rae = None
        else:
            rae = np.mean(absolute_errors) / mean_actual
        rmse = np.sqrt(mse)
        return {'mae': round(mae, mantissa),
                'mape': round(mape, mantissa),
                'mse': round(mse, mantissa),
                'r': round(r, mantissa),
                'r2': round(r2, mantissa),
                'rae': round(rae, mantissa),
                'rmse': round(rmse, mantissa)
                }