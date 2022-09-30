import dgl
import topologic
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dgl.data import DGLDataset
from dgl.nn import GraphConv
import Dictionary
import os
import plotly.express as px
import Replication
import random

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

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.list_of_layers = []
        if not isinstance(h_feats, list):
            h_feats = [h_feats]
        dim = [in_feats] + h_feats
        for i in range(1, len(dim)):
            self.list_of_layers.append(GraphConv(dim[i-1], dim[i]))
        self.list_of_layers = nn.ModuleList(self.list_of_layers)
        self.final = GraphConv(dim[-1], num_classes)

    def forward(self, g, in_feat):
        h = in_feat
        for i in range(len(self.list_of_layers)):
            h = self.list_of_layers[i](g, h)
            h = F.relu(h)
        h = self.final(g, h)
        return h

class DGL:
    @staticmethod
    def DGLAccuracy(dgl_labels, dgl_predictions):
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
    def DGLClassifierByFilePath(item):
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
    def DGLDatasetByDGLGraph(dgl_graphs, dgl_labels, node_attr_key):
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
    def DGLDatasetByImportedCSV_NC(item):
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
    def DGLDatasetBySamples(item):
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
        TYPE
            DESCRIPTION.

        """
        sample = item
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
    def DGLDatasetBySamples_NC(item):
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
        sample = item
        if sample == 'Cora':
            return [dgl.data.CoraGraphDataset(), 7]
        elif sample == 'Citeseer':
            return [dgl.data.CiteseerGraphDataset(), 6]
        elif sample == 'Pubmed':
            return [dgl.data.PubmedGraphDataset(), 3]
        else:
            raise NotImplementedError
    
    @staticmethod
    def DGLDatasetGraphs_NC(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        graphs : TYPE
            DESCRIPTION.

        """
        dataset = item
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
    def oneHotEncode(item, categories):
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
    def DGLGraphByGraph(graph, bidirectional, key, categories, node_attr_key, tolerance=0.0001):
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
            vLabel = Dictionary.DictionaryValueAtKey(vDict, key)
            graph_dict["node_labels"][i] = vLabel
            # appending tensor of onehotencoded feature for each node following index i
            graph_dict["node_features"].append(torch.tensor(DGL.oneHotEncode(vLabel, categories)))
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
    def DGLGraphByImportedCSV(graphs_file_path, edges_file_path,
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
                graph_dict["node_features"].append(torch.tensor(DGL.oneHotEncode(node_label, categories)))
            # Create a graph and add it to the list of graphs and labels.
            dgl_graph = dgl.graph((src, dst), num_nodes=num_nodes)
            # Setting the node features as node_attr_key using onehotencoding of node_label
            dgl_graph.ndata[node_attr_key] = torch.stack(graph_dict["node_features"])
            if bidirectional:
                dgl_graph = dgl.add_reverse_edges(dgl_graph)        
            dgl_graphs.append(dgl_graph)
        return [dgl_graphs, labels]

    @staticmethod
    def DGLGraphByImportedDGCNN(file_path, categories, bidirectional):
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
                    graph_dict["node_features"].append(torch.tensor(DGL.oneHotEncode(node_label, categories)))
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
    def DGLGraphEdgeData_NC(item):
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
        return item.edata
    
    @staticmethod
    def DGLGraphNodeData_NC(item):
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
        return item.ndata
    
    @staticmethod
    def DGLHyperparameters(optimizer, cv_type, split, k_folds,
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
    def DGLPlot(data, data_labels, chart_title, x_title, x_spacing, y_title,
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
        chart_type : TYPE
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
        if chart_type == "Line":
            fig = px.line(df, x = data_labels[0], y=data_labels[1:], title=chart_title, markers=use_markers)
        elif chart_type == "Bar":
            fig = px.bar(df, x = data_labels[0], y=data_labels[1:], title=chart_title)
        elif chart_type == "Scatter":
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
    def DGLPredict(test_dataset, classifier, node_attr_key):
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
    def DGLPredict_NC(classifier, dataset):
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
            
        return [Replication.flatten(allLabels), Replication.flatten(allPredictions),Replication.flatten(trainLabels), Replication.flatten(trainPredictions), Replication.flatten(valLabels), Replication.flatten(valPredictions), Replication.flatten(testLabels), Replication.flatten(testPredictions)]
    
    @staticmethod
    def train2(graphs, model, hparams):
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
    def DGLTrainClassifier_NC(hparams, dataset, numLabels, sample):
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
    
    
    
    