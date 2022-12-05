'''
    @staticmethod
    def adjacentVertices(graph, vertex):
        vertices = []
        _ = graph.AdjacentVertices(vertex, vertices)
        return list(vertices)
    
    @staticmethod
    def vertexIndex(vertex, vertices):
        for i in range(len(vertices)):
            if topologic.Topology.IsSame(vertex, vertices[i]):
                return i
        return None

    @staticmethod
    def ExportToCSV(graph_list, graph_label_list, graphs_file_path, 
                         edges_file_path, nodes_file_path, graph_id_header,
                         graph_label_header, graph_num_nodes_header,
                         edge_src_header, edge_dst_header, node_label_header,
                         node_label_key, default_node_label, overwrite):
        """
        Description
        -----------
        Creates a vertex at the coordinates specified by the x, y, z inputs.

        Parameters
        ----------
        graph_list : TYPE
            DESCRIPTION.
        graph_label_list : TYPE
            DESCRIPTION.
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
        graph_num_nodes_header : TYPE
            DESCRIPTION.
        edge_src_header : TYPE
            DESCRIPTION.
        edge_dst_header : TYPE
            DESCRIPTION.
        node_label_header : TYPE
            DESCRIPTION.
        node_label_key : TYPE
            DESCRIPTION.
        default_node_label : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        # graph_list, \
        # graph_label_list, \
        # graphs_file_path, \
        # edges_file_path, \
        # nodes_file_path, \
        # graph_id_header, \
        # graph_label_header, \
        # graph_num_nodes_header, \
        # edge_src_header, \
        # edge_dst_header, \
        # node_label_header, \
        # node_label_key, \
        # default_node_label, \
        # overwrite = item

        if not isinstance(graph_list, list):
            graph_list = [graph_list]
        for graph_index, graph in enumerate(graph_list):
            graph_label = graph_label_list[graph_index]
            # Export Graph Properties
            vertices = Graph.graphVertices(graph)
            graph_num_nodes = len(vertices)
            if overwrite == False:
                graphs = pd.read_csv(graphs_file_path)
                max_id = max(list(graphs[graph_id_header]))
                graph_id = max_id + graph_index + 1
            else:
                graph_id = graph_index
            data = [[graph_id], [graph_label], [graph_num_nodes]]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, graph_label_header, graph_num_nodes_header])
            if overwrite == False:
                df.to_csv(graphs_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(graphs_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(graphs_file_path, mode='a', index = False, header=False)

            # Export Edge Properties
            edge_src = []
            edge_dst = []
            edge_graph_id = [] #Repetitive list of graph_id for each edge
            node_graph_id = [] #Repetitive list of graph_id for each vertex/node
            node_labels = []
            x_list = []
            y_list = []
            z_list = []
            node_data = []
            node_columns = [graph_id_header, node_label_header, "X", "Y", "Z"]
            # All keys should be the same for all vertices, so we can get them from the first vertex
            d = vertices[0].GetDictionary()
            keys = d.Keys()
            for key in keys:
                if key != node_label_key: #We have already saved that in its own column
                    node_columns.append(key)
            for i, v in enumerate(vertices):
                # Might as well get the node labels since we are iterating through the vertices
                d = v.GetDictionary()
                vLabel = Dictionary.DictionaryValueAtKey(d, node_label_key)
                if not(vLabel):
                    vLabel = default_node_label        
                single_node_data = [graph_id, vLabel, round(float(v.X()),5), round(float(v.Y()),5), round(float(v.Z()),5)]
                keys = d.Keys()
                for key in keys:
                    if key != node_label_key and (key in node_columns):
                        value = Dictionary.DictionaryValueAtKey(d, key)
                        if not value:
                            value = 'None'
                        single_node_data.append(value)
                node_data.append(single_node_data)
                av = Graph.adjacentVertices(graph, v)
                for k in range(len(av)):
                    vi = Graph.vertexIndex(av[k], vertices)
                    edge_graph_id.append(graph_id)
                    edge_src.append(i)
                    edge_dst.append(vi)
            data = [edge_graph_id, edge_src, edge_dst]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, edge_src_header, edge_dst_header])
            if overwrite == False:
                df.to_csv(edges_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(edges_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(edges_file_path, mode='a', index = False, header=False)

            # Export Node Properties
            df = pd.DataFrame(node_data, columns= node_columns)

            if overwrite == False:
                df.to_csv(nodes_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(nodes_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(nodes_file_path, mode='a', index = False, header=False)
        return True

    
    @staticmethod
    def ExportToCSV_NC(graph_list, graph_label_list, graphs_folder_path,
                            node_label_key, node_features_keys, default_node_label, edge_label_key,
                            edge_features_keys, default_edge_label,
                            train_ratio, test_ratio, validate_ratio,
                            overwrite):
        """
        Description
        -----------
        Creates a vertex at the coordinates specified by the x, y, z inputs.

        Parameters
        ----------
        graph_list : TYPE
            DESCRIPTION.
        graph_label_list : TYPE
            DESCRIPTION.
        graphs_folder_path : TYPE
            DESCRIPTION.
        node_label_key : TYPE
            DESCRIPTION.
        node_features_keys : TYPE
            DESCRIPTION.
        default_node_label : TYPE
            DESCRIPTION.
        edge_label_key : TYPE
            DESCRIPTION.
        edge_features_keys : TYPE
            DESCRIPTION.
        default_edge_label : TYPE
            DESCRIPTION.
        train_ratio : TYPE
            DESCRIPTION.
        test_ratio : TYPE
            DESCRIPTION.
        validate_ratio : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # graph_list, \
        # graph_label_list, \
        # graphs_folder_path, \
        # node_label_key, \
        # node_features_keys, \
        # default_node_label, \
        # edge_label_key, \
        # edge_features_keys, \
        # default_edge_label, \
        # train_ratio, \
        # test_ratio, \
        # validate_ratio, \
        # overwrite = item
        
        def graphVertices(graph):
            import random
            vertices = []
            if graph:
                try:
                    _ = graph.Vertices(vertices)
                except:
                    print("ERROR: (Topologic>Graph.Vertices) operation failed. Returning None.")
                    vertices = None
            if vertices:
                return random.sample(vertices, len(vertices))
            else:
                return []

        assert (train_ratio+test_ratio+validate_ratio > 0.99), "GraphExportToCSV_NC - Error: Train_Test_Validate ratios do not add up to 1."

        if not isinstance(graph_list, list):
            graph_list = [graph_list]
        for graph_index, graph in enumerate(graph_list):
            graph_label = graph_label_list[graph_index]
            # Export Graph Properties
            vertices = graphVertices(graph)
            train_max = math.floor(float(len(vertices))*train_ratio)
            test_max = math.floor(float(len(vertices))*test_ratio)
            validate_max = len(vertices) - train_max - test_max
            graph_num_nodes = len(vertices)
            if overwrite == False:
                graphs = pd.read_csv(os.path.join(graphs_folder_path,"graphs.csv"))
                max_id = max(list(graphs["graph_id"]))
                graph_id = max_id + graph_index + 1
            else:
                graph_id = graph_index
            data = [[graph_id], [graph_label], [graph_num_nodes]]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= ["graph_id", "label", "num_nodes"])
            if overwrite == False:
                df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='a', index = False, header=False)

            # Export Edge Properties
            edge_graph_id = [] #Repetitive list of graph_id for each edge
            edge_src = []
            edge_dst = []
            edge_lab = []
            edge_feat = []
            node_graph_id = [] #Repetitive list of graph_id for each vertex/node
            node_labels = []
            x_list = []
            y_list = []
            z_list = []
            node_data = []
            node_columns = ["graph_id", "node_id","label", "train_mask","val_mask","test_mask","feat", "X", "Y", "Z"]
            # All keys should be the same for all vertices, so we can get them from the first vertex
            d = vertices[0].GetDictionary()
            '''
            keys = d.Keys()
            for key in keys:
                if key != node_label_key: #We have already saved that in its own column
                    node_columns.append(key)
            '''
            train = 0
            test = 0
            validate = 0
            
            for i, v in enumerate(vertices):
                if train < train_max:
                    train_mask = True
                    test_mask = False
                    validate_mask = False
                    train = train + 1
                elif test < test_max:
                    train_mask = False
                    test_mask = True
                    validate_mask = False
                    test = test + 1
                elif validate < validate_max:
                    train_mask = False
                    test_mask = False
                    validate_mask = True
                    validate = validate + 1
                else:
                    train_mask = True
                    test_mask = False
                    validate_mask = False
                    train = train + 1
                # Might as well get the node labels since we are iterating through the vertices
                d = v.GetDictionary()
                vLabel = Dictionary.DictionaryValueAtKey(d, node_label_key)
                if not(vLabel):
                    vLabel = default_node_label
                # Might as well get the features since we are iterating through the vertices
                features = ""
                node_features_keys = Replication.flatten(node_features_keys)
                for node_feature_key in node_features_keys:
                    if len(features) > 0:
                        features = features + ","+ str(round(float(Dictionary.DictionaryValueAtKey(d, node_feature_key)),5))
                    else:
                        features = str(round(float(Dictionary.DictionaryValueAtKey(d, node_feature_key)),5))
                single_node_data = [graph_id, i, vLabel, train_mask, validate_mask, test_mask, features, round(float(v.X()),5), round(float(v.Y()),5), round(float(v.Z()),5)]
                '''
                keys = d.Keys()
                for key in keys:
                    if key != node_label_key and (key in node_columns):
                        value = DictionaryValueAtKey.processItem([d, key])
                        if not value:
                            value = 'None'
                        single_node_data.append(value)
                '''
                node_data.append(single_node_data)
                av = Graph.adjacentVertices(graph, v)
                for k in range(len(av)):
                    vi = Graph.vertexIndex(av[k], vertices)
                    edge_graph_id.append(graph_id)
                    edge_src.append(i)
                    edge_dst.append(vi)
                    edge = graph.Edge(v, av[k], 0.0001)
                    ed = edge.GetDictionary()
                    edge_label = Dictionary.DictionaryValueAtKey(d, edge_label_key)
                    if not(edge_label):
                        edge_label = default_edge_label
                    edge_lab.append(edge_label)
                    edge_features = ""
                    edge_features_keys = Replication.flatten(edge_features_keys)
                    for edge_feature_key in edge_features_keys:
                        if len(edge_features) > 0:
                            edge_features = edge_features + ","+ str(round(float(Dictionary.DictionaryValueAtKey(ed, edge_feature_key)),5))
                        else:
                            edge_features = str(round(float(Dictionary.DictionaryValueAtKey(ed, edge_feature_key)),5))
                    edge_feat.append(edge_features)
            data = [edge_graph_id, edge_src, edge_dst, edge_lab, edge_feat]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= ["graph_id", "src_id", "dst_id", "label", "feat"])
            if overwrite == False:
                df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='a', index = False, header=False)

            # Export Node Properties
            df = pd.DataFrame(node_data, columns= node_columns)

            if overwrite == False:
                df.to_csv(os.path.join(graphs_folder_path, "nodes.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "nodes.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "nodes.csv"), mode='a', index = False, header=False)
        # Write out the meta.yaml file
        yaml_file = open(os.path.join(graphs_folder_path,"meta.yaml"), "w")
        yaml_file.write('dataset_name: topologic_dataset\nedge_data:\n- file_name: edges.csv\nnode_data:\n- file_name: nodes.csv\ngraph_data:\n  file_name: graphs.csv')
        yaml_file.close()
        return True
    
    @staticmethod
    def ExportToCSVGC(graph_list, graph_label_list, graphs_file_path, edges_file_path,
                           nodes_file_path, graph_id_header, graph_label_header, graph_num_nodes_header, 
                           edge_src_header, edge_dst_header, node_label_header, node_label_key, default_node_label, overwrite):
        """
        Description
        -----------
        Creates a vertex at the coordinates specified by the x, y, z inputs.

        Parameters
        ----------
        graph_list : TYPE
            DESCRIPTION.
        graph_label_list : TYPE
            DESCRIPTION.
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
        graph_num_nodes_header : TYPE
            DESCRIPTION.
        edge_src_header : TYPE
            DESCRIPTION.
        edge_dst_header : TYPE
            DESCRIPTION.
        node_label_header : TYPE
            DESCRIPTION.
        node_label_key : TYPE
            DESCRIPTION.
        default_node_label : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        # graph_list, \
        # graph_label_list, \
        # graphs_file_path, \
        # edges_file_path, \
        # nodes_file_path, \
        # graph_id_header, \
        # graph_label_header, \
        # graph_num_nodes_header, \
        # edge_src_header, \
        # edge_dst_header, \
        # node_label_header, \
        # node_label_key, \
        # default_node_label, \
        # overwrite = item

        if not isinstance(graph_list, list):
            graph_list = [graph_list]
        for graph_index, graph in enumerate(graph_list):
            graph_label = graph_label_list[graph_index]
            # Export Graph Properties
            vertices = Graph.graphVertices(graph)
            graph_num_nodes = len(vertices)
            if overwrite == False:
                graphs = pd.read_csv(graphs_file_path)
                max_id = max(list(graphs[graph_id_header]))
                graph_id = max_id + graph_index + 1
            else:
                graph_id = graph_index
            data = [[graph_id], [graph_label], [graph_num_nodes]]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, graph_label_header, graph_num_nodes_header])
            if overwrite == False:
                df.to_csv(graphs_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(graphs_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(graphs_file_path, mode='a', index = False, header=False)

            # Export Edge Properties
            edge_src = []
            edge_dst = []
            edge_graph_id = [] #Repetitive list of graph_id for each edge
            node_graph_id = [] #Repetitive list of graph_id for each vertex/node
            node_labels = []
            x_list = []
            y_list = []
            z_list = []
            node_data = []
            node_columns = [graph_id_header, node_label_header, "X", "Y", "Z"]
            # All keys should be the same for all vertices, so we can get them from the first vertex
            d = vertices[0].GetDictionary()
            keys = d.Keys()
            for key in keys:
                if key != node_label_key: #We have already saved that in its own column
                    node_columns.append(key)
            for i, v in enumerate(vertices):
                # Might as well get the node labels since we are iterating through the vertices
                d = v.GetDictionary()
                vLabel = Dictionary.DictionaryValueAtKey(d, node_label_key)
                if not(vLabel):
                    vLabel = default_node_label        
                single_node_data = [graph_id, vLabel, round(float(v.X()),5), round(float(v.Y()),5), round(float(v.Z()),5)]
                keys = d.Keys()
                for key in keys:
                    if key != node_label_key and (key in node_columns):
                        value = Dictionary.DictionaryValueAtKey(d, key)
                        if not value:
                            value = 'None'
                        single_node_data.append(value)
                node_data.append(single_node_data)
                av = Graph.adjacentVertices(graph, v)
                for k in range(len(av)):
                    vi = Graph.vertexIndex(av[k], vertices)
                    edge_graph_id.append(graph_id)
                    edge_src.append(i)
                    edge_dst.append(vi)
            data = [edge_graph_id, edge_src, edge_dst]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, edge_src_header, edge_dst_header])
            if overwrite == False:
                df.to_csv(edges_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(edges_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(edges_file_path, mode='a', index = False, header=False)

            # Export Node Properties
            df = pd.DataFrame(node_data, columns= node_columns)

            if overwrite == False:
                df.to_csv(nodes_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(nodes_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(nodes_file_path, mode='a', index = False, header=False)
        return True
    
    @staticmethod
    def ExportToCSVNC(graph_list, graph_label_list, graphs_folder_path, graph_id_header,
                           graph_label_header, graph_num_nodes_header, edge_src_header, edge_dst_header,
                           node_label_header, node_label_key, node_features_keys, default_node_label, overwrite):
        """
        Description
        -----------
        Creates a vertex at the coordinates specified by the x, y, z inputs.

        Parameters
        ----------
        graph_list : TYPE
            DESCRIPTION.
        graph_label_list : TYPE
            DESCRIPTION.
        graphs_folder_path : TYPE
            DESCRIPTION.
        graph_id_header : TYPE
            DESCRIPTION.
        graph_label_header : TYPE
            DESCRIPTION.
        graph_num_nodes_header : TYPE
            DESCRIPTION.
        edge_src_header : TYPE
            DESCRIPTION.
        edge_dst_header : TYPE
            DESCRIPTION.
        node_label_header : TYPE
            DESCRIPTION.
        node_label_key : TYPE
            DESCRIPTION.
        node_features_keys : TYPE
            DESCRIPTION.
        default_node_label : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        # graph_list, \
        # graph_label_list, \
        # graphs_folder_path, \
        # graph_id_header, \
        # graph_label_header, \
        # graph_num_nodes_header, \
        # edge_src_header, \
        # edge_dst_header, \
        # node_label_header, \
        # node_label_key, \
        # node_features_keys, \
        # default_node_label, \
        # overwrite = item

        if not isinstance(graph_list, list):
            graph_list = [graph_list]
        for graph_index, graph in enumerate(graph_list):
            graph_label = graph_label_list[graph_index]
            # Export Graph Properties
            vertices = Graph.graphVertices(graph)
            graph_num_nodes = len(vertices)
            if overwrite == False:
                graphs = pd.read_csv(graphs_folder_path)
                max_id = max(list(graphs[graph_id_header]))
                graph_id = max_id + graph_index + 1
            else:
                graph_id = graph_index
            data = [[graph_id], [graph_label], [graph_num_nodes]]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, graph_label_header, graph_num_nodes_header])
            if overwrite == False:
                df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='a', index = False, header=False)

            # Export Edge Properties
            edge_src = []
            edge_dst = []
            edge_graph_id = [] #Repetitive list of graph_id for each edge
            node_graph_id = [] #Repetitive list of graph_id for each vertex/node
            node_labels = []
            x_list = []
            y_list = []
            z_list = []
            node_data = []
            node_columns = [graph_id_header, node_label_header, "feat", "X", "Y", "Z"]
            # All keys should be the same for all vertices, so we can get them from the first vertex
            d = vertices[0].GetDictionary()
            keys = d.Keys()
            for key in keys:
                if key != node_label_key: #We have already saved that in its own column
                    node_columns.append(key)
            for i, v in enumerate(vertices):
                # Might as well get the node labels since we are iterating through the vertices
                d = v.GetDictionary()
                vLabel = Dictionary.DictionaryValueAtKey(d, node_label_key)
                if not(vLabel):
                    vLabel = default_node_label
                # Might as well get the features since we are iterating through the vertices
                features = ""
                for node_feature_key in node_features_keys:
                    if len(features) > 0:
                        features = features + ","+ str(round(float(Dictionary.DictionaryValueAtKey(d, node_feature_key)),5))
                    else:
                        features = str(round(float(Dictionary.DictionaryValueAtKey(d, node_feature_key)),5))
                single_node_data = [graph_id, vLabel, features, round(float(v.X()),5), round(float(v.Y()),5), round(float(v.Z()),5)]
                keys = d.Keys()
                for key in keys:
                    if key != node_label_key and (key in node_columns):
                        value = Dictionary.DictionaryValueAtKey(d, key)
                        if not value:
                            value = 'None'
                        single_node_data.append(value)
                node_data.append(single_node_data)
                av = Graph.adjacentVertices(graph, v)
                for k in range(len(av)):
                    vi = Graph.vertexIndex(av[k], vertices)
                    edge_graph_id.append(graph_id)
                    edge_src.append(i)
                    edge_dst.append(vi)
            data = [edge_graph_id, edge_src, edge_dst]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, edge_src_header, edge_dst_header])
            if overwrite == False:
                df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='a', index = False, header=False)

            # Export Node Properties
            df = pd.DataFrame(node_data, columns= node_columns)

            if overwrite == False:
                df.to_csv(nodes_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "nodes.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "nodes.csv"), mode='a', index = False, header=False)
        return True
    
    @staticmethod
    def ExportToDGCNN(graph, graph_label, key, default_vertex_label, filepath, overwrite):
        """
        Description
        -----------
        Creates a vertex at the coordinates specified by the x, y, z inputs.

        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        graph_label : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.
        default_vertex_label : TYPE
            DESCRIPTION.
        filepath : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        # graph, graph_label, key, default_vertex_label, filepath, overwrite = item
        vertices = Graph.graphVertices(graph)
        new_lines = []
        new_lines.append("\n"+str(len(vertices))+" "+str(graph_label))
        for j in range(len(vertices)):
            d = vertices[j].GetDictionary()
            vLabel = Dictionary.DictionaryValueAtKey(d, key)
            if not(vLabel):
                vLabel = default_vertex_label
            av = Graph.adjacentVertices(graph, vertices[j])
            line = "\n"+str(vLabel)+" "+ str(len(av))+" "
            for k in range(len(av)):
                vi = Graph.vertexIndex(av[k], vertices)
                line = line+str(vi)+" "
            new_lines.append(line)
        # Make sure the file extension is .txt
        ext = filepath[len(filepath)-4:len(filepath)]
        if ext.lower() != ".txt":
            filepath = filepath+".txt"
        old_lines = ["1"]
        if overwrite == False:
            with open(filepath) as f:
                old_lines = f.readlines()
                if len(old_lines):
                    if old_lines[0] != "":
                        old_lines[0] = str(int(old_lines[0])+1)+"\n"
                else:
                    old_lines[0] = "1"
        lines = old_lines+new_lines
        with open(filepath, "w") as f:
            f.writelines(lines)
        return True
    '''
    @staticmethod
    def VerticesAtKeyValue(vertexList, key, value):
        """
        Description
        -----------
        Creates a vertex at the coordinates specified by the x, y, z inputs.

        Parameters
        ----------
        vertexList : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.
        value : TYPE
            DESCRIPTION.

        Returns
        -------
        returnVertices : TYPE
            DESCRIPTION.

        """
        # key = item[0]
        # value = item[1]
        
        def listAttributeValues(listAttribute):
            listAttributes = listAttribute.ListValue()
            returnList = []
            for attr in listAttributes:
                if isinstance(attr, IntAttribute):
                    returnList.append(attr.IntValue())
                elif isinstance(attr, DoubleAttribute):
                    returnList.append(attr.DoubleValue())
                elif isinstance(attr, StringAttribute):
                    returnList.append(attr.StringValue())
            return returnList

        def valueAtKey(item, key):
            try:
                attr = item.ValueAtKey(key)
            except:
                raise Exception("Dictionary.ValueAtKey - Error: Could not retrieve a Value at the specified key ("+key+")")
            if isinstance(attr, topologic.IntAttribute):
                return (attr.IntValue())
            elif isinstance(attr, topologic.DoubleAttribute):
                return (attr.DoubleValue())
            elif isinstance(attr, topologic.StringAttribute):
                return (attr.StringValue())
            elif isinstance(attr, topologic.ListAttribute):
                return (listAttributeValues(attr))
            else:
                return None

        if isinstance(value, list):
            value.sort()
        returnVertices = []
        for aVertex in vertexList:
            d = aVertex.GetDictionary()
            v = valueAtKey(d, key)
            if isinstance(v, list):
                v.sort()
            if str(v) == str(value):
                returnVertices.append(aVertex)
        return returnVertices

@staticmethod
    def VisibilityGraph(cluster):
        """
        Description
        -----------
        Creates a vertex at the coordinates specified by the x, y, z inputs.

        Parameters
        ----------
        cluster : TYPE
            DESCRIPTION.

        Returns
        -------
        graph : TYPE
            DESCRIPTION.

        """
        wires = []
        _ = cluster.Wires(None, wires)
        polys = []
        for aWire in wires:
            vertices = []
            _ = aWire.Vertices(None, vertices)
            poly = []
            for v in vertices:
                p = vg.Point(round(v.X(),4),round(v.Y(),4), 0)
                poly.append(p)
            polys.append(poly)
        g = vg.VisGraph()
        g.build(polys)
        tpEdges = []
        vgEdges = g.visgraph.get_edges()
        for vgEdge in vgEdges:
            sv = topologic.Vertex.ByCoordinates(vgEdge.p1.x, vgEdge.p1.y,0)
            ev = topologic.Vertex.ByCoordinates(vgEdge.p2.x, vgEdge.p2.y,0)
            tpEdges.append(topologic.Edge.ByStartVertexEndVertex(sv, ev))
        tpVertices = []
        vgPoints = g.visgraph.get_points()
        for vgPoint in vgPoints:
            v = topologic.Vertex.ByCoordinates(vgPoint.x, vgPoint.y,0)
            tpVertices.append(v)
        graph = topologic.Graph(tpVertices, tpEdges)
        return graph
    