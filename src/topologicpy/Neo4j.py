# Copyright (C) 2026
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

import random
import warnings

try:
    import neo4j
    from neo4j import GraphDatabase
except Exception:
    warnings.warn("Neo4j - Error: Could not import neo4j. Please install it using pip install neo4j.")
    neo4j = None
    GraphDatabase = None


class Neo4j:
    @staticmethod
    def _is_driver(driver):
        """
        Returns True if the input object appears to be a valid Neo4j driver.

        Parameters
        ----------
        driver : object
            The input object.

        Returns
        -------
        bool
            True if the input object appears to be a valid Neo4j driver.
            False otherwise.
        """
        return driver is not None and hasattr(driver, "session") and hasattr(driver, "close")

    @staticmethod
    def _sanitize_identifier(value, default="X"):
        """
        Returns a Neo4j-safe identifier.

        Parameters
        ----------
        value : any
            The input value.
        default : str , optional
            The default value to use if the input is invalid. Default is "X".

        Returns
        -------
        str
            The sanitized identifier.
        """
        import re

        if value is None:
            value = default
        value = str(value).strip()
        if len(value) < 1:
            value = default
        value = re.sub(r"[^0-9a-zA-Z_]", "_", value)
        if len(value) < 1:
            value = default
        if not value[0].isalpha() and value[0] != "_":
            value = "_"+value
        return value

    @staticmethod
    def _node_properties(node):
        """
        Returns a Python dictionary of the input Neo4j node properties.

        Parameters
        ----------
        node : neo4j.graph.Node
            The input Neo4j node.

        Returns
        -------
        dict
            The dictionary of properties.
        """
        props = dict(node.items())
        props["id"] = getattr(node, "element_id", None)
        props["labels"] = list(getattr(node, "labels", []))
        return props

    @staticmethod
    def _relationship_properties(relationship):
        """
        Returns a Python dictionary of the input Neo4j relationship properties.

        Parameters
        ----------
        relationship : neo4j.graph.Relationship
            The input Neo4j relationship.

        Returns
        -------
        dict
            The dictionary of properties.
        """
        props = dict(relationship.items())
        props["id"] = getattr(relationship, "element_id", None)
        props["type"] = getattr(relationship, "type", None)
        return props

    @staticmethod
    def Connect(url, username, password, database=None, silent=False):
        """
        Returns a Neo4j driver created from the input connection parameters.

        Parameters
        ----------
        url : str
            The Neo4j server URL.
        username : str
            The username.
        password : str
            The password.
        database : str , optional
            The default database to test against. If set to None, the driver's
            default database is used. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        neo4j.Driver
            The created Neo4j driver.
        """
        if GraphDatabase is None:
            if not silent:
                print("Neo4j.Connect - Error: Could not import neo4j. Returning None.")
            return None

        try:
            driver = GraphDatabase.driver(url, auth=(username, password))
            try:
                if database:
                    driver.verify_connectivity(database=database)
                else:
                    driver.verify_connectivity()
            except TypeError:
                driver.verify_connectivity()
            return driver
        except Exception as ex:
            if not silent:
                print("Neo4j.Connect - Error: Could not connect to the Neo4j server. Returning None.")
                print(ex)
            return None

    @staticmethod
    def Close(driver, silent=False):
        """
        Closes the input Neo4j driver.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        bool
            True if the driver was successfully closed. False otherwise.
        """
        if not Neo4j._is_driver(driver):
            if not silent:
                print("Neo4j.Close - Error: The input driver is not a valid Neo4j driver. Returning False.")
            return False
        try:
            driver.close()
            return True
        except Exception as ex:
            if not silent:
                print("Neo4j.Close - Error: Could not close the driver. Returning False.")
                print(ex)
            return False

    @staticmethod
    def Execute(driver, cypher, parameters=None, write=False, database=None, silent=False):
        """
        Executes the input Cypher statement and returns the resulting records.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        cypher : str
            The input Cypher statement.
        parameters : dict , optional
            The input query parameters. Default is None.
        write : bool , optional
            If set to True, the statement is executed as a write transaction.
            Otherwise, it is executed as a read transaction. Default is False.
        database : str , optional
            The database name. If set to None, the driver's default database is
            used. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        list
            The resulting list of Neo4j records.
        """
        if not Neo4j._is_driver(driver):
            if not silent:
                print("Neo4j.Execute - Error: The input driver is not a valid Neo4j driver. Returning None.")
            return None
        if not isinstance(cypher, str) or len(cypher.strip()) < 1:
            if not silent:
                print("Neo4j.Execute - Error: The input cypher is not a valid string. Returning None.")
            return None

        parameters = parameters or {}

        def _run(tx):
            result = tx.run(cypher, parameters)
            return list(result)

        try:
            kwargs = {}
            if database:
                kwargs["database"] = database
            with driver.session(**kwargs) as session:
                if write:
                    try:
                        return session.execute_write(_run)
                    except AttributeError:
                        return session.write_transaction(_run)
                try:
                    return session.execute_read(_run)
                except AttributeError:
                    return session.read_transaction(_run)
        except Exception as ex:
            if not silent:
                print("Neo4j.Execute - Error: Could not execute the Cypher statement. Returning None.")
                print(ex)
            return None

    @staticmethod
    def Query(driver, cypher, parameters=None, database=None, silent=False):
        """
        Executes the input Cypher query and returns the resulting records.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        cypher : str
            The input Cypher query.
        parameters : dict , optional
            The input query parameters. Default is None.
        database : str , optional
            The database name. If set to None, the driver's default database is
            used. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        list
            The resulting list of Neo4j records.
        """
        return Neo4j.Execute(driver=driver, cypher=cypher, parameters=parameters, write=False, database=database, silent=silent)

    @staticmethod
    def BatchExecute(driver, cypher, data, batchSize=1000, database=None, silent=False):
        """
        Executes the input Cypher statement repeatedly in batches.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        cypher : str
            The input Cypher statement. This statement should expect a
            parameter named ``rows``.
        data : list
            The input list of dictionaries.
        batchSize : int , optional
            The desired batch size. Default is 1000.
        database : str , optional
            The database name. If set to None, the driver's default database is
            used. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        bool
            True if the operation completed successfully. False otherwise.
        """
        if not isinstance(data, list):
            if not silent:
                print("Neo4j.BatchExecute - Error: The input data is not a valid list. Returning False.")
            return False
        if len(data) < 1:
            return True
        try:
            batchSize = max(1, int(batchSize))
        except Exception:
            batchSize = 1000

        for i in range(0, len(data), batchSize):
            batch = data[i:i+batchSize]
            result = Neo4j.Execute(driver=driver,
                                   cypher=cypher,
                                   parameters={"rows": batch},
                                   write=True,
                                   database=database,
                                   silent=silent)
            if result is None:
                return False
        return True

    @staticmethod
    def Reset(driver, database=None, silent=False):
        """
        Resets the input Neo4j database completely.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        database : str , optional
            The database name. If set to None, the driver's default database is
            used. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        bool
            True if successful. False otherwise.
        """
        if not Neo4j._is_driver(driver):
            if not silent:
                print("Neo4j.Reset - Error: The input driver is not a valid Neo4j driver. Returning False.")
            return False

        ok = Neo4j.Execute(driver, "MATCH (n) DETACH DELETE n", write=True, database=database, silent=silent)
        if ok is None:
            return False

        indexes = Neo4j.Query(driver, "SHOW INDEXES", database=database, silent=True) or []
        for index in indexes:
            try:
                name = index["name"]
                Neo4j.Execute(driver, f"DROP INDEX {Neo4j._sanitize_identifier(name)}", write=True, database=database, silent=True)
            except Exception:
                pass

        constraints = Neo4j.Query(driver, "SHOW CONSTRAINTS", database=database, silent=True) or []
        for constraint in constraints:
            try:
                name = constraint["name"]
                Neo4j.Execute(driver, f"DROP CONSTRAINT {Neo4j._sanitize_identifier(name)}", write=True, database=database, silent=True)
            except Exception:
                pass
        return True

    @staticmethod
    def CreateIndex(driver, label, property, indexName=None, ifNotExists=True, database=None, silent=False):
        """
        Creates an index on the input node label and property.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        label : str
            The node label.
        property : str
            The property name.
        indexName : str , optional
            The index name. If set to None, a name is generated automatically.
            Default is None.
        ifNotExists : bool , optional
            If set to True, the index is only created if it does not already
            exist. Default is True.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        bool
            True if successful. False otherwise.
        """
        label = Neo4j._sanitize_identifier(label, default="Node")
        property = Neo4j._sanitize_identifier(property, default="id")
        if indexName is None:
            indexName = f"idx_{label}_{property}"
        indexName = Neo4j._sanitize_identifier(indexName)
        ifClause = " IF NOT EXISTS" if ifNotExists else ""
        cypher = f"CREATE INDEX {indexName}{ifClause} FOR (n:{label}) ON (n.{property})"
        result = Neo4j.Execute(driver, cypher, write=True, database=database, silent=silent)
        return result is not None

    @staticmethod
    def CreateConstraint(driver, label, property, unique=True, constraintName=None, ifNotExists=True, database=None, silent=False):
        """
        Creates a node property constraint.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        label : str
            The node label.
        property : str
            The property name.
        unique : bool , optional
            If set to True, a uniqueness constraint is created. Otherwise,
            a property existence constraint is created. Default is True.
        constraintName : str , optional
            The constraint name. If set to None, a name is generated
            automatically. Default is None.
        ifNotExists : bool , optional
            If set to True, the constraint is only created if it does not
            already exist. Default is True.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        bool
            True if successful. False otherwise.
        """
        label = Neo4j._sanitize_identifier(label, default="Node")
        property = Neo4j._sanitize_identifier(property, default="id")
        if constraintName is None:
            suffix = "unique" if unique else "exists"
            constraintName = f"con_{label}_{property}_{suffix}"
        constraintName = Neo4j._sanitize_identifier(constraintName)
        ifClause = " IF NOT EXISTS" if ifNotExists else ""
        if unique:
            cypher = f"CREATE CONSTRAINT {constraintName}{ifClause} FOR (n:{label}) REQUIRE n.{property} IS UNIQUE"
        else:
            cypher = f"CREATE CONSTRAINT {constraintName}{ifClause} FOR (n:{label}) REQUIRE n.{property} IS NOT NULL"
        result = Neo4j.Execute(driver, cypher, write=True, database=database, silent=silent)
        return result is not None

    @staticmethod
    def Schema(driver, database=None, silent=False):
        """
        Returns the schema information of the input Neo4j database.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        dict
            A dictionary containing the database indexes and constraints.
        """
        indexes = Neo4j.Query(driver, "SHOW INDEXES", database=database, silent=silent)
        constraints = Neo4j.Query(driver, "SHOW CONSTRAINTS", database=database, silent=silent)
        if indexes is None or constraints is None:
            return None
        return {"indexes": indexes, "constraints": constraints}

    @staticmethod
    def CountNodes(driver, label=None, database=None, silent=False):
        """
        Returns the number of nodes in the input Neo4j database.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        label : str , optional
            The node label to filter by. Default is None.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        int
            The number of nodes.
        """
        if label:
            label = Neo4j._sanitize_identifier(label, default="Node")
            cypher = f"MATCH (n:{label}) RETURN count(n) AS count"
        else:
            cypher = "MATCH (n) RETURN count(n) AS count"
        result = Neo4j.Query(driver, cypher, database=database, silent=silent)
        if result is None or len(result) < 1:
            return None
        return result[0]["count"]

    @staticmethod
    def CountRelationships(driver, relationshipType=None, database=None, silent=False):
        """
        Returns the number of relationships in the input Neo4j database.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        relationshipType : str , optional
            The relationship type to filter by. Default is None.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        int
            The number of relationships.
        """
        if relationshipType:
            relationshipType = Neo4j._sanitize_identifier(relationshipType, default="CONNECTED_TO")
            cypher = f"MATCH ()-[r:{relationshipType}]->() RETURN count(r) AS count"
        else:
            cypher = "MATCH ()-[r]->() RETURN count(r) AS count"
        result = Neo4j.Query(driver, cypher, database=database, silent=silent)
        if result is None or len(result) < 1:
            return None
        return result[0]["count"]

    @staticmethod
    def Labels(driver, database=None, silent=False):
        """
        Returns the list of labels in the input Neo4j database.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        list
            The list of labels.
        """
        result = Neo4j.Query(driver, "CALL db.labels()", database=database, silent=silent)
        if result is None:
            return None
        labels = []
        for record in result:
            try:
                labels.append(record["label"])
            except Exception:
                try:
                    labels.append(record[0])
                except Exception:
                    pass
        return labels

    @staticmethod
    def RelationshipTypes(driver, database=None, silent=False):
        """
        Returns the list of relationship types in the input Neo4j database.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        list
            The list of relationship types.
        """
        result = Neo4j.Query(driver, "CALL db.relationshipTypes()", database=database, silent=silent)
        if result is None:
            return None
        types = []
        for record in result:
            try:
                types.append(record["relationshipType"])
            except Exception:
                try:
                    types.append(record[0])
                except Exception:
                    pass
        return types

    @staticmethod
    def Info(driver, database=None, silent=False):
        """
        Returns summary information about the input Neo4j database.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        dict
            A dictionary containing summary information.
        """
        return {
            "nodeCount": Neo4j.CountNodes(driver, database=database, silent=silent),
            "relationshipCount": Neo4j.CountRelationships(driver, database=database, silent=silent),
            "labels": Neo4j.Labels(driver, database=database, silent=silent),
            "relationshipTypes": Neo4j.RelationshipTypes(driver, database=database, silent=silent)
        }

    @staticmethod
    def MatchNodes(driver, label=None, properties=None, database=None, silent=False):
        """
        Returns the list of nodes matching the input label and properties.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        label : str , optional
            The node label. Default is None.
        properties : dict , optional
            The input property dictionary. Default is None.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        list
            The list of matching nodes as Python dictionaries.
        """
        properties = properties or {}
        clauses = []
        params = {}
        if label:
            label_clause = ":"+Neo4j._sanitize_identifier(label, default="Node")
        else:
            label_clause = ""
        for i, (k, v) in enumerate(properties.items()):
            pk = Neo4j._sanitize_identifier(k, default=f"p{i}")
            clauses.append(f"n.{pk} = $p{i}")
            params[f"p{i}"] = v
        where = ""
        if len(clauses) > 0:
            where = " WHERE " + " AND ".join(clauses)
        cypher = f"MATCH (n{label_clause}){where} RETURN n"
        result = Neo4j.Query(driver, cypher, parameters=params, database=database, silent=silent)
        if result is None:
            return None
        output = []
        for record in result:
            try:
                output.append(Neo4j._node_properties(record["n"]))
            except Exception:
                pass
        return output

    @staticmethod
    def DeleteNodes(driver, label=None, properties=None, database=None, silent=False):
        """
        Deletes the nodes matching the input label and properties.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        label : str , optional
            The node label. Default is None.
        properties : dict , optional
            The input property dictionary. Default is None.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        bool
            True if successful. False otherwise.
        """
        properties = properties or {}
        clauses = []
        params = {}
        if label:
            label_clause = ":"+Neo4j._sanitize_identifier(label, default="Node")
        else:
            label_clause = ""
        for i, (k, v) in enumerate(properties.items()):
            pk = Neo4j._sanitize_identifier(k, default=f"p{i}")
            clauses.append(f"n.{pk} = $p{i}")
            params[f"p{i}"] = v
        where = ""
        if len(clauses) > 0:
            where = " WHERE " + " AND ".join(clauses)
        cypher = f"MATCH (n{label_clause}){where} DETACH DELETE n"
        result = Neo4j.Execute(driver, cypher, parameters=params, write=True, database=database, silent=silent)
        return result is not None

    @staticmethod
    def DeleteRelationships(driver, relationshipType=None, database=None, silent=False):
        """
        Deletes the relationships matching the input type.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        relationshipType : str , optional
            The relationship type. Default is None.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        bool
            True if successful. False otherwise.
        """
        if relationshipType:
            relationshipType = Neo4j._sanitize_identifier(relationshipType, default="CONNECTED_TO")
            cypher = f"MATCH ()-[r:{relationshipType}]->() DELETE r"
        else:
            cypher = "MATCH ()-[r]->() DELETE r"
        result = Neo4j.Execute(driver, cypher, write=True, database=database, silent=silent)
        return result is not None

    @staticmethod
    def Validate(driver, idKey="topologic_id", database=None, silent=False):
        """
        Returns a validation report for the input Neo4j database.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        idKey : str , optional
            The node property used as a stable identifier. Default is
            "topologic_id".
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        dict
            A validation report.
        """
        idKey = Neo4j._sanitize_identifier(idKey, default="topologic_id")
        duplicate_result = Neo4j.Query(
            driver,
            f"MATCH (n) WHERE n.{idKey} IS NOT NULL WITH n.{idKey} AS id, count(n) AS c WHERE c > 1 RETURN count(id) AS count",
            database=database,
            silent=silent
        )
        missing_result = Neo4j.Query(
            driver,
            f"MATCH (n) WHERE n.{idKey} IS NULL RETURN count(n) AS count",
            database=database,
            silent=silent
        )
        orphan_result = Neo4j.Query(
            driver,
            "MATCH (n) WHERE NOT (n)--() RETURN count(n) AS count",
            database=database,
            silent=silent
        )
        if duplicate_result is None or missing_result is None or orphan_result is None:
            return None
        return {
            "duplicateNodeIds": duplicate_result[0]["count"],
            "missingNodeIds": missing_result[0]["count"],
            "orphanNodes": orphan_result[0]["count"]
        }

    @staticmethod
    def ToDataFrame(driver, cypher, parameters=None, database=None, silent=False):
        """
        Returns the result of the input Cypher query as a pandas DataFrame.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        cypher : str
            The input Cypher query.
        parameters : dict , optional
            The input query parameters. Default is None.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        pandas.DataFrame
            The resulting DataFrame.
        """
        try:
            import pandas as pd
        except Exception:
            if not silent:
                print("Neo4j.ToDataFrame - Error: Could not import pandas. Returning None.")
            return None

        records = Neo4j.Query(driver, cypher, parameters=parameters, database=database, silent=silent)
        if records is None:
            return None
        rows = []
        for record in records:
            row = {}
            for key in record.keys():
                value = record[key]
                if hasattr(value, "items") and hasattr(value, "element_id"):
                    try:
                        value = dict(value.items())
                    except Exception:
                        pass
                row[key] = value
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _prepare_graph_rows(graph,
                            nodeLabelKey="label",
                            defaultNodeLabel="Node",
                            nodeCategoryKey="category",
                            defaultNodeCategory=None,
                            relationshipTypeKey="label",
                            defaultRelationshipType="CONNECTED_TO",
                            relationshipCategoryKey="category",
                            defaultRelationshipCategory=None,
                            mantissa=6,
                            tolerance=0.0001,
                            silent=False):
        """
        Returns dictionaries representing the input Topologic graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input Topologic graph.
        nodeLabelKey : str , optional
            The dictionary key used to find the node label. Default is "label".
        defaultNodeLabel : str , optional
            The default node label. Default is "Node".
        nodeCategoryKey : str , optional
            The dictionary key used to find the node category. Default is
            "category".
        defaultNodeCategory : str , optional
            The default node category. Default is None.
        relationshipTypeKey : str , optional
            The dictionary key used to find the relationship type. Default is
            "label".
        defaultRelationshipType : str , optional
            The default relationship type. Default is "CONNECTED_TO".
        relationshipCategoryKey : str , optional
            The dictionary key used to find the relationship category.
            Default is "category".
        defaultRelationshipCategory : str , optional
            The default relationship category. Default is None.
        mantissa : int , optional
            The desired mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        tuple
            A tuple containing the node rows, relationship rows, and vertices.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Neo4j._prepare_graph_rows - Error: The input graph is not a valid Topologic graph. Returning None.")
            return None

        vertices = Graph.Vertices(graph)
        edges = Graph.Edges(graph)

        node_rows = []
        width = max(3, len(str(max(1, len(vertices)))))
        for i, vertex in enumerate(vertices):
            d = Topology.Dictionary(vertex)
            pd = Dictionary.PythonDictionary(d) if d else {}
            label = Neo4j._sanitize_identifier(pd.get(nodeLabelKey, f"{defaultNodeLabel}_{str(i+1).zfill(width)}"), default=defaultNodeLabel)
            category = pd.get(nodeCategoryKey, defaultNodeCategory)
            properties = dict(pd)
            properties["x"] = Vertex.X(vertex, mantissa=mantissa)
            properties["y"] = Vertex.Y(vertex, mantissa=mantissa)
            properties["z"] = Vertex.Z(vertex, mantissa=mantissa)
            properties["topologic_id"] = i
            if category is not None:
                properties[nodeCategoryKey] = category
            properties[nodeLabelKey] = label
            node_rows.append({"topologic_id": i, "label": label, "properties": properties})

        relationship_rows = []
        for edge in edges:
            d = Topology.Dictionary(edge)
            pd = Dictionary.PythonDictionary(d) if d else {}
            relType = Neo4j._sanitize_identifier(pd.get(relationshipTypeKey, defaultRelationshipType), default=defaultRelationshipType)
            category = pd.get(relationshipCategoryKey, defaultRelationshipCategory)
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            sid = Vertex.Index(vertex=sv, vertices=vertices, strict=False, tolerance=tolerance)
            eid = Vertex.Index(vertex=ev, vertices=vertices, strict=False, tolerance=tolerance)
            properties = dict(pd)
            if category is not None:
                properties[relationshipCategoryKey] = category
            properties[relationshipTypeKey] = relType
            relationship_rows.append({"start_id": sid, "end_id": eid, "type": relType, "properties": properties})

        return node_rows, relationship_rows, vertices

    @staticmethod
    def ByGraph(driver,
                graph,
                nodeLabelKey="label",
                defaultNodeLabel="Node",
                nodeCategoryKey="category",
                defaultNodeCategory=None,
                relationshipTypeKey="label",
                defaultRelationshipType="CONNECTED_TO",
                relationshipCategoryKey="category",
                defaultRelationshipCategory=None,
                bidirectional=True,
                deleteAll=False,
                createIndex=True,
                createConstraint=False,
                mantissa=6,
                tolerance=0.0001,
                database=None,
                silent=False):
        """
        Writes the input Topologic graph to the input Neo4j database.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        graph : topologic_core.Graph
            The input Topologic graph.
        nodeLabelKey : str , optional
            The dictionary key used to find the node label. Default is "label".
        defaultNodeLabel : str , optional
            The default node label. Default is "Node".
        nodeCategoryKey : str , optional
            The dictionary key used to find the node category. Default is
            "category".
        defaultNodeCategory : str , optional
            The default node category. Default is None.
        relationshipTypeKey : str , optional
            The dictionary key used to find the relationship type. Default is
            "label".
        defaultRelationshipType : str , optional
            The default relationship type. Default is "CONNECTED_TO".
        relationshipCategoryKey : str , optional
            The dictionary key used to find the relationship category.
            Default is "category".
        defaultRelationshipCategory : str , optional
            The default relationship category. Default is None.
        bidirectional : bool , optional
            If set to True, reverse relationships are also created. Default is
            True.
        deleteAll : bool , optional
            If set to True, all existing nodes and relationships are deleted
            before the graph is written. Default is False.
        createIndex : bool , optional
            If set to True, an index is created on the ``topologic_id``
            property. Default is True.
        createConstraint : bool , optional
            If set to True, a uniqueness constraint is created on the
            ``topologic_id`` property for each encountered label. Default is
            False.
        mantissa : int , optional
            The desired mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        neo4j.Driver
            The input Neo4j driver.
        """
        rows = Neo4j._prepare_graph_rows(graph=graph,
                                         nodeLabelKey=nodeLabelKey,
                                         defaultNodeLabel=defaultNodeLabel,
                                         nodeCategoryKey=nodeCategoryKey,
                                         defaultNodeCategory=defaultNodeCategory,
                                         relationshipTypeKey=relationshipTypeKey,
                                         defaultRelationshipType=defaultRelationshipType,
                                         relationshipCategoryKey=relationshipCategoryKey,
                                         defaultRelationshipCategory=defaultRelationshipCategory,
                                         mantissa=mantissa,
                                         tolerance=tolerance,
                                         silent=silent)
        if rows is None:
            return None
        node_rows, relationship_rows, vertices = rows

        if deleteAll:
            if not Neo4j.Reset(driver, database=database, silent=silent):
                return None

        labels = sorted(list(set([row["label"] for row in node_rows])))

        node_cypher = """
        UNWIND $rows AS row
        CALL apoc.create.node([row.label], row.properties) YIELD node
        RETURN count(node) AS count
        """
        use_apoc = True
        test_apoc = Neo4j.Query(driver, "RETURN apoc.version() AS version", database=database, silent=True)
        if test_apoc is None:
            use_apoc = False

        if use_apoc:
            ok = Neo4j.BatchExecute(driver, node_cypher, node_rows, batchSize=1000, database=database, silent=silent)
            if not ok:
                return None
        else:
            for row in node_rows:
                label = row["label"]
                cypher = f"CREATE (n:{label} $properties)"
                result = Neo4j.Execute(driver,
                                       cypher,
                                       parameters={"properties": row["properties"]},
                                       write=True,
                                       database=database,
                                       silent=silent)
                if result is None:
                    return None

        for label in labels:
            if createIndex:
                Neo4j.CreateIndex(driver, label=label, property="topologic_id", database=database, silent=True)
            if createConstraint:
                Neo4j.CreateConstraint(driver, label=label, property="topologic_id", unique=True, database=database, silent=True)

        for row in relationship_rows:
            relType = row["type"]
            cypher = f"""
            MATCH (a {{topologic_id: $start_id}}), (b {{topologic_id: $end_id}})
            CREATE (a)-[r:{relType} $properties]->(b)
            """
            result = Neo4j.Execute(driver,
                                   cypher,
                                   parameters={"start_id": row["start_id"], "end_id": row["end_id"], "properties": row["properties"]},
                                   write=True,
                                   database=database,
                                   silent=silent)
            if result is None:
                return None

            if bidirectional:
                result = Neo4j.Execute(driver,
                                       cypher,
                                       parameters={"start_id": row["end_id"], "end_id": row["start_id"], "properties": row["properties"]},
                                       write=True,
                                       database=database,
                                       silent=silent)
                if result is None:
                    return None
        return driver

    @staticmethod
    def MergeGraph(driver,
                   graph,
                   nodeLabelKey="label",
                   defaultNodeLabel="Node",
                   nodeCategoryKey="category",
                   defaultNodeCategory=None,
                   relationshipTypeKey="label",
                   defaultRelationshipType="CONNECTED_TO",
                   relationshipCategoryKey="category",
                   defaultRelationshipCategory=None,
                   bidirectional=True,
                   createIndex=True,
                   createConstraint=False,
                   mantissa=6,
                   tolerance=0.0001,
                   database=None,
                   silent=False):
        """
        Merges the input Topologic graph into the input Neo4j database.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        graph : topologic_core.Graph
            The input Topologic graph.
        nodeLabelKey : str , optional
            The dictionary key used to find the node label. Default is "label".
        defaultNodeLabel : str , optional
            The default node label. Default is "Node".
        nodeCategoryKey : str , optional
            The dictionary key used to find the node category. Default is
            "category".
        defaultNodeCategory : str , optional
            The default node category. Default is None.
        relationshipTypeKey : str , optional
            The dictionary key used to find the relationship type. Default is
            "label".
        defaultRelationshipType : str , optional
            The default relationship type. Default is "CONNECTED_TO".
        relationshipCategoryKey : str , optional
            The dictionary key used to find the relationship category.
            Default is "category".
        defaultRelationshipCategory : str , optional
            The default relationship category. Default is None.
        bidirectional : bool , optional
            If set to True, reverse relationships are also merged. Default is
            True.
        createIndex : bool , optional
            If set to True, an index is created on the ``topologic_id``
            property. Default is True.
        createConstraint : bool , optional
            If set to True, a uniqueness constraint is created on the
            ``topologic_id`` property for each encountered label. Default is
            False.
        mantissa : int , optional
            The desired mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        neo4j.Driver
            The input Neo4j driver.
        """
        rows = Neo4j._prepare_graph_rows(graph=graph,
                                         nodeLabelKey=nodeLabelKey,
                                         defaultNodeLabel=defaultNodeLabel,
                                         nodeCategoryKey=nodeCategoryKey,
                                         defaultNodeCategory=defaultNodeCategory,
                                         relationshipTypeKey=relationshipTypeKey,
                                         defaultRelationshipType=defaultRelationshipType,
                                         relationshipCategoryKey=relationshipCategoryKey,
                                         defaultRelationshipCategory=defaultRelationshipCategory,
                                         mantissa=mantissa,
                                         tolerance=tolerance,
                                         silent=silent)
        if rows is None:
            return None
        node_rows, relationship_rows, vertices = rows

        labels = sorted(list(set([row["label"] for row in node_rows])))
        for label in labels:
            if createIndex:
                Neo4j.CreateIndex(driver, label=label, property="topologic_id", database=database, silent=True)
            if createConstraint:
                Neo4j.CreateConstraint(driver, label=label, property="topologic_id", unique=True, database=database, silent=True)

        for row in node_rows:
            label = row["label"]
            cypher = f"""
            MERGE (n:{label} {{topologic_id: $topologic_id}})
            SET n += $properties
            """
            result = Neo4j.Execute(driver,
                                   cypher,
                                   parameters={"topologic_id": row["topologic_id"], "properties": row["properties"]},
                                   write=True,
                                   database=database,
                                   silent=silent)
            if result is None:
                return None

        for row in relationship_rows:
            relType = row["type"]
            cypher = f"""
            MATCH (a {{topologic_id: $start_id}}), (b {{topologic_id: $end_id}})
            MERGE (a)-[r:{relType}]->(b)
            SET r += $properties
            """
            result = Neo4j.Execute(driver,
                                   cypher,
                                   parameters={"start_id": row["start_id"], "end_id": row["end_id"], "properties": row["properties"]},
                                   write=True,
                                   database=database,
                                   silent=silent)
            if result is None:
                return None
            if bidirectional:
                result = Neo4j.Execute(driver,
                                       cypher,
                                       parameters={"start_id": row["end_id"], "end_id": row["start_id"], "properties": row["properties"]},
                                       write=True,
                                       database=database,
                                       silent=silent)
                if result is None:
                    return None
        return driver

    @staticmethod
    def _collect_graph_value(value, nodes_by_id, rels_by_id):
        """
        Collects Neo4j graph entities from the input value.

        Parameters
        ----------
        value : any
            The input value.
        nodes_by_id : dict
            The dictionary of nodes keyed by element id.
        rels_by_id : dict
            The dictionary of relationships keyed by element id.

        Returns
        -------
        None
            None.
        """
        try:
            from neo4j.graph import Node, Relationship, Path
        except Exception:
            Node = None
            Relationship = None
            Path = None

        if value is None:
            return

        if Node is not None and isinstance(value, Node):
            nodes_by_id[value.element_id] = value
            return

        if Relationship is not None and isinstance(value, Relationship):
            rels_by_id[value.element_id] = value
            try:
                nodes_by_id[value.start_node.element_id] = value.start_node
                nodes_by_id[value.end_node.element_id] = value.end_node
            except Exception:
                pass
            return

        if Path is not None and isinstance(value, Path):
            try:
                for node in value.nodes:
                    nodes_by_id[node.element_id] = node
            except Exception:
                pass
            try:
                for rel in value.relationships:
                    rels_by_id[rel.element_id] = rel
                    try:
                        nodes_by_id[rel.start_node.element_id] = rel.start_node
                        nodes_by_id[rel.end_node.element_id] = rel.end_node
                    except Exception:
                        pass
            except Exception:
                pass
            return

        if isinstance(value, dict):
            for v in value.values():
                Neo4j._collect_graph_value(v, nodes_by_id, rels_by_id)
            return

        if isinstance(value, (list, tuple, set)):
            for v in value:
                Neo4j._collect_graph_value(v, nodes_by_id, rels_by_id)
            return

    @staticmethod
    def ToGraph(driver,
                cypher: str = None,
                parameters: dict = None,
                xMin: float = -0.5,
                yMin: float = -0.5,
                zMin: float = -0.5,
                xMax: float = 0.5,
                yMax: float = 0.5,
                zMax: float = 0.5,
                tolerance: float = 0.0001,
                silent: bool = False):
        """
        Returns a Topologic graph from the input Neo4j database or query result.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        cypher : str , optional
            The input Cypher query. The query may return nodes, relationships,
            paths, lists, and dictionaries that contain graph entities. If set
            to None, the entire database graph is imported. Default is None.
        parameters : dict , optional
            The dictionary of Cypher parameters. Default is None.
        xMin : float , optional
            The minimum random X coordinate to use when a node does not contain
            an ``x`` property. Default is -0.5.
        yMin : float , optional
            The minimum random Y coordinate to use when a node does not contain
            a ``y`` property. Default is -0.5.
        zMin : float , optional
            The minimum random Z coordinate to use when a node does not contain
            a ``z`` property. Default is -0.5.
        xMax : float , optional
            The maximum random X coordinate to use when a node does not contain
            an ``x`` property. Default is 0.5.
        yMax : float , optional
            The maximum random Y coordinate to use when a node does not contain
            a ``y`` property. Default is 0.5.
        zMax : float , optional
            The maximum random Z coordinate to use when a node does not contain
            a ``z`` property. Default is 0.5.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            The created Topologic graph.
        """
        import random
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if driver is None or not hasattr(driver, "session"):
            if not silent:
                print("Neo4j.ToGraph - Error: The input driver is not a valid Neo4j driver. Returning None.")
            return None

        if cypher is None:
            cypher = "MATCH p=(a)-[r]->(b) RETURN p AS result UNION MATCH (n) RETURN n AS result"

        parameters = parameters or {}

        try:
            records = Neo4j.Query(driver, cypher, parameters=parameters)
        except Exception as ex:
            if not silent:
                print("Neo4j.ToGraph - Error: Could not execute the Cypher query. Returning None.")
                print(ex)
            return None

        nodes_by_id = {}
        rels_by_id = {}

        def collect_value(value):
            if value is None:
                return

            try:
                from neo4j.graph import Node, Relationship, Path
            except Exception:
                Node = None
                Relationship = None
                Path = None

            if Node is not None and isinstance(value, Node):
                nodes_by_id[value.element_id] = value
                return

            if Relationship is not None and isinstance(value, Relationship):
                rels_by_id[value.element_id] = value
                try:
                    nodes_by_id[value.start_node.element_id] = value.start_node
                    nodes_by_id[value.end_node.element_id] = value.end_node
                except Exception:
                    pass
                return

            if Path is not None and isinstance(value, Path):
                try:
                    for node in value.nodes:
                        nodes_by_id[node.element_id] = node
                except Exception:
                    pass
                try:
                    for rel in value.relationships:
                        rels_by_id[rel.element_id] = rel
                        try:
                            nodes_by_id[rel.start_node.element_id] = rel.start_node
                            nodes_by_id[rel.end_node.element_id] = rel.end_node
                        except Exception:
                            pass
                except Exception:
                    pass
                return

            if isinstance(value, (list, tuple, set)):
                for item in value:
                    collect_value(item)
                return

            if isinstance(value, dict):
                for v in value.values():
                    collect_value(v)
                return

        for record in records:
            for value in record.values():
                collect_value(value)

        topologic_vertices = {}
        vertices = []
        edges = []

        for _, node in nodes_by_id.items():
            properties = dict(node.items())
            x = properties.get("x", random.uniform(xMin, xMax))
            y = properties.get("y", random.uniform(yMin, yMax))
            z = properties.get("z", random.uniform(zMin, zMax))

            vertex = Vertex.ByCoordinates(x, y, z)
            properties["id"] = node.element_id
            properties["labels"] = list(node.labels)

            d = Dictionary.ByPythonDictionary(properties)
            vertex = Topology.SetDictionary(vertex, d)

            topologic_vertices[node.element_id] = vertex
            vertices.append(vertex)

        for _, rel in rels_by_id.items():
            sv = topologic_vertices.get(rel.start_node.element_id)
            ev = topologic_vertices.get(rel.end_node.element_id)

            if sv is None or ev is None:
                continue

            edge = Edge.ByVertices(sv, ev)
            properties = dict(rel.items())
            properties["id"] = rel.element_id
            properties["type"] = rel.type

            d = Dictionary.ByPythonDictionary(properties)
            edge = Topology.SetDictionary(edge, d)
            edges.append(edge)

        return Graph.ByVerticesEdges(vertices, edges)

    @staticmethod
    def Subgraph(driver,
                 cypher,
                 xMin=-0.5,
                 yMin=-0.5,
                 zMin=-0.5,
                 xMax=0.5,
                 yMax=0.5,
                 zMax=0.5,
                 tolerance=0.0001,
                 database=None,
                 silent=False):
        """
        Returns a Topologic graph from the input Cypher subgraph query.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        cypher : str
            The input Cypher query.
        xMin : float , optional
            The minimum random X coordinate to use if a node does not have one.
            Default is -0.5.
        yMin : float , optional
            The minimum random Y coordinate to use if a node does not have one.
            Default is -0.5.
        zMin : float , optional
            The minimum random Z coordinate to use if a node does not have one.
            Default is -0.5.
        xMax : float , optional
            The maximum random X coordinate to use if a node does not have one.
            Default is 0.5.
        yMax : float , optional
            The maximum random Y coordinate to use if a node does not have one.
            Default is 0.5.
        zMax : float , optional
            The maximum random Z coordinate to use if a node does not have one.
            Default is 0.5.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Graph
            The created Topologic graph.
        """
        return Neo4j.ToGraph(driver=driver,
                             cypher=cypher,
                             xMin=xMin,
                             yMin=yMin,
                             zMin=zMin,
                             xMax=xMax,
                             yMax=yMax,
                             zMax=zMax,
                             tolerance=tolerance,
                             database=database,
                             silent=silent)

    @staticmethod
    def Neighborhood(driver,
                     nodeId,
                     depth: int = 1,
                     xMin: float = -0.5,
                     yMin: float = -0.5,
                     zMin: float = -0.5,
                     xMax: float = 0.5,
                     yMax: float = 0.5,
                     zMax: float = 0.5,
                     tolerance: float = 0.0001,
                     silent: bool = False):
        """
        Returns the neighborhood of the input Neo4j node as a Topologic graph.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        nodeId : str
            The Neo4j internal element id of the source node.
        depth : int , optional
            The neighborhood depth. Default is 1.
        xMin : float , optional
            The minimum random X coordinate to use when a node does not contain
            an ``x`` property. Default is -0.5.
        yMin : float , optional
            The minimum random Y coordinate to use when a node does not contain
            a ``y`` property. Default is -0.5.
        zMin : float , optional
            The minimum random Z coordinate to use when a node does not contain
            a ``z`` property. Default is -0.5.
        xMax : float , optional
            The maximum random X coordinate to use when a node does not contain
            an ``x`` property. Default is 0.5.
        yMax : float , optional
            The maximum random Y coordinate to use when a node does not contain
            a ``y`` property. Default is 0.5.
        zMax : float , optional
            The maximum random Z coordinate to use when a node does not contain
            a ``z`` property. Default is 0.5.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            The returned Topologic graph.
        """
        if driver is None or not hasattr(driver, "session"):
            if not silent:
                print("Neo4j.Neighborhood - Error: The input driver is not a valid Neo4j driver. Returning None.")
            return None

        if not isinstance(nodeId, str) or len(nodeId) < 1:
            if not silent:
                print("Neo4j.Neighborhood - Error: The input nodeId is not a valid string. Returning None.")
            return None

        if not isinstance(depth, int) or depth < 1:
            if not silent:
                print("Neo4j.Neighborhood - Error: The input depth is not a valid positive integer. Returning None.")
            return None

        cypher = f"""
        MATCH (n)
        WHERE elementId(n) = $nodeId
        OPTIONAL MATCH p=(n)-[*1..{depth}]-(m)
        RETURN n AS result
        UNION
        MATCH (n)
        WHERE elementId(n) = $nodeId
        OPTIONAL MATCH p=(n)-[*1..{depth}]-(m)
        WITH p
        WHERE p IS NOT NULL
        RETURN p AS result
        """

        return Neo4j.ToGraph(driver,
                             cypher=cypher,
                             parameters={"nodeId": nodeId},
                             xMin=xMin,
                             yMin=yMin,
                             zMin=zMin,
                             xMax=xMax,
                             yMax=yMax,
                             zMax=zMax,
                             tolerance=tolerance,
                             silent=silent)
