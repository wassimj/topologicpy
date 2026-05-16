# -*- coding: utf-8 -*-

"""
Minimal GQL-like parser for TopologicPy.

Phase 1 read-query subset:
- MATCH one node-edge-node pattern
- directed, reverse-directed, and undirected edge patterns
- optional WHERE with AND / OR / parentheses
- RETURN / RETURN *
- RETURN DISTINCT
- COUNT(...)
- RETURN aliases with AS
- ORDER BY
- SKIP
- LIMIT

Phase 2 graph mutation subset:
- CREATE node or node-edge-node patterns
- MERGE node or node-edge-node patterns
- MATCH ... SET property assignments
- MATCH ... DELETE bound variables
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


try:
    from lark import Lark, Transformer, UnexpectedInput, v_args
except ImportError as e:
    raise ImportError(
        "topologicpy.gql.Parser requires lark. Install it with: pip install lark"
    ) from e


@dataclass(frozen=True)
class NodePattern:
    variable: Optional[str]
    label: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class EdgePattern:
    variable: Optional[str] = None
    label: Optional[str] = None
    direction: str = "out"
    properties: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class MatchPattern:
    left_node: NodePattern
    edge: Optional[EdgePattern] = None
    right_node: Optional[NodePattern] = None


@dataclass(frozen=True)
class WherePredicate:
    variable: str
    property: str
    operator: str
    value: Any


@dataclass(frozen=True)
class BooleanExpression:
    operator: str
    left: Any
    right: Any


@dataclass(frozen=True)
class WhereClause:
    expression: Any


@dataclass(frozen=True)
class AggregateExpression:
    function: str
    argument: Any
    distinct: bool = False


@dataclass(frozen=True)
class ReturnItem:
    expression: Any
    alias: Optional[str] = None


@dataclass(frozen=True)
class ReturnClause:
    items: List[Any]
    distinct: bool = False


@dataclass(frozen=True)
class SetItem:
    variable: str
    property: str
    value: Any


@dataclass(frozen=True)
class QueryAST:
    query_type: str
    match: Optional[MatchPattern] = None
    where: Optional[WhereClause] = None
    returns: Optional[ReturnClause] = None
    order_by: Optional[Any] = None
    skip: Optional[int] = None
    limit: Optional[int] = None
    create: Optional[MatchPattern] = None
    merge: Optional[MatchPattern] = None
    set_items: Optional[List[SetItem]] = None
    delete_items: Optional[List[str]] = None


_GRAMMAR = r"""
    start: query

    ?query: read_query
          | create_query
          | merge_query
          | set_query
          | delete_query

    read_query: match_clause where_clause? return_clause order_by_clause? skip_clause? limit_clause?
    create_query: CREATE create_pattern return_clause?
    merge_query: MERGE create_pattern return_clause?
    set_query: match_clause where_clause? set_clause return_clause? order_by_clause? skip_clause? limit_clause?
    delete_query: match_clause where_clause? delete_clause return_clause?

    match_clause: MATCH node_pattern edge_pattern node_pattern
    node_pattern: "(" NAME label? ")"

    create_pattern: node_create edge_create node_create
                  | node_create

    node_create: "(" NAME? label? properties? ")"
    edge_create: "-" "[" edge_create_inner? "]" RIGHT_ARROW
               | LEFT_ARROW "[" edge_create_inner? "]" "-"
               | "--"                       -> empty_undirected_edge
    edge_create_inner: NAME? label? properties?

    edge_pattern: "-" "[" edge_inner? "]" RIGHT_ARROW
                | LEFT_ARROW "[" edge_inner? "]" "-"
                | "--"                       -> empty_undirected_edge

    RIGHT_ARROW: "->" | "-"
    LEFT_ARROW: "<-"

    edge_inner: edge_variable? label?
    edge_variable: NAME
    label: ":" NAME

    properties: "{" prop_pair ("," prop_pair)* "}"
    prop_pair: NAME ":" value

    where_clause: WHERE boolean_expr

    ?boolean_expr: or_expr
    ?or_expr: and_expr (OR and_expr)*
    ?and_expr: atom (AND atom)*
    ?atom: predicate
         | "(" boolean_expr ")"

    predicate: property_ref comparison_op value
    comparison_op: OP | EQUAL
    property_ref: NAME "." NAME

    set_clause: SET set_item ("," set_item)*
    set_item: property_ref EQUAL value

    delete_clause: DELETE NAME ("," NAME)*

    return_clause: RETURN DISTINCT? return_list
    return_list: STAR                         -> return_star
               | return_item ("," return_item)*

    return_item: return_expr alias?
    ?return_expr: count_expr
                | property_ref
                | NAME

    count_expr: COUNT "(" DISTINCT? count_arg ")"
    count_arg: STAR | property_ref | NAME

    alias: AS NAME

    order_by_clause: ORDER BY order_item ("," order_item)*
    order_item: order_expr order_direction?
    ?order_expr: property_ref | NAME
    order_direction: ASC | DESC

    skip_clause: SKIP SIGNED_NUMBER
    limit_clause: LIMIT SIGNED_NUMBER

    value: SIGNED_NUMBER      -> number
         | ESCAPED_STRING     -> string
         | SINGLE_STRING      -> single_string
         | TRUE               -> true
         | FALSE              -> false
         | NULL               -> null
         | NAME               -> bareword

    SINGLE_STRING: /'[^']*'/

    MATCH.2: /MATCH/i
    WHERE.2: /WHERE/i
    RETURN.2: /RETURN/i
    DISTINCT.2: /DISTINCT/i
    COUNT.2: /COUNT/i
    AS.2: /AS/i
    ORDER.2: /ORDER/i
    BY.2: /BY/i
    ASC.2: /ASC/i
    DESC.2: /DESC/i
    SKIP.2: /SKIP/i
    LIMIT.2: /LIMIT/i
    AND.2: /AND/i
    OR.2: /OR/i
    CREATE.2: /CREATE/i
    MERGE.2: /MERGE/i
    SET.2: /SET/i
    DELETE.2: /DELETE/i
    TRUE.2: /TRUE/i
    FALSE.2: /FALSE/i
    NULL.2: /NULL/i

    STAR: "*"
    OP.3: "==" | ">=" | "<=" | "<>" | "!=" | ">" | "<"
    EQUAL.3: "="
    NAME: /[A-Za-z_][A-Za-z0-9_]*/

    %import common.SIGNED_NUMBER
    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
"""


@v_args(inline=True)
class _GQLTransformer(Transformer):
    def start(self, query):
        return query

    def read_query(self, match_clause, *items):
        where_clause = None
        return_clause = None
        order_by_clause = None
        skip_clause = None
        limit_clause = None

        for item in items:
            if isinstance(item, WhereClause):
                where_clause = item
            elif isinstance(item, ReturnClause):
                return_clause = item
            elif isinstance(item, dict) and "items" in item:
                order_by_clause = item
            elif isinstance(item, tuple) and len(item) == 2:
                if item[0] == "SKIP":
                    skip_clause = item[1]
                elif item[0] == "LIMIT":
                    limit_clause = item[1]

        return QueryAST(
            query_type="MATCH",
            match=match_clause,
            where=where_clause,
            returns=return_clause,
            order_by=order_by_clause,
            skip=skip_clause,
            limit=limit_clause,
        )

    def create_query(self, _create, create_pattern, return_clause=None):
        return QueryAST(
            query_type="CREATE",
            create=create_pattern,
            returns=return_clause,
        )

    def merge_query(self, _merge, create_pattern, return_clause=None):
        return QueryAST(
            query_type="MERGE",
            merge=create_pattern,
            returns=return_clause,
        )

    def set_query(self, match_clause, *items):
        where_clause = None
        set_items = None
        return_clause = None
        order_by_clause = None
        skip_clause = None
        limit_clause = None

        for item in items:
            if isinstance(item, WhereClause):
                where_clause = item
            elif isinstance(item, list) and all(isinstance(x, SetItem) for x in item):
                set_items = item
            elif isinstance(item, ReturnClause):
                return_clause = item
            elif isinstance(item, dict) and "items" in item:
                order_by_clause = item
            elif isinstance(item, tuple) and len(item) == 2:
                if item[0] == "SKIP":
                    skip_clause = item[1]
                elif item[0] == "LIMIT":
                    limit_clause = item[1]

        return QueryAST(
            query_type="SET",
            match=match_clause,
            where=where_clause,
            set_items=set_items or [],
            returns=return_clause,
            order_by=order_by_clause,
            skip=skip_clause,
            limit=limit_clause,
        )

    def delete_query(self, match_clause, *items):
        where_clause = None
        delete_items = None
        return_clause = None

        for item in items:
            if isinstance(item, WhereClause):
                where_clause = item
            elif isinstance(item, list) and all(isinstance(x, str) for x in item):
                delete_items = item
            elif isinstance(item, ReturnClause):
                return_clause = item

        return QueryAST(
            query_type="DELETE",
            match=match_clause,
            where=where_clause,
            delete_items=delete_items or [],
            returns=return_clause,
        )

    def match_clause(self, _match, left_node, edge, right_node):
        return MatchPattern(left_node=left_node, edge=edge, right_node=right_node)

    def node_pattern(self, name, label=None):
        label_value = None
        if isinstance(label, tuple) and label[0] == "label":
            label_value = label[1]
        elif label:
            label_value = str(label)
        return NodePattern(variable=str(name), label=label_value)

    def create_pattern(self, left_node, edge=None, right_node=None):
        return MatchPattern(left_node=left_node, edge=edge, right_node=right_node)

    def node_create(self, *items):
        variable = None
        label = None
        properties = None

        for item in items:
            if isinstance(item, dict):
                properties = item
            elif isinstance(item, tuple) and item[0] == "label":
                label = item[1]
            elif variable is None:
                variable = str(item)

        # If a node was written as (:Label {...}), Lark only passes label, so variable remains None.
        return NodePattern(variable=variable, label=label, properties=properties)

    def edge_create(self, *items):
        edge_inner = None
        direction = "out"

        for item in items:
            if isinstance(item, dict) and ("variable" in item or "label" in item or "properties" in item):
                edge_inner = item
            elif str(item) == "<-":
                direction = "in"
            elif str(item) == "-":
                direction = "undirected"
            elif str(item) == "->":
                direction = "out"

        variable = None
        label = None
        properties = None
        if isinstance(edge_inner, dict):
            variable = edge_inner.get("variable")
            label = edge_inner.get("label")
            properties = edge_inner.get("properties")

        return EdgePattern(variable=variable, label=label, direction=direction, properties=properties)

    def edge_create_inner(self, *items):
        result = {"variable": None, "label": None, "properties": None}
        for item in items:
            if isinstance(item, dict):
                result["properties"] = item
            elif isinstance(item, tuple) and item[0] == "label":
                result["label"] = item[1]
            elif result["variable"] is None:
                result["variable"] = str(item)
        return result

    def edge_pattern(self, *items):
        edge_inner = None
        direction = "out"

        for item in items:
            if isinstance(item, dict):
                edge_inner = item
            elif str(item) == "<-":
                direction = "in"
            elif str(item) == "-":
                direction = "undirected"
            elif str(item) == "->":
                direction = "out"

        variable = None
        label = None
        if isinstance(edge_inner, dict):
            variable = edge_inner.get("variable")
            label = edge_inner.get("label")

        return EdgePattern(variable=variable, label=label, direction=direction)

    def edge_inner(self, *items):
        result = {"variable": None, "label": None}
        for item in items:
            if isinstance(item, tuple) and item[0] == "edge_variable":
                result["variable"] = item[1]
            elif isinstance(item, tuple) and item[0] == "label":
                result["label"] = item[1]
            else:
                result["label"] = str(item)
        return result

    def empty_undirected_edge(self):
        return EdgePattern(variable=None, label=None, direction="undirected")

    def edge_variable(self, name):
        return ("edge_variable", str(name))

    def label(self, name):
        return ("label", str(name))

    def properties(self, *pairs):
        result = {}
        for key, value in pairs:
            result[key] = value
        return result

    def prop_pair(self, name, value):
        return (str(name), value)

    def where_clause(self, _where, expression):
        return WhereClause(expression=expression)

    def or_expr(self, first, *rest):
        expression = first
        i = 0
        while i < len(rest):
            expression = BooleanExpression(operator=str(rest[i]).upper(), left=expression, right=rest[i + 1])
            i += 2
        return expression

    def and_expr(self, first, *rest):
        expression = first
        i = 0
        while i < len(rest):
            expression = BooleanExpression(operator=str(rest[i]).upper(), left=expression, right=rest[i + 1])
            i += 2
        return expression

    def comparison_op(self, operator):
        op = str(operator)
        return "=" if op == "==" else op

    def predicate(self, property_ref, operator, value):
        variable, prop = property_ref
        op = str(operator)
        return WherePredicate(variable=variable, property=prop, operator=("=" if op == "==" else op), value=value)

    def property_ref(self, variable, prop):
        return (str(variable), str(prop))

    def set_clause(self, _set, *items):
        return list(items)

    def set_item(self, property_ref, _equal, value):
        variable, prop = property_ref
        return SetItem(variable=variable, property=prop, value=value)

    def delete_clause(self, _delete, *names):
        return [str(name) for name in names]

    def return_clause(self, _return, *items):
        distinct = False
        return_list = None
        for item in items:
            if str(item).upper() == "DISTINCT":
                distinct = True
            else:
                return_list = item
        return ReturnClause(items=return_list if isinstance(return_list, list) else [], distinct=distinct)

    def return_star(self, _star):
        return ["*"]

    def return_list(self, *items):
        return list(items)

    def return_item(self, expression, alias=None):
        return ReturnItem(expression=expression, alias=str(alias) if alias else None)

    def alias(self, _as, name):
        return str(name)

    def count_expr(self, _count, *items):
        distinct = False
        argument = "*"
        for item in items:
            if str(item).upper() == "DISTINCT":
                distinct = True
            else:
                argument = item
        return AggregateExpression(function="COUNT", argument=argument, distinct=distinct)

    def count_arg(self, item):
        return "*" if str(item) == "*" else item

    def order_by_clause(self, _order, _by, *items):
        return {"items": list(items)}

    def order_item(self, item, direction=None):
        return {"item": item, "direction": str(direction).upper() if direction else "ASC"}

    def order_direction(self, direction):
        return str(direction).upper()

    def skip_clause(self, _skip, number):
        return ("SKIP", int(str(number)))

    def limit_clause(self, _limit, number):
        return ("LIMIT", int(str(number)))

    def number(self, value):
        text = str(value)
        if "." in text:
            return float(text)
        return int(text)

    def string(self, value):
        return str(value)[1:-1]

    def single_string(self, value):
        return str(value)[1:-1]

    def true(self, _value):
        return True

    def false(self, _value):
        return False

    def null(self, _value):
        return None

    def bareword(self, value):
        return str(value)


class Parser:
    """Minimal GQL-like parser."""

    _parser = Lark(
        _GRAMMAR,
        parser="lalr",
        transformer=_GQLTransformer(),
        maybe_placeholders=False,
        propagate_positions=True,
    )

    @staticmethod
    def Parse(query: str, silent: bool = False):
        """Parses a minimal GQL-like query and returns a QueryAST."""

        if not isinstance(query, str) or not query.strip():
            if not silent:
                print("GQL.Parser.Parse - Error: The input query is not a valid string. Returning None.")
            return None

        try:
            return Parser._parser.parse(query.strip())
        except UnexpectedInput as e:
            if not silent:
                print("GQL.Parser.Parse - Error: Could not parse the input query. Returning None.")
                print(f"Line: {getattr(e, 'line', None)}, Column: {getattr(e, 'column', None)}")
                print("Query:")
                print(query)
                try:
                    print(e.get_context(query))
                except Exception:
                    pass
                expected = getattr(e, "expected", None)
                if expected:
                    print("Expected:", ", ".join(sorted(expected)))
                print("Error:", e)
            return None
        except Exception as e:
            if not silent:
                print("GQL.Parser.Parse - Error: Could not parse the input query. Returning None.")
                print("Query:", query)
                print("Error:", e)
            return None
