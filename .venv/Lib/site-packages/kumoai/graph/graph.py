import copy
import io
import logging
import time
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

if TYPE_CHECKING:
    import graphviz

import kumoapi.data_snapshot as snapshot_api
import kumoapi.graph as api
from kumoapi.common import JobStatus
from kumoapi.data_snapshot import GraphSnapshotID
from tqdm.auto import tqdm
from typing_extensions import Self

from kumoai import global_state
from kumoai.client.graph import GraphID
from kumoai.graph.table import Table
from kumoai.mixin import CastMixin

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 20


@dataclass(frozen=True)
class Edge(CastMixin, api.Edge):
    r"""An edge represents a relationship between two tables in a
    :class:`~kumoai.graph.Graph`. Note that edges are **always** bidirectional
    within the Kumo platform.

    Args:
        src_table: The name of the source table of the edge. This table must
            have a foreign key with name :obj:`fkey` that links to the primary
            key in the destination table.
        fkey: The name of the foreign key in the source table.
        dst_table: The name of the destination table in the graph. This table
            must have a primary key that links to the
            source table's foreign key.

    Example:
        >>> import kumoai
        >>> edge = kumoai.Edge("table_with_fkey", "fkey", "table_with_pkey")
    """
    def __iter__(self) -> Iterator[str]:
        # Allows unwrapping an edge via `src_table, fkey, dst_table = edge`
        return iter((self.src_table, self.fkey, self.dst_table))

    def __hash__(self) -> int:
        return hash((self.src_table, self.fkey, self.dst_table))

    @property
    def _fully_qualified_name(self) -> str:
        return f"{self.src_table}.{self.fkey}.{self.dst_table}"


@dataclass
class GraphHealthStats:
    r"""Graph health statistics contain important statistics that represent the
    healthiness of each defined edge in a graph. These statistics are computed
    as part of a :class:`~kumoai.graph.Graph` snapshot, and can be fetched by
    indexing with an :class:`~kumoai.graph.graph.Edge` object.
    """
    _stats: Dict[str, api.EdgeHealthStatistics]

    def __init__(self, stats: Dict[str, api.EdgeHealthStatistics]):
        self._stats = stats

    def __getitem__(self, key: Edge) -> api.EdgeHealthStatistics:
        return self._stats[key._fully_qualified_name]

    def __repr__(self) -> str:
        representation = "GraphHealthStats\n"
        for key, stats in self._stats.items():
            src, fkey, dst = key.split('.')
            representation += (f" - Edge({src} ({fkey})-> {dst}) \n")
            representation += (f"    - {stats.total_num_edges} total edges\n")
            representation += (
                f"    - {int(stats.absolute_match_stats.src_in_dst)} "
                f"({round(stats.percent_match_stats.src_in_dst, 2)}%) rows "
                f"in {src} have valid edges to {dst}\n")
            representation += (
                f"    - {int(stats.absolute_match_stats.dst_in_src)} "
                f"({round(stats.percent_match_stats.dst_in_src, 2)}%) rows "
                f"in {dst} have valid edges to {src}\n")
        return representation


class Graph:
    r"""A graph defines the relationships between a set of Kumo tables, akin
    to relationships between tables in a relational database. Creating a graph
    is the final step of data definition in Kumo; after a graph is created, you
    are ready to write a :class:`~kumoai.pquery.PredictiveQuery` and train a
    predictive model.


    .. code-block:: python

        import kumoai

        # Define connector to source data:
        connector = kumoai.S3Connector('s3://...')

        # Create Kumo Tables. See Table documentation for more information:
        customer = kumoai.Table(...)
        article = kumoai.Table(...)
        transaction = kumoai.Table(...)

        # Create a graph:
        graph = kumo.Graph(
            # These are the tables that participate in the graph: the keys of this
            # dictionary are the names of the tables, and the values are the Table
            # objects that correspond to these names:
            tables={
                'customer': customer,
                'stock': stock,
                'transaction': transaction,
            },

            # These are the edges that define the primary key / foreign key
            # relationships between the tables defined above. Here, `src_table`
            # is the table that has the foreign key `fkey`, which maps to the
            # table `dst_table`'s primary key:`
            edges=[
                dict(src_table='transaction', fkey='StockCode', dst_table='stock'),
                dict(src_table='transaction', fkey='CustomerID', dst_table='customer'),
            ],
        )

        # Validate the graph configuration, for use in Kumo downstream models:
        graph.validate(verbose=True)

        # Visualize the graph:
        graph.visualize()

        # Fetch the statistics of the tables in this graph (this method will
        # take a graph snapshot, and as a result may have high latency):
        graph.get_table_stats(wait_for="minimal")

        # Fetch link health statistics (this method will
        # take a graph snapshot, and as a result may have high latency):
        graph.get_edge_stats(non_blocking=Falsej)

    Args:
        tables: The tables in the graph, represented as a dictionary mapping
            unique table names (within the context of this graph) to the
            :class:`~kumoai.graph.Table` definition for the table.
        edges: The edges (relationships) between the :obj:`tables` in the
            graph. Edges must specify the source table, foreign key, and
            destination table that they link.

    .. # noqa: E501
    """
    def __init__(
        self,
        tables: Optional[Dict[str, Table]] = None,
        edges: Optional[List[Edge]] = None,
    ) -> None:
        self._tables: Dict[str, Table] = {}
        self._edges: List[Edge] = []

        for name, table in (tables or {}).items():
            self.add_table(name, table)

        for edge in (edges or []):
            self.link(Edge._cast(edge))

        # Cached from backend:
        self._graph_snapshot_id: Optional[GraphSnapshotID] = None

    def print_definition(self) -> None:
        r"""Prints the full definition for this graph; the definition uses
        placeholder names in place of `kumoai.graph.Table` variables. Copy and
        paste this definition, modify the table variable names to re-create
        the original graph.

        Example:
            >>> import kumoai
            >>> graph = kumoai.Graph(...)  # doctest: +SKIP
            >>> graph.print_definition()  # doctest: +SKIP
            Graph(
              tables={
                'table-1' : <table-1>,
                'table-2' : <table-2>,
                ...
                'table-N' : <table-N>,
              },
              edges=[
                Edge(src_table='table-A', fkey='fkey-AD', dst_table='table-D'),
                Edge(src_table='table-B', fkey='fkey-BE', dst_table='table-E'),
                ...
                Edge(src_table='table-C', fkey='fkey-CF', dst_table='table-F'),
              ],
            )
        """
        definition = f"{self.__class__.__name__}(\n"
        definition += "  tables={"
        for table in self._tables.keys():
            definition += f"\n    '{table}' : <{table}>,"
        definition += "\n  },\n"
        definition += "  edges=["
        for edge in self._edges:
            src_table, fkey, dst_table = edge
            definition += (
                f"\n    {edge.__class__.__name__}(src_table='{src_table}', "
                f"fkey='{fkey}', dst_table='{dst_table}'),")
        definition += "\n  ],\n)"
        print(definition)

    # Properties ##############################################################

    @property
    def id(self) -> str:
        r"""Returns the unique ID for this graph, determined from its
        schema and the schemas of the tables and columns that it contains. Two
        graphs with any differences in their constituent tables or columns are
        guaranteed to have unique identifiers.
        """
        return self.save()

    # Save / load #############################################################

    def _to_api_graph_definition(self) -> api.GraphDefinition:
        col_groups_by_dst_table: Dict[str, List[api.ColumnKey]] = dict()
        for edge in self.edges:
            dst_pkey = self[edge.dst_table].primary_key
            if dst_pkey is None:
                raise ValueError(
                    f"The destination table {edge.dst_table} of edge "
                    f"{edge} does not have a primary key.")
            if edge.dst_table not in col_groups_by_dst_table:
                col_groups_by_dst_table[edge.dst_table] = [
                    api.ColumnKey(edge.dst_table, dst_pkey.name)
                ]
            col_groups_by_dst_table[edge.dst_table].append(
                api.ColumnKey(edge.src_table, edge.fkey))

        return api.GraphDefinition(
            tables={
                table_name: table._to_api_table_definition()
                for table_name, table in self.tables.items()
            },
            col_groups=[
                api.ColumnKeyGroup(columns=tuple(col_keys))
                for col_keys in col_groups_by_dst_table.values()
            ],
        )

    @staticmethod
    def _edges_from_api_graph_definition(
            graph_definition: api.GraphDefinition) -> List[Edge]:
        edges: List[Edge] = []
        for col_group in graph_definition.col_groups:
            pkey_col = None
            for col in col_group.columns:
                table_def = graph_definition.tables[col.table_name]
                if col.col_name == table_def.pkey:
                    pkey_col = col
                    break
            assert pkey_col is not None
            for col in col_group.columns:
                if col != pkey_col:
                    edges.append(
                        Edge(src_table=col.table_name, fkey=col.col_name,
                             dst_table=pkey_col.table_name))

        return edges

    @staticmethod
    def _from_api_graph_definition(
            graph_definition: api.GraphDefinition) -> 'Graph':
        tables = {
            k: Table._from_api_table_definition(v)
            for k, v in graph_definition.tables.items()
        }
        edges = Graph._edges_from_api_graph_definition(graph_definition)
        return Graph(tables, edges)

    def save(
        self,
        name: Optional[str] = None,
        skip_validation: bool = False,
    ) -> Union[GraphID, str]:
        r"""Associates this graph with a unique name, that can later be
        used to fetch the graph either in the Kumo UI or in the Kumo SDK
        with method :meth:`~kumoai.Graph.load`.

        Args:
            name: The name to associate with this table definition. If the
                name is already associated with another table, that table will
                be overridden.
            skip_validation: Whether to skip validation of the graph. If
                :obj:`True`, validation will be skipped, but saving an invalid
                graph may result in undefined behavior.
                If :obj:`False`, the graph will be validated before saving.

        Example:
            >>> import kumoai
            >>> graph = kumoai.Graph(...)  # doctest: +SKIP
            >>> graph.save()  # doctest: +SKIP
            graph-xxx
            >>> graph.save("template_name")  # doctest: +SKIP
            >>> loaded = kumoai.Graph.load("template_name")  # doctest: +SKIP
        """
        if not skip_validation:
            self.validate(verbose=False)

        template_resource = (global_state.client.graph_api.get_graph_if_exists(
            graph_id_or_name=name)) if name else None
        if template_resource is not None:
            config = self._from_api_graph_definition(template_resource.graph)
            logger.warning(
                ("Graph template %s already exists, with configuration %s. "
                 "This template will be overridden with configuration %s."),
                name, str(config), str(self))

        # Save as named template
        return global_state.client.graph_api.create_graph(
            graph_def=self._to_api_graph_definition(),
            force_rename=True if name else False,
            name_alias=name,
        )

    @classmethod
    def load(cls, graph_id_or_template: str) -> 'Graph':
        r"""Loads a graph from either a graph ID or a named template. Returns a
        :class:`Graph` object that contains the loaded graph along with its
        associated tables, columns, etc.
        """
        api = global_state.client.graph_api
        res = api.get_graph_if_exists(graph_id_or_template)
        if not res:
            raise ValueError(f"Graph {graph_id_or_template} was not found.")
        out = cls._from_api_graph_definition(res.graph)
        return out

    # Snapshot ################################################################

    @property
    def snapshot_id(self) -> Optional[snapshot_api.GraphSnapshotID]:
        r"""Returns the snapshot ID of this graph's snapshot, if a snapshot
        has been taken. Returns `None` otherwise.

        .. warning::
            This function currently only returns a snapshot ID if a snapshot
            has been taken *in this session.*
        """
        return self._graph_snapshot_id

    def snapshot(
        self,
        *,
        force_refresh: bool = False,
        non_blocking: bool = False,
    ) -> snapshot_api.GraphSnapshotID:
        r"""Takes a *snapshot* of this graph's underlying data, and returns a
        unique identifier for this snapshot.

        This is equivalent to taking a snapshot for each constituent table in
        the graph.  For more information, please see the documentation for
        :meth:`~kumoai.graph.Table.snapshot`.

        .. warning::
            Please familiarize yourself with the warnings for this method in
            :class:`~kumoai.graph.Table` before proceeding.

        Args:
            force_refresh: Indicates whether a snapshot should be taken, if one
                already exists in Kumo. If :obj:`False`, a previously existing
                snapshot may be re-used. If :obj:`True`, a new snapshot is
                always taken.
            non_blocking: Whether this operation should return immediately
                after creating the snapshot, or await completion of the
                snapshot. If :obj:`True`, the snapshot will proceed in the
                background, and will be used for any downstream job.

        Raises:
            RuntimeError: if ``non_blocking`` is set to :obj:`False` and the
                graph snapshot fails.
        """
        if self._graph_snapshot_id is None or force_refresh:
            self.save()
            if not force_refresh:
                snapshotted_table_names: List[str] = []
                for table_name, table in self.tables.items():
                    if table.snapshot_id is not None:
                        snapshotted_table_names.append(table_name)
                if len(snapshotted_table_names) > 0:
                    logger.warning(
                        "Tables %s have already been snapshot, and will not "
                        "be refreshed. If you would like to refresh all "
                        "tables, please set 'force_refresh=True'.",
                        snapshotted_table_names)

            self._graph_snapshot_id = (
                global_state.client.graph_api.create_snapshot(
                    graph_id=self.id,
                    refresh_source=True,
                ))
            logger.info("Graph snapshot with identifier %s created.",
                        self._graph_snapshot_id)

            # Perform initial GET to update table snapshot IDs:
            graph_resource: snapshot_api.GraphSnapshotResource = (
                global_state.client.graph_api.get_snapshot(
                    snapshot_id=self._graph_snapshot_id))
            for table_name, table_id in graph_resource.table_ids.items():
                self[table_name]._table_snapshot_id = table_id

            # NOTE we do not use a `KumoFuture` here as we do not want to treat
            # a graph refresh as having its own state; since we only ever
            # operate on the latest graph version (and do not let users to time
            # travel), there is no need for a separate Future object:
            if not non_blocking:
                stage = snapshot_api.GraphSnapshotStage.INGEST
                table_status: Dict[str, JobStatus] = {
                    table_name: JobStatus.NOT_STARTED
                    for table_name in self.tables
                }

                # Increment progress bar with table refresh stages:
                done = [status.is_terminal for status in table_status.values()]
                graph_done = False
                if logger.isEnabledFor(logging.INFO):
                    pbar = tqdm(total=len(done), unit="table",
                                desc="Ingesting")
                while not (all(done) and graph_done):
                    graph_resource = (
                        global_state.client.graph_api.get_snapshot(
                            snapshot_id=self._graph_snapshot_id))
                    for table_name, table_id in graph_resource.table_ids.items(
                    ):
                        resource = (global_state.client.table_api.get_snapshot(
                            snapshot_id=table_id))
                        table_status[table_name] = resource.stages[
                            stage].status
                    done = [
                        status.is_terminal for status in table_status.values()
                    ]
                    graph_done = graph_resource.stages[
                        stage].status.is_terminal
                    if logger.isEnabledFor(logging.INFO):
                        pbar.update(sum(done) - pbar.n)
                    time.sleep(_DEFAULT_INTERVAL_S)
                if logger.isEnabledFor(logging.INFO):
                    pbar.update(len(done) - pbar.n)
                    pbar.close()

                state = graph_resource.stages[stage]
                status = state.status
                warnings = "\n".join([
                    f"{i}. {message}"
                    for i, message in enumerate(state.warnings)
                ])
                error = state.error
                if status == JobStatus.FAILED:
                    raise RuntimeError(
                        f"Graph snapshot with identifier "
                        f"{self._graph_snapshot_id} failed, with error "
                        f"{error} and warnings {warnings}")
                if len(state.warnings) > 0:
                    logger.warning(
                        "Graph snapshot completed with the following "
                        "warnings: %s", warnings)
        else:
            logger.warning(
                "Graph snapshot with identifier %s already exists, and will "
                "not be refreshed.", self._graph_snapshot_id)

        # <prefix>@<data_version>:
        assert self._graph_snapshot_id is not None
        return self._graph_snapshot_id

    # Statistics ##############################################################

    def get_table_stats(
        self,
        wait_for: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        r"""Returns all currently computed statistics on the latest snapshot of
        this graph. If a snapshot on this graph has not been taken, this method
        will take a snapshot.

        .. note::
            Graph statistics are computed in multiple stages after ingestion is
            complete. These stages are called *minimal* and *full*; minimal
            statistics are always computed before full statistics.

        Args:
            wait_for: Whether this operation should block on the existence of
                statistics availability. This argument can take one of three
                values: :obj:`None`, which indicates that the method should
                return immediately with whatever statistics are present,
                :obj:`"minimal"`, which indicates that the method should return
                the when the minimum, maximum, and fraction of NA values
                statistics are present, or :obj:`"full"`, which indicates that
                the method should return when all computed statistics are
                present.
        """
        assert wait_for is None or wait_for in {"minimal", "full"}

        # Wait for graph ingestion to be done:
        if not self._graph_snapshot_id:
            self.snapshot(force_refresh=False, non_blocking=False)
        assert self._graph_snapshot_id is not None

        # Wait for all table snapshots to match the `wait_for` stage, if
        # we support that:
        if wait_for:
            if wait_for == "minimal":
                stage = snapshot_api.TableSnapshotStage.MIN_COL_STATS
            else:
                stage = snapshot_api.TableSnapshotStage.FULL_COL_STATS

            table_status: Dict[str, JobStatus] = {
                table_name: JobStatus.NOT_STARTED
                for table_name in self.tables
            }
            done = [status.is_terminal for status in table_status.values()]
            if logger.isEnabledFor(logging.INFO):
                pbar = tqdm(total=len(done), unit="table",
                            desc="Computing Statistics")
            while not all(done):
                for table_name, table in self.tables.items():
                    resource = (global_state.client.table_api.get_snapshot(
                        snapshot_id=table._table_snapshot_id))
                    table_status[table_name] = resource.stages[stage].status
                done = [status.is_terminal for status in table_status.values()]
                if logger.isEnabledFor(logging.INFO):
                    pbar.update(sum(done) - pbar.n)
                time.sleep(_DEFAULT_INTERVAL_S)
            if logger.isEnabledFor(logging.INFO):
                pbar.update(len(done) - pbar.n)
                pbar.close()

        # Write out statistics:
        out = {}
        for table_name, table in self.tables.items():
            resource = (global_state.client.table_api.get_snapshot(
                snapshot_id=table._table_snapshot_id))
            out[table_name] = {
                stat.column_name: stat.stats
                for stat in resource.column_stats
            }
        return out

    def get_edge_stats(
        self,
        *,
        non_blocking: bool = False,
    ) -> Optional[GraphHealthStats]:
        """Retrieves edge health statistics for the edges in a graph, if these
        statistics have been computed by a graph snapshot.

        Edge health statistics are returned in a
        :class:`~kumoai.graph.GraphHealthStats` object, and contain information
        about the match rate between primary key / foreign key relationships
        between the tables in the graph.

        Args:
            non_blocking: Whether this operation should return immediately
                after querying edge statistics (returning `None` if statistics
                are not available), or await completion of statistics
                computation.
        """
        if self._graph_snapshot_id is None:
            raise ValueError('In order to calculate edge health statistics, '
                             'you must first create a snapshot of the graph '
                             'on which to calculate match statistics for each '
                             'edge. Please call Graph.snapshot() and then '
                             'this function.')

        edge_health_response = global_state.client.graph_api.get_edge_stats(
            graph_snapshot_id=self._graph_snapshot_id)

        if non_blocking:
            if not edge_health_response.is_ready:
                return None
        else:
            while not edge_health_response.is_ready:
                edge_health_response = (
                    global_state.client.graph_api.get_edge_stats(
                        graph_snapshot_id=self._graph_snapshot_id))

        return GraphHealthStats(edge_health_response.statistics)

    # Tables ##################################################################

    def has_table(self, name: str) -> bool:
        r"""Returns True if a table by `name` is present in this Graph."""
        return name in self._tables

    def table(self, name: str) -> Table:
        r"""Returns a table in this Kumo Graph.

        Raises:
            KeyError: if no such table is present.
        """
        if name not in self._tables:
            raise KeyError(f"Table '{name}' not found in this graph.")
        return self._tables[name]

    def add_table(self, name: str, table: Table) -> 'Graph':
        r"""Adds a table to this Kumo Graph.

        Raises:
            KeyError: if a table with the same name already exists in this
                graph.
        """
        if name in self._tables:
            raise KeyError(
                f"Cannot add table with name '{name}' to this graph; names "
                f"must be globally unique within a graph.")
        self._tables[name] = table
        return self

    def remove_table(self, name: str) -> Self:
        r"""Removes a table from this graph.

        Raises:
            KeyError: if no such table is present.
        """
        if not self.has_table(name):
            raise KeyError(f"Table '{name}' not found in this graph.'")

        del self._tables[name]
        self._edges = [
            edge for edge in self._edges
            if edge.src_table != name and edge.dst_table != name
        ]
        return self

    @property
    def tables(self) -> Dict[str, Table]:
        r"""Returns a list of all :class:`~kumoai.graph.Table` objects that
        are contained in this graph.
        """
        return self._tables

    def infer_metadata(self, inplace: bool = True) -> 'Graph':
        r"""Infers metadata for the tables in this Graph, by inferring the
        metadata of each table in the graph. For more information, please
        see the documentation for
        :meth:`~kumoai.table.Table.infer_metadata`.
        """
        out = self
        if not inplace:
            out = copy.deepcopy(self)

        for table in out.tables.values():
            table.infer_metadata(inplace=True)
        return out

    # Edges ###################################################################

    def infer_links(self) -> 'Graph':
        r"""Infers edges for the tables in this Graph. It adds edges to the
        graph.

        Note that the function only works if the graph edges are empty.
        """
        if self._edges is not None and len(self._edges) > 0:
            raise ValueError(
                "Cannot infer links if graph edges are not empty.")

        graph_def_with_col_groups = global_state.client.graph_api.infer_links(
            graph=self._to_api_graph_definition())

        edges = Graph._edges_from_api_graph_definition(
            graph_def_with_col_groups)

        for edge in (edges or []):
            logger.info("Inferring edge: %s", edge)
            self.link(Edge._cast(edge))
        return self

    def link(self, *args: Optional[Union[str, Edge]],
             **kwargs: str) -> 'Graph':
        r"""Links two tables (:obj:`src_table` and :obj:`dst_table`) from the
        foreign key :obj:`fkey` in the source table to the primary key in the
        destination table. These edges are treated bidirectionally in Kumo.

        Args:
            *args: Any arguments to construct a
                :class:`kumoai.graph.Edge`, or a :class:`kumoai.graph.Edge`
                itself.
            **kwargs: Any keyword arguments to construct a
                :class:`kumoai.graph.Edge`.

        Raises:
            ValueError: if the edge is already present in the graph, if the
                source table does not exist in the graph, if the destination
                table does not exist in the graph, if the source key does not
                exist in the source table, or if the primary key of the source
                table is being treated as a foreign key.
        """
        edge = Edge._cast(*args, **kwargs)
        if edge is None:
            raise ValueError("Cannot add a 'None' edge to a graph.")

        (src_table, fkey, dst_table) = edge

        if edge in self._edges:
            raise ValueError(f"Cannot add edge {edge} to graph; edge is "
                             f"already present.")

        if src_table not in self._tables:
            raise ValueError(
                f"Source table '{src_table}' does not exist in the graph. "
                f"Please add it via `Graph.add_table(...)` before proceeding.")

        if dst_table not in self._tables:
            raise ValueError(
                f"Destination table '{dst_table}' does not exist in the "
                f"graph. Please add it via `Graph.add_table(...)` before "
                f"proceeding.")

        if fkey not in self._tables[src_table]:
            raise ValueError(
                f"Source key '{fkey}' does not exist in source table "
                f"'{src_table}'; please check that you have added it as a "
                f"column.")

        # Backend limitations: ensure the source is not its primary key:
        src_pkey = self.table(src_table).primary_key
        src_is_pkey = src_pkey is not None and src_pkey.name == fkey
        if src_is_pkey:
            raise ValueError(f"Cannot treat the primary key of table "
                             f"'{src_table}' as a foreign key; please "
                             f"select a different key.")

        self._edges.append(edge)
        return self

    def unlink(self, *args: Optional[Union[str, Edge]],
               **kwargs: str) -> 'Graph':
        r"""Removes an edge added to a Kumo Graph.

        Args:
            *args: Any arguments to construct a
                :class:`~kumoai.graph.Edge`, or a :class:`~kumoai.graph.Edge`
                itself.
            **kwargs: Any keyword arguments to construct a
                :class:`~kumoai.graph.Edge`.

        Raises:
            ValueError: if the edge is not present in the graph.
        """
        edge = Edge._cast(*args, **kwargs)
        if edge not in self._edges:
            raise ValueError(f"Edge {edge} is not present in {self._edges}")
        self._edges.remove(edge)
        return self

    @property
    def edges(self) -> List[Edge]:
        r"""Returns a list of all :class:`~kumoai.graph.Edge` objects that
        represent links in this graph.
        """
        return self._edges

    def validate(self, verbose: bool = True) -> Self:
        r"""Validates a Graph to ensure that all relevant metadata is specified
        for its Tables and Edges.

        Concretely, validation ensures that all tables are valid (see
        :meth:`~kumoai.graph.table.validate` for more information), and that
        edges properly link primary keys and foreign keys between valid tables.
        It additionally ensures that primary and foreign keys between tables
        in an edge are of the same data type, so that unexpected mismatches do
        not occur within the Kumo platform.

        Example:
            >>> import kumoai
            >>> graph = kumoai.Graph(...)  # doctest: +SKIP
            >>> graph.validate()  # doctest: +SKIP
            ValidationResponse(warnings=[], errors=[])

        Args:
            verbose: Whether to log non-error output of this validation.

        Raises:
            ValueError:
                if validation fails.
        """
        # Validate table definitions, so we can properly create a graph
        # definition:
        for table_name, table in self.tables.items():
            try:
                table.validate(verbose=verbose)
            except ValueError as e:
                raise ValueError(
                    f"Validation of table {table_name} failed. {e}") from e

        resp = global_state.client.graph_api.validate_graph(
            api.GraphValidationRequest(self._to_api_graph_definition()))
        if not resp.ok:
            raise ValueError(resp.error_message())
        if verbose:
            if resp.empty():
                logger.info("Graph is configured correctly.")
            else:
                logger.warning(resp.message())
        return self

    def visualize(
        self,
        path: Optional[Union[str, io.BytesIO]] = None,
        show_cols: bool = True,
    ) -> 'graphviz.Graph':
        r"""Visualizes the tables and edges in this graph using the
        ``graphviz`` library.

        Args:
            path: An optional local path to write the produced image to. If
                None, the image will not be written to disk.
            show_cols: Whether to show all columns of every table in the graph.
                If False, will only show the primary key, foreign key(s),
                time column, and end time column of each table.

        Returns:
            A ``graphviz.Graph`` instance representing the visualized graph.
        """
        def has_graphviz_executables() -> bool:
            import graphviz
            try:
                graphviz.Digraph().pipe()
            except graphviz.backend.ExecutableNotFound:
                return False

            return True

        # Check basic dependency:
        if not find_spec('graphviz'):
            raise ModuleNotFoundError(
                "The `graphviz` Python package is required for visualization.")
        elif not has_graphviz_executables():
            raise RuntimeError(
                "Could not visualize graph as `graphviz` executables have not "
                "been installed. These dependencies are required in addition "
                "to the `graphviz` Python package. Please install them to "
                "continue. Instructions at https://graphviz.org/download/.")
        else:
            import graphviz

        fmt = None
        if isinstance(path, str):
            fmt = path.split('.')[-1]
        elif isinstance(path, io.BytesIO):
            fmt = 'svg'
        graph = graphviz.Graph(format=fmt)

        def left_align(list_of_text: List[str]) -> str:
            return '\\l'.join(list_of_text) + '\\l'

        table_to_fkey: Dict[str, List[str]] = {}
        for edge in self.edges:
            src, fkey, dst = edge
            if src not in table_to_fkey:
                table_to_fkey[src] = []
            table_to_fkey[src].append(fkey)

        for table_name, table in self.tables.items():
            keys = []
            if table.has_primary_key():
                assert table.primary_key is not None
                keys += [f'{table.primary_key.name} (PK)']
            if table_name in table_to_fkey:
                keys += [f'{fkey} (FK)' for fkey in table_to_fkey[table_name]]
            if table.has_time_column():
                assert table.time_column is not None
                keys += [f'{table.time_column.name} (Time)']
            if table.has_end_time_column():
                assert table.end_time_column is not None
                keys += [f'{table.end_time_column.name} (End Time)']

            keys_aligned = left_align(keys)

            cols = []
            cols_aligned = ""
            if show_cols and len(table.columns) > 0:
                cols += [
                    f'{col.name}: {col.stype or "???"} ({col.dtype or "???"})'
                    for col in table.columns
                ]
                cols_aligned = left_align(cols)

            if cols:
                label = f'{{{table_name}|{keys_aligned}|{cols_aligned}}}'
            else:
                label = f'{{{table_name}|{keys_aligned}}}'

            graph.node(table_name, shape='record', label=label)

        for edge in self.edges:
            src, fkey, dst = edge
            pkey_obj = self[dst].primary_key
            assert pkey_obj is not None
            pkey = pkey_obj.name
            # Print both key names if different:
            if fkey != pkey:
                label = f' {fkey}\n< >\n{pkey} '
            else:
                label = f' {fkey} '
            headlabel, taillabel = '1', '*'
            graph.edge(src, dst, label=label, headlabel=headlabel,
                       taillabel=taillabel, minlen='2', fontsize='11pt',
                       labeldistance='1.5')

        if isinstance(path, str):
            path = '.'.join(path.split('.')[:-1])
            graph.render(path, cleanup=True)
        elif isinstance(path, io.BytesIO):
            path.write(graph.pipe())
        else:
            try:
                graph.view()
            except Exception as e:
                logger.warning(
                    "Could not visualize graph due to an unexpected error in "
                    "`graphviz`. If you are in a notebook environment, "
                    "consider calling `display()` on the returned object "
                    "from `visualize()`. Error: %s", e)
        return graph

    # Class properties ########################################################

    def __hash__(self) -> int:
        return hash((tuple(self.edges), self.tables.values()))

    def __contains__(self, name: str) -> bool:
        return self.has_table(name)

    def __getitem__(self, name: str) -> Table:
        return self.table(name)

    def __delitem__(self, name: str) -> None:
        self.remove_table(name)

    def __repr__(self) -> str:
        table_names = str(list(self._tables.keys())).replace("'", "")
        return (f'{self.__class__.__name__}(\n'
                f'  tables={table_names},\n'
                f'  edges={self._edges},\n'
                f')')
