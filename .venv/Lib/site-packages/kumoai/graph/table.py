import copy
import logging
import time
from typing import Any, Dict, List, Optional, Union

import kumoapi.data_snapshot as snapshot_api
import kumoapi.table as api
import pandas as pd
from kumoapi.common import JobStatus
from kumoapi.data_snapshot import TableSnapshotID
from kumoapi.typing import Stype
from typing_extensions import Self

from kumoai import global_state
from kumoai.client.table import TableID
from kumoai.connector import SourceColumn, SourceTable
from kumoai.graph.column import Column

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 20


class Table:
    r"""A Table represents metadata information for a table in a Kumo
    :class:`~kumoai.graph.Graph`.

    Whereas a :class:`~kumoai.connector.SourceTable` is simply a reference to a
    table behind a backing :class:`~kumoai.connector.Connector`, a table fully
    specifies the relevant metadata (including selected source columns, column
    data type and semantic type, and relational constraint information)
    necessary to train a :class:`~kumoai.pquery.PredictiveQuery` on graph of
    tables. A table can either be constructed explicitly, or with the
    convenience method :meth:`~kumoai.graph.Table.from_source_table`.

    .. code-block:: python

        import kumoai

        # Define connector to source data:
        connector = kumoai.S3Connector('s3://...')

        # Create table using `from_source_table`:
        customer = kumoai.Table.from_source_table(
            source_table=connector['customer'],
            primary_key='CustomerID',
        )

        # Create a table by constructing it directly:
        customer = kumoai.Table(
            source_table=connector['customer'],
            columns=[kumoai.Column(name='CustomerID', dtype='string', stype='ID')],
            primary_key='CustomerID',
        )

        # Infer any missing metadata in the table, from source table
        # properties:
        print("Current metadata: ", customer.metadata)
        customer.infer_metadata()

        # Validate the table configuration, for use in Kumo downstream models:
        customer.validate(verbose=True)

        # Fetch statistics from a snapshot of this table (this method will
        # take a table snapshot, and as a result may have high latency):
        customer.get_stats(wait_for="minimal")

    Args:
        source_table: The source table this Kumo table is created from.
        columns: The selected columns of the source table that are part of this
            Kumo table. Note that each column must specify its data type and
            semantic type; see the :class:`~kumoai.graph.Column` documentation
            for more information. If `None`  all columns from the
            source table are included by default.
        primary_key: The primary key of the table, if present. The primary key
            must exist in the :obj:`columns` argument.
        time_column: The time column of the table, if present. The time column
            must exist in the :obj:`columns` argument.
        end_time_column: The end time column of the table, if present. The end
            time column must exist in the :obj:`columns` argument.
    """  # noqa: E501

    def __init__(
        self,
        source_table: SourceTable,
        columns: Optional[List[Union[SourceColumn, Column]]] = None,
        primary_key: Optional[str] = None,
        time_column: Optional[str] = None,
        end_time_column: Optional[str] = None,
    ) -> None:
        # Reference to the source (raw) table:
        self.source_table = source_table
        self.source_name = source_table.name

        # Columns. Note that there is no distinction between columns treated as
        # features and those treated as constraints at this stage. The
        # treatment of columns as "feature" or "schema-only" columns will be
        # decided at the model plan stage (e.g. by encoding as `Null()`):
        self._columns: Dict[str, Column] = {}

        # Basic schema. This information is defined at the table level:
        self._primary_key: Optional[str] = None
        self._time_column: Optional[str] = None
        self._end_time_column: Optional[str] = None

        # Update values:
        if columns is None:
            columns = list(source_table.column_dict.values())
        for col in (columns or []):
            if isinstance(col, SourceColumn):
                col = Column(name=col.name, stype=col.stype, dtype=col.dtype)
            self.add_column(Column._cast(col))
        self.primary_key = Column._cast(primary_key)
        self.time_column = Column._cast(time_column)
        self.end_time_column = Column._cast(end_time_column)

        # Cached from backend. Note there is no such thing as a table resource
        # as tables are only persisted in the context of a graph. However,
        # table snapshot resources exist, as tables can be ingested and have
        # data fetched:
        self._table_snapshot_id: Optional[TableSnapshotID] = None

    @staticmethod
    def from_source_table(
        source_table: SourceTable,
        column_names: Optional[List[str]] = None,
        primary_key: Optional[str] = None,
        time_column: Optional[str] = None,
        end_time_column: Optional[str] = None,
    ) -> 'Table':
        r"""Creates a Kumo Table from a source table. If no column names are
        specified, all source columns are included by default.

        Args:
            source_table: The :class:`~kumoai.connector.SourceTable` object
                that this table is constructed on.
            column_names: A list of columns to include from the source table;
                if not specified, all columns are included by default.
            primary_key: The name of the primary key of this table, if it
                exists.
            time_column: The name of the time column of this table, if it
                exists.
            end_time_column: The name of the end time column of this table, if
                it exists.
        """
        cols = [
            Column(name, col.stype, col.dtype)
            for name, col in source_table.column_dict.items()
            if (name in column_names if column_names is not None else True)
        ]
        out = Table(source_table, cols)
        out.primary_key = Column._cast(primary_key)
        out.time_column = Column._cast(time_column)
        out.end_time_column = Column._cast(end_time_column)
        return out

    def print_definition(self) -> None:
        r"""Prints the full definition for this table; this definition can be
        copied-and-pasted verbatim to re-create this table.
        """
        pkey_name = (f"\"{self.primary_key.name}\""
                     if self.primary_key is not None else "None")
        t_name = (f"\"{self.time_column.name}\""
                  if self.time_column is not None else "None")
        et_name = (f"\"{self.end_time_column.name}\""
                   if self.end_time_column is not None else "None")
        col_dict = "\n".join([f'    {c},' for c in self.columns])
        source_repr = f"{self.source_table.connector}[\"{self.source_name}\"]"
        print(f'{self.__class__.__name__}(\n'
              f'  source_table={source_repr},\n'
              f'  primary_key={pkey_name},\n'
              f'  time_column={t_name},\n'
              f'  end_time_column={et_name},\n'
              f'  columns=[\n{col_dict}\n'
              f'  ],\n'
              f')')

    # Data column #############################################################

    def has_column(self, name: str) -> bool:
        r"""Returns True if this table has column with name :obj:`name`; False
        otherwise.
        """
        return name in self._columns

    def column(self, name: str) -> Column:
        r"""Returns the data column named with name :obj:`name` in this table,
        or raises a :obj:`KeyError` if no such column is present.

        Raises:
            :class:`KeyError`
                if :obj:`name` is not present in this table.
        """
        if not self.has_column(name):
            raise KeyError(
                f"Column '{name}' not found in table '{self.source_name}'")
        return self._columns[name]

    @property
    def columns(self) -> List[Column]:
        r"""Returns a list of :class:`~kumoai.Column` objects that represent
        the columns in this table.
        """
        return list(self._columns.values())

    def add_column(self, *args: Any, **kwargs: Any) -> None:
        r"""Adds a :obj:`~kumoai.graph.Column` to this table. A column can
        either be added by directly specifying its configuration in this call,
        or by creating a Column object and passing it as an argument.

        Example:
            >>> import kumoai
            >>> table = kumoai.Table(source_table=...)  # doctest: +SKIP
            >>> table.add_column(name='col1', dtype='string')  # doctest: +SKIP
            >>> table.add_column(kumoai.Column('col2', 'int'))  # doctest: +SKIP

        .. # noqa: E501
        """
        col = Column._cast(*args, **kwargs)
        if col is None:
            raise ValueError("Cannot add a 'None' column to a table.")
        if self.has_column(col.name):
            self._columns[col.name].update(col)
        else:
            self._columns[col.name] = col

    def remove_column(self, name: str) -> Self:
        r"""Removes a :obj:`~kumoai.graph.Column` from this table.

        Raises:
            :class:`KeyError`
                if :obj:`name` is not present in this table.
        """
        if not self.has_column(name):
            raise KeyError(
                f"Column '{name}' not found in table '{self.source_name}'")

        if self.has_primary_key() and self._primary_key == name:
            self.primary_key = None
        if self.has_time_column() and self._time_column == name:
            self.time_column = None
        if self.has_end_time_column() and self._end_time_column == name:
            self.end_time_column = None
        del self._columns[name]
        return self

    # Primary key #############################################################

    def has_primary_key(self) -> bool:
        r"""Returns :obj:`True` if this table has a primary key; :obj:`False`
        otherwise.
        """
        return self._primary_key is not None

    @property
    def primary_key(self) -> Optional[Column]:
        r"""The primary key column of this table.

        The getter returns the primary key column of this table, or None if no
        such primary key is present.

        The setter sets a column as a primary key on this table, and raises a
        :class:`ValueError` if the primary key has a non-ID semantic type.
        """
        if not self.has_primary_key():
            return None
        assert self._primary_key is not None
        return self._columns[self._primary_key]

    @primary_key.setter
    def primary_key(self, *args: Any, **kwargs: Any) -> Self:
        col = Column._cast(*args, **kwargs)
        if col is None:
            self._primary_key = None
            return self

        if col.stype is not None and col.stype != Stype.ID:
            raise ValueError(
                f"The semantic type of a primary key must be 'ID' (got "
                f"{col.stype}).")

        col.stype = Stype.ID
        self.add_column(col)
        self._primary_key = col.name
        return self

    # Time column #############################################################

    def has_time_column(self) -> bool:
        r"""Returns :obj:`True` if this table has a time column; :obj:`False`
        otherwise.
        """
        return self._time_column is not None

    @property
    def time_column(self) -> Optional[Column]:
        r"""The time column of this table.

        The getter returns the time column of this table, or :obj:`None` if no
        such time column is present.

        The setter sets a column as a time column on this table, and raises a
        :class:`ValueError` if the time column is the same as the end time
        column, or has a non-timestamp semantic type.
        """
        if not self.has_time_column():
            return None
        assert self._time_column is not None
        return self._columns[self._time_column]

    @time_column.setter
    def time_column(self, *args: Any, **kwargs: Any) -> Self:
        col = Column._cast(*args, **kwargs)
        if col is None:
            self._time_column = None
            return self

        if self.has_end_time_column() and self._end_time_column == col.name:
            raise ValueError(f"Cannot set the time column ('{col.name}') "
                             f"to be the same as the end time column "
                             f"('{self._end_time_column}')")

        if col.stype is not None and col.stype != Stype.timestamp:
            raise ValueError(
                f"The semantic type of a time column must be 'timestamp' (got "
                f"{col.stype}).")

        col.stype = Stype.timestamp
        self.add_column(col)
        self._time_column = col.name
        return self

    # End time column #########################################################

    def has_end_time_column(self) -> bool:
        r"""Returns :obj:`True` if this table has an end time column;
        :obj:`False` otherwise.
        """
        return self._end_time_column is not None

    @property
    def end_time_column(self) -> Optional[Column]:
        r"""The end time column of this table.

        The getter returns the end time column of this table, or :obj:`None` if
        no such column is present.

        The setter sets a column as a time column on this table, and raises a
        :class:`ValueError` if the time column is the same as the end time
        column, or has a non-timestamp semantic type.
        """
        if not self.has_end_time_column():
            return None
        assert self._end_time_column is not None
        return self._columns[self._end_time_column]

    @end_time_column.setter
    def end_time_column(self, *args: Any, **kwargs: Any) -> Self:
        col = Column._cast(*args, **kwargs)
        if col is None:
            self._end_time_column = None
            return self

        if self.has_time_column() and self._time_column == col.name:
            raise ValueError(f"Cannot set the end time column ('{col.name}') "
                             f"to be the same as the time column "
                             f"('{self._time_column}')")

        if col.stype is not None and col.stype != Stype.timestamp:
            raise ValueError(
                f"The semantic type of an end time column must be 'timestamp' "
                f"(got {col.stype}).")

        col.stype = Stype.timestamp
        self.add_column(col)
        self._end_time_column = col.name
        return self

    # Metadata ################################################################

    @property
    def metadata(self) -> pd.DataFrame:
        r"""Returns a :class:`~pandas.DataFrame` object containing Kumo metadata
        information about the columns in this table.

        The returned dataframe has columns ``name``, ``dtype``, ``stype``,
        ``is_primary_key``, ``is_time_column``, and ``is_end_time_column``,
        which provide an aggregate view of the properties of the columns of
        this table.

        Example:
            >>> import kumoai
            >>> table = kumoai.Table(source_table=...)  # doctest: +SKIP
            >>> table.add_column(name='CustomerID', dtype='float64', stype='ID')  # doctest: +SKIP
            >>> table.metadata  # doctest: +SKIP
                name        dtype       stype    is_time_column  is_end_time_column
            0   CustomerID  float64     ID       False           False
        """  # noqa: E501
        items = self._columns.items()
        col_names: List[str] = [i[0] for i in items]
        cols: List[Column] = [i[1] for i in items]

        return pd.DataFrame({
            'name':
            pd.Series(dtype=str, data=col_names),
            'dtype':
            pd.Series(
                dtype=str, data=[
                    c.dtype.value if c.dtype is not None else None
                    for c in cols
                ]),
            'stype':
            pd.Series(
                dtype=str, data=[
                    c.stype.value if c.stype is not None else None
                    for c in cols
                ]),
            'is_primary_key':
            pd.Series(dtype=bool, data=[self.primary_key == c for c in cols]),
            'is_time_column':
            pd.Series(dtype=bool, data=[self.time_column == c for c in cols]),
            'is_end_time_column':
            pd.Series(dtype=bool,
                      data=[self.end_time_column == c for c in cols]),
        })

    def infer_metadata(self, inplace: bool = True) -> Self:
        r"""Infers all metadata for this table's specified columns, including
        the column data types, semantic types, timestamp formats, primary keys,
        and time/end-time columns

        Args:
            inplace: Whether the method should modify the table columns in
                place or return a new :class:`~kumoai.graph.Table` object.

        .. note::
            This method in-place modifies the Table object if `inplace = True`,
            and returns a copy if ``inplace = False``.
        """
        col_requests: List[api.ColumnMetadataRequest] = []
        for col in self.columns:
            col_requests.append(
                # stype and dtype are None to support inferral:
                api.ColumnMetadataRequest(
                    name=col.name,
                    stype=None,
                    dtype=None,
                    timestamp_format=col.timestamp_format,
                ))

        pk_name: Optional[str] = None
        if self.has_primary_key():
            pk_name = self.primary_key.name  # type: ignore

        tc_name: Optional[str] = None
        if self.has_time_column():
            tc_name = self.time_column.name  # type: ignore

        request = api.TableMetadataRequest(
            cols=col_requests,
            source_table=self.source_table._to_api_source_table(),
            pkey=pk_name,
            time_col=tc_name,
        )

        response = global_state.client.table_api.infer_metadata(request)
        inferred_cols: Dict[str, api.Column] = {
            col.name: col
            for col in response.cols
        }

        # Handle inplace:
        out = self
        if not inplace:
            out = copy.deepcopy(self)

        # TODO(manan): respect user overrides
        # TODO(manan): what happens when the ts format is set based on an
        # override?
        for col in out.columns:
            inferred_col = inferred_cols[col.name]

            col.dtype = inferred_col.dtype
            col.stype = inferred_col.stype
            col.timestamp_format = (col.timestamp_format
                                    or inferred_col.timestamp_format)

        # TODO(manan): support end-time column
        if not out.has_primary_key() and response.pkey is not None:
            out.primary_key = response.pkey
        if not out.has_time_column() and response.time_col is not None:
            out.time_column = response.time_col

        # Override for Kumo backend, always:
        if out.has_primary_key():
            out.primary_key.stype = Stype.ID  # type: ignore

        if out.has_time_column():
            out.time_column.stype = Stype.timestamp  # type: ignore

        if out.has_end_time_column():
            out.end_time_column.stype = Stype.timestamp  # type: ignore

        return out

    def _validate_definition(self) -> None:
        for col in self.columns:
            if col.dtype is None or col.stype is None:
                raise ValueError(
                    f"Column {col.name} is not fully specified. Please "
                    f"specify this column's data type and semantic type "
                    f"before proceeding. {col.name} currently has a "
                    f"data type of {col.dtype} and semantic type of "
                    f"{col.stype}.")

    def validate(self, verbose: bool = True) -> Self:
        r"""Validates a Table to ensure that all relevant metadata is specified
        for a table to be used in a downstream :class:`~kumoai.graph.Graph` and
        :class:`~kumoai.pquery.PredictiveQuery`.

        Conceretely, validation ensures that all columns have valid
        data and semantic types, with respect to the table's source data.
        For example, if a text column is assigned a ``dtype`` of ``"int"``,
        this method will raise an exception detailing the mismatch. Similarly,
        if a column cannot be cast from its source data type to the specified
        data type (*e.g* ``"int"`` to ``"binary"``), this method will raise an
        exception.

        .. warning::
            Data type validation is performed on a sample of table data. A
            valid response may not indicate your entire data source is
            configured correctly.

        Args:
            verbose: Whether to log non-error output of this validation.

        Example:
            >>> import kumoai
            >>> table = kumoai.Table(...)  # doctest: +SKIP
            >>> table.validate()  # doctest: +SKIP

        Raises:
            ValueError:
                if validation fails.
        """
        self._validate_definition()

        # Actual heavy lifting:
        resp = global_state.client.table_api.validate_table(
            api.TableValidationRequest(self._to_api_table_definition()))
        if not resp.ok:
            raise ValueError(resp.error_message())
        if verbose:
            if resp.empty():
                logger.info("Table %s is configured correctly.",
                            self.source_name)
            else:
                logger.warning(resp.message())
        return self

    # Snapshot ################################################################

    @property
    def snapshot_id(self) -> Optional[snapshot_api.TableSnapshotID]:
        r"""Returns the snapshot ID of this table's snapshot, if a snapshot
        has been taken. Returns `None` otherwise.

        .. warning::
            This property currently only returns a snapshot ID if a snapshot
            has been taken *in this session.*
        """
        return self._table_snapshot_id

    def snapshot(
        self,
        *,
        force_refresh: bool = False,
        non_blocking: bool = False,
    ) -> snapshot_api.TableSnapshotID:
        r"""Takes a *snapshot* of this table's underlying data, and returns a
        unique identifier for this snapshot.

        The *snapshot* functionality allows one to freeze a table in time, so
        that underlying data changes do not require Kumo to re-process the
        data. This allows for fast iterative machine learning model
        development, on a consistent set of input data.

        .. warning::
            Please note that snapshots are intended to freeze tables in
            time, and not to allow for "time-traveling" to an earlier version
            of data with a prior snapshot. In particular, this means that a
            table can only have one version of a snapshot, which represents
            the latest snapshot taken for that table.

        .. note::
            If you are using Kumo as a Snowpark Container Services native
            application, please note that *snapshot* is a no-op for all
            non-view tables.

        Args:
            force_refresh: Indicates whether a snapshot should be taken, if one
                already exists in Kumo. If :obj:`False`, a previously existing
                snapshot may be re-used. If :obj:`True`, a new snapshot is
                always taken.
            non_blocking: Whether this operation should return immediately
                after creating the snapshot, or await completion of the
                snapshot. If :obj:`True`, the snapshot will proceed in the
                background, and will be used for any downstream job.
        """
        if self._table_snapshot_id is None or force_refresh:
            self._table_snapshot_id = (
                global_state.client.table_api.create_snapshot(
                    table_definition=self._to_api_table_definition(),
                    refresh_source=True,
                ))

            stage = snapshot_api.TableSnapshotStage.INGEST
            resource: snapshot_api.TableSnapshotResource = (
                global_state.client.table_api.get_snapshot(
                    snapshot_id=self._table_snapshot_id))

            if not non_blocking:
                status = resource.stages[stage].status
                while not status.is_terminal:
                    # TODO(manan, siyang): fix start and end time
                    resource = (global_state.client.table_api.get_snapshot(
                        snapshot_id=self._table_snapshot_id))
                    logger.info(
                        "Awaiting snapshot completion: current status is %s ",
                        status)
                    time.sleep(_DEFAULT_INTERVAL_S)
                    status = resource.stages[stage].status

                state = resource.stages[stage]
                status = state.status
                warnings = "\n".join([
                    f"{i}. {message}"
                    for i, message in enumerate(state.warnings)
                ])
                error = state.error
                if status == JobStatus.FAILED:
                    raise RuntimeError(
                        f"Table snapshot with identifier "
                        f"{self._table_snapshot_id} failed, with error "
                        f"{error} and warnings {warnings}")
                if len(state.warnings) > 0:
                    logger.warning(
                        "Table snapshot completed with the following "
                        "warnings: %s", warnings)

        # <prefix>@<data_version>
        assert self._table_snapshot_id is not None
        return self._table_snapshot_id

    def get_stats(
        self,
        wait_for: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        r"""Returns all currently computed statistics on the latest snapshot of
        this table. If a snapshot on this table has not been taken, this method
        will take a snapshot.

        .. note::
            Table statstics are computed in multiple stages after ingestion is
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

        # Attempt to snapshot, use cached snapshot if possible:
        if not self._table_snapshot_id:
            self.snapshot(force_refresh=False, non_blocking=False)
        assert self._table_snapshot_id is not None

        # Fetch resource:
        resource: snapshot_api.TableSnapshotResource = (
            global_state.client.table_api.get_snapshot(
                snapshot_id=self._table_snapshot_id))

        # Wait for a stage, if we need to:
        if wait_for:
            if wait_for == "minimal":
                stage = snapshot_api.TableSnapshotStage.MIN_COL_STATS
            else:
                stage = snapshot_api.TableSnapshotStage.FULL_COL_STATS

            status = resource.stages[stage].status
            while not status.is_terminal:
                resource = (global_state.client.table_api.get_snapshot(
                    snapshot_id=self._table_snapshot_id))
                logger.info(
                    "Awaiting %s column statistics: current status is %s ",
                    wait_for, status)
                time.sleep(_DEFAULT_INTERVAL_S)
                status = resource.stages[stage].status

        # Write out statistics:
        out = {}
        col_stats = resource.column_stats
        for stat in (col_stats or []):
            out[stat.column_name] = stat.stats
        return out

    # Persistence #############################################################

    def _to_api_table_definition(self) -> api.TableDefinition:
        # TODO(manan): type narrowing?
        pk_name: Optional[str] = None
        if self.has_primary_key():
            pk_name = self.primary_key.name  # type: ignore

        tc_name: Optional[str] = None
        if self.has_time_column():
            tc_name = self.time_column.name  # type: ignore

        etc_name: Optional[str] = None
        if self.has_end_time_column():
            etc_name = self.end_time_column.name  # type: ignore

        return api.TableDefinition(
            cols=[
                api.Column(col.name, col.stype, col.dtype,
                           col.timestamp_format) for col in self.columns
            ],
            source_table=self.source_table._to_api_source_table(),
            pkey=pk_name,
            time_col=tc_name,
            end_time_col=etc_name,
        )

    @staticmethod
    def _from_api_table_definition(
            table_definition: api.TableDefinition) -> 'Table':
        return Table(
            source_table=SourceTable._from_api_table_definition(
                table_definition),
            columns=[
                Column(col.name, col.stype, col.dtype, col.timestamp_format)
                for col in table_definition.cols
            ],
            primary_key=table_definition.pkey,
            time_column=table_definition.time_col,
            end_time_column=table_definition.end_time_col,
        )

    def save(self, name: Optional[str] = None) -> Union[TableID, str]:
        r"""Associates this table with a unique name, that can later be
        used to fetch the table either in the Kumo UI or in the Kumo SDK
        with method :meth:`~kumoai.Table.load`.

        Args:
            name: The name to associate with this table definition. If the
                name is already associated with another table, that table will
                be overridden.

        Example:
            >>> import kumoai
            >>> table = kumoai.Table(...)  # doctest: +SKIP
            >>> unique_id = table.save()  # doctest: +SKIP
            >>> loaded = kumoai.Table.load(unique_id) # doctest: +SKIP
            >>> name = table.save("name")  # doctest: +SKIP
            >>> loaded = kumoai.Table.load("name")  # doctest: +SKIP
        """
        self.validate(verbose=False)
        template_resource = (global_state.client.table_api.get_table_if_exists(
            table_id_or_name=name)) if name else None

        if template_resource is not None:
            config = self._from_api_table_definition(template_resource.table)
            logger.warning(
                ("Table template %s already exists, with configuration %s. "
                 "This template will be overridden with configuration %s."),
                name, str(config), str(self))

        # TODO(manan): fix
        _id = global_state.client.table_api.create_table(
            table_def=self._to_api_table_definition(),
            name_alias=name,
            force_rename=True if name else False,
        )
        return f"table-{_id.split('-', maxsplit=1)[1]}"

    @classmethod
    def load(cls, table_id_or_template: str) -> 'Table':
        r"""Loads a table from either a table ID or a named template. Returns a
        :class:`Table` object that contains the loaded table along with its
        columns, etc.
        """
        api = global_state.client.table_api
        res = api.get_table_if_exists(table_id_or_template)
        if not res:
            raise ValueError(f"Table {table_id_or_template} was not found.")
        out = cls._from_api_table_definition(res.table)
        return out

    # Class properties ########################################################

    def __hash__(self) -> int:
        return hash(
            tuple(self.columns +
                  [self.primary_key, self.time_column, self.end_time_column]))

    def __contains__(self, name: str) -> bool:
        return self.has_column(name)

    def __getitem__(self, name: str) -> Column:
        return self.column(name)

    def __delitem__(self, name: str) -> None:
        self.remove_column(name)

    def __repr__(self) -> str:
        col_names = str(list(self._columns.keys())).replace("'", "")
        pkey_name = (self.primary_key.name
                     if self.primary_key is not None else "None")
        t_name = (self.time_column.name
                  if self.time_column is not None else "None")
        et_name = (self.end_time_column.name
                   if self.end_time_column is not None else "None")
        return (f'{self.__class__.__name__}(\n'
                f'  source_name={self.source_name},\n'
                f'  data_source={self.source_table.connector.name},\n'
                f'  columns={col_names},\n'
                f'  primary_key={pkey_name},\n'
                f'  time_column={t_name},\n'
                f'  end_time_column={et_name},\n'
                f')')
