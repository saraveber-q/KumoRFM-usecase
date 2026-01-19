from abc import ABC, abstractmethod
from typing import List, Set, Union

from kumoapi.data_source import DataSourceType
from kumoapi.source_table import (
    BigQuerySourceTableRequest,
    DatabricksSourceTableRequest,
    S3SourceTableRequest,
    SnowflakeSourceTableRequest,
    SourceTableConfigRequest,
    SourceTableConfigResponse,
    SourceTableDataRequest,
    SourceTableDataResponse,
    SourceTableListRequest,
    SourceTableListResponse,
    SourceTableValidateRequest,
    SourceTableValidateResponse,
)

from kumoai import global_state
from kumoai.connector.source_table import SourceTable
from kumoai.exceptions import HTTPException


class Connector(ABC):
    r"""A connector to a backing data source, that can be used to create Kumo
    tables.
    """

    _validated_tables: Set[str] = set()

    # Metadata ################################################################

    @property
    @abstractmethod
    def source_type(self) -> DataSourceType:
        r"""Returns the data source type accessible by this connector."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        r"""Returns the name of the connector.

        .. note::
            If the connector does not support naming, the name refers to an
            internal specifier.
        """
        raise NotImplementedError()

    @abstractmethod
    def _source_table_request(
        self,
        table_names: List[str],
    ) -> Union[S3SourceTableRequest, BigQuerySourceTableRequest,
               DatabricksSourceTableRequest, SnowflakeSourceTableRequest]:
        raise NotImplementedError()

    # Tables ##################################################################

    def table_names(self) -> List[str]:
        r"""Returns a list of table names accessible through this connector."""
        return self._list_tables().table_names

    def has_table(self, name: str) -> bool:
        r"""Returns :obj:`True` if the table exists in this connector,
        :obj:`False` otherwise.

        Args:
            name: The table name.
        """
        try:
            resp = self._validate_table(name)
            return resp.is_valid
        except HTTPException:
            # In case of HTTPException, Kumo backend doesn't have api
            # implemented and we skip check by returns True.
            return True

    def table(self, name: str) -> SourceTable:
        r"""Returns a :class:`~kumoai.connector.SourceTable` object
        corresponding to a source table behind this connector. A source table
        is a view into the raw data of table :obj:`name`. To use a source
        table in Kumo, you will need to construct a
        :class:`~kumoai.graph.Table` from the source table.

        Args:
            name: The table name.

        Raises:
            :class:`ValueError`: if :obj:`name` does not exist in the backing
                connector.
        """
        if not self.has_table(name):
            raise ValueError(f"The table '{name}' does not exist in {self}. "
                             f"Please check the existence of the source data.")

        return SourceTable(name=name, connector=self)

    def _validate_table(self, table_name: str) -> SourceTableValidateResponse:
        if table_name in self._validated_tables:
            return SourceTableValidateResponse(is_valid=True, msg='')

        req = SourceTableValidateRequest(
            connector_id=self.name,
            table_name=table_name,
            source_type=self.source_type,
        )
        ret = global_state.client.source_table_api.validate_table(req)

        # Cache the result for the whole session.
        if ret.is_valid:
            self._validated_tables.add(table_name)
        return ret

    def _list_tables(self) -> SourceTableListResponse:
        req = SourceTableListRequest(connector_id=self.name,
                                     source_type=self.source_type)
        return global_state.client.source_table_api.list_tables(req)

    def _get_table_data(
        self,
        table_names: List[str],
        sample_rows: int,
    ) -> SourceTableDataResponse:
        req = SourceTableDataRequest(
            source_table_request=self._source_table_request(table_names),
            sample_rows=sample_rows,
        )
        return global_state.client.source_table_api.get_table_data(req)

    def _get_table_config(self, table_name: str) -> SourceTableConfigResponse:
        # TODO(manan): rest backend for this is a bit broken, it never returns
        # directories...
        req = SourceTableConfigRequest(connector_id=self.name,
                                       table_name=table_name,
                                       source_type=self.source_type)
        return global_state.client.source_table_api.get_table_config(req)

    # Class properties ########################################################

    def __hash__(self) -> int:
        return hash(self.__dict__)

    def __contains__(self, name: str) -> bool:
        return self.has_table(name)

    def __getitem__(self, name: str) -> SourceTable:
        return self.table(name)

    def __repr__(self) -> str:
        # TODO(manan): class-overrideable metadata?
        return f'{self.__class__.__name__}()'
