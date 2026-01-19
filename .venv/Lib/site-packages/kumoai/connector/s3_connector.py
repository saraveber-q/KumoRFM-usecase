import logging
from typing import List, Optional

from kumoapi.data_source import DataSourceType, FileConnectorResourceConfig
from kumoapi.source_table import (
    S3SourceTableRequest,
    SourceTableConfigRequest,
    SourceTableConfigResponse,
    SourceTableListRequest,
    SourceTableValidateRequest,
    SourceTableValidateResponse,
)
from typing_extensions import Self, override

from kumoai import global_state
from kumoai.connector import Connector
from kumoai.connector.source_table import SourceTable

logger = logging.getLogger(__name__)

_DEFAULT_NAME = 's3_connector'


class S3Connector(Connector):
    r"""Defines a connector to a table stored as a file (or partitioned
    set of files) on the Amazon `S3 <https://aws.amazon.com/s3/>`__ object
    store. Any table behind an S3 bucket accessible by the shared external IAM
    role can be accessed through this connector.

    .. code-block:: python

        import kumoai
        connector = kumoai.S3Connector(root_dir="s3://...")  # an S3 path.

        # List all tables:
        print(connector.table_names())  # Returns: ['articles', 'customers', 'users']

        # Check whether a table is present:
        assert "articles" in connector

        # Fetch a source table (both approaches are equivalent):
        source_table = connector["articles"]
        source_table = connector.table("articles")

    Args:
        root_dir: The root directory of this connector. If provided, the root
            directory is used as a prefix for tables in this connector. If not
            provided, all tables must be specified by their full S3 paths.
    """  # noqa

    def __init__(
        self,
        root_dir: Optional[str] = None,
        _connector_id: Optional[str] = None,
    ) -> None:
        if _connector_id is not None:
            # UI S3Connector, named:
            self._connector_id = _connector_id
            self.root_dir = None
            return

        self._connector_id = _DEFAULT_NAME
        if root_dir is not None:
            # Remove trailing / to be consistent with boto s3
            root_dir = root_dir.rstrip('/')
        self.root_dir = root_dir
        if global_state.is_spcs and root_dir is not None \
                and root_dir.startswith('s3://'):
            raise ValueError(
                "S3 connectors are not supported when running Kumo in "
                "Snowpark container services. Please use a Snowflake "
                "connector instead.")

    @override
    @property
    def name(self) -> str:
        r"""Not supported by :class:`S3Connector`; returns an internal
        specifier.
        """
        return self._connector_id

    @override
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.S3

    @override
    def _source_table_request(
        self,
        table_names: List[str],
    ) -> S3SourceTableRequest:
        root_dir = self.root_dir
        if not root_dir and self.name == _DEFAULT_NAME:
            # Handle None root directories (table name is a path):
            table_path = S3URI(table_names[0]).validate()
            root_dir = table_path.root_dir
            for i, v in enumerate(table_names):
                uri = S3URI(v)
                if uri.root_dir != root_dir:
                    # TODO(manan): fix
                    raise ValueError(
                        f"Please ensure that all of your tables are behind "
                        f"the same root directory ({root_dir}).")
                table_names[i] = uri.object_name

        connector_id = self.name if self.name != _DEFAULT_NAME else None
        root_dir = root_dir if self.name == _DEFAULT_NAME else ""

        # TODO(manan): file type?
        return S3SourceTableRequest(
            s3_root_dir=root_dir,
            connector_id=connector_id,
            table_names=table_names,
        )

    @override
    def table(self, name: str) -> SourceTable:
        r"""Returns a :class:`~kumoai.connector.SourceTable` object
        corresponding to a source table on Amazon S3.

        Args:
            name: The name of the table on S3. If :obj:`root_dir` is provided,
                the path will be specified as :obj:`root_dir/name`. If
                :obj:`root_dir` is not provided, the name should be the full
                path (e.g. starting with ``s3://``).

        Raises:
            :class:`ValueError`: if ``name`` does not exist in the backing
                connector.
        """
        # NOTE only overridden for documentation purposes.
        return super().table(name)

    @override
    def _list_tables(self) -> List[str]:
        connector_id = self.name if self.name != _DEFAULT_NAME else None
        root_dir = self.root_dir if self.name == _DEFAULT_NAME else None
        if root_dir is None and connector_id is None:
            raise ValueError(
                "Listing tables without a specified root directory is not "
                "supported. Please specify a root directory to continue; "
                "alternatively, please access individual tables with their "
                "full S3 paths.")

        req = SourceTableListRequest(connector_id=connector_id,
                                     root_dir=root_dir,
                                     source_type=DataSourceType.S3)
        return global_state.client.source_table_api.list_tables(req)

    @override
    def _get_table_config(self, table_name: str) -> SourceTableConfigResponse:
        root_dir = self.root_dir
        if not root_dir and self.name == _DEFAULT_NAME:
            # Handle None root directories (table name is a path):
            table_path = S3URI(table_name).validate()
            root_dir = table_path.root_dir
            table_name = table_path.object_name

        connector_id = self.name if self.name != _DEFAULT_NAME else None
        root_dir = root_dir if self.name == _DEFAULT_NAME else None

        req = SourceTableConfigRequest(
            connector_id=connector_id,
            root_dir=root_dir,
            table_name=table_name,
            source_type=self.source_type,
        )
        return global_state.client.source_table_api.get_table_config(req)

    @override
    def _validate_table(self, table_name: str) -> SourceTableValidateResponse:
        if table_name in self._validated_tables:
            return SourceTableValidateResponse(is_valid=True, msg='')

        if self.name == _DEFAULT_NAME:
            # For S3 connector without name, pass root_dir to validate table -
            # need to infer from table name if root_dir is not provided
            if self.root_dir:
                req = SourceTableValidateRequest(
                    connector_id=None,
                    table_name=table_name,
                    source_type=self.source_type,
                    root_dir=self.root_dir,
                )
            else:
                req_root_dir = '/'.join(table_name.split('/')[:-1])
                req_table_name = table_name.split('/')[-1]

                req = SourceTableValidateRequest(
                    connector_id=None,
                    table_name=req_table_name,
                    source_type=self.source_type,
                    root_dir=req_root_dir,
                )
        else:
            req = SourceTableValidateRequest(connector_id=self.name,
                                             table_name=table_name,
                                             source_type=self.source_type)

        ret = global_state.client.source_table_api.validate_table(req)
        # Cache the result for the whole session.
        if ret.is_valid:
            self._validated_tables.add(table_name)
        return ret

    # Class properties ########################################################

    @classmethod
    def get_by_name(cls, name: str) -> Self:
        r"""Returns an instance of a named S3 Connector, created in the Kumo UI.

        .. note::
            Named S3 connectors are read-only: if you would like to modify the
            root directory, please do so from the UI.

        Args:
            name: The name of the existing connector.

        Example:
            >>> import kumoai
            >>> connector = kumoai.S3Connector.get_by_name("name")  # doctest: +SKIP # noqa: E501
        """
        api = global_state.client.connector_api
        resp = api.get(name)
        if resp is None:
            raise ValueError(
                f"There does not exist an existing stored connector with name "
                f"{name}.")
        config = resp.config
        assert isinstance(config, FileConnectorResourceConfig)
        return cls(
            root_dir=None,
            _connector_id=config.name,
        )

    def __repr__(self) -> str:
        if self.name != _DEFAULT_NAME:
            return f'{self.__class__.__name__}(name={self.name})'
        root_dir_name = f"\"{self.root_dir}\"" if self.root_dir else "None"
        return f'{self.__class__.__name__}(root_dir={root_dir_name})'


class S3URI:
    r"""A utility class to parse and navigate S3 URIs."""
    def __init__(self, uri: str):
        self.uri: str = uri
        if uri.endswith('/'):  # remove trailing slash
            self.uri = uri[:-1]

    @property
    def is_valid(self) -> bool:
        # TODO(zeyuan): For SPCS, the path can be a local filesystem path
        # For train/pred table.
        if global_state.is_spcs:
            return True
        # TODO(manan): implement more checks...
        return self.uri.startswith("s3://")

    def validate(self) -> Self:
        if not self.is_valid:
            raise ValueError(f"Path {self.uri} is not a valid S3 URI.")
        return self

    @property
    def root_dir(self) -> str:
        self.validate()
        return self.uri.rsplit('/', 1)[0]

    @property
    def object_name(self) -> str:
        self.validate()
        return self.uri.rsplit('/', 1)[1]

    # Class properties ########################################################

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'uri={self.uri}, valid={self.is_valid})')
