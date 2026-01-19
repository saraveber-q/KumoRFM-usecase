import logging
from typing import List

from kumoapi.data_source import (
    CreateConnectorArgs,
    DataSourceType,
    GlueConnectorResourceConfig,
)
from kumoapi.source_table import GlueSourceTableRequest
from typing_extensions import Self, override

from kumoai import global_state
from kumoai.connector import Connector

logger = logging.getLogger(__name__)

_DEFAULT_NAME = 'glue_connector'


class GlueConnector(Connector):
    r"""Defines a connector to a table stored in AWS Glue catalog. Currently,
    only supports tables in partitioned parquet format. Authenticated via IAM
    permissions on Glue catalog and data location in S3.

    .. code-block:: python

        import kumoai
        connector = kumoai.GlueConnector(database="...", region="...", account="...")

        # List all tables:
        print(connector.table_names())  # Returns: ['articles', 'customers', 'users']

        # Check whether a table is present:
        assert "articles" in connector

        # Fetch a source table (both approaches are equivalent):
        source_table = connector["articles"]
        source_table = connector.table("articles")

    Args:
        account: The account of the Glue catalog.
        region: The region of the Glue catalog.
        database: The name of the database in the Glue Catalog

    """  # noqa

    def __init__(
            self,
            name: str,
            account: str,
            region: str,
            database: str,
            _bypass_creation: bool = False,  # INTERNAL ONLY.
    ) -> None:
        self._name = name
        self.account = account
        self.region = region
        self.database = database
        if global_state.is_spcs and database is not None:
            raise ValueError(
                "Glue connectors are not supported when running Kumo in "
                "Snowpark container services. Please use a Snowflake "
                "connector instead.")

        if _bypass_creation:
            return

        self._create_connector()

    @override
    @property
    def name(self) -> str:
        r"""Returns the name of this connector."""
        return self._name

    @override
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.GLUE

    @override
    def _source_table_request(
        self,
        table_names: List[str],
    ) -> GlueSourceTableRequest:
        return GlueSourceTableRequest(
            connector_id=self.name,
            table_names=table_names,
        )

    def _create_connector(self) -> None:
        r"""Creates and persists a Glue connector in the REST DB.
        Currently only intended for internal use.


        Raises:
            RuntimeError: if connector creation failed
        """
        args = CreateConnectorArgs(
            config=GlueConnectorResourceConfig(
                name=self.name,
                account=self.account,
                region=self.region,
                database=self.database,
            ), )
        global_state.client.connector_api.create_if_not_exist(args)

    def _delete_connector(self) -> None:
        r"""Deletes a connector in the REST DB. Only intended for internal
        use.
        """
        global_state.client.connector_api.delete_if_exists(self.name)

    # Class properties ########################################################

    @classmethod
    def get_by_name(cls, name: str) -> Self:
        r"""Returns an instance of a named Glue Connector, including those created
        in the Kumo UI.

        Args:
            name: The name of the existing connector.

        Example:
            >>> import kumoai
            >>> connector = kumoai.GlueConnector.get_by_name("name")  # doctest: +SKIP # noqa: E501
        """
        api = global_state.client.connector_api
        resp = api.get(name)
        if resp is None:
            raise ValueError(
                f"There does not exist an existing stored connector with name "
                f"{name}.")
        config = resp.config
        assert isinstance(config, GlueConnectorResourceConfig)
        return cls(
            name=config.name,
            account=config.account,
            region=config.region,
            database=config.database,
            _bypass_creation=True,
        )

    @override
    def __repr__(self) -> str:
        account_name = f"\"{self.account}\"" if self.account else "None"
        region_name = f"\"{self.region}\"" if self.region else "None"
        database_name = f"\"{self.database}\"" if self.database else "None"
        return (f'{self.__class__.__name__}(account={account_name}, '
                f'region={region_name}, database={database_name})')
