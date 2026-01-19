import os
from typing import Dict, List, Optional

from kumoapi.data_source import (
    CreateConnectorArgs,
    DatabricksConnectorResourceConfig,
    DatabricksCredentials,
    DataSourceType,
)
from kumoapi.source_table import DatabricksSourceTableRequest
from typing_extensions import Self, override

from kumoai import global_state
from kumoai.connector import Connector

_ENV_DATABRICKS_CLIENT_ID = 'DATABRICKS_CLIENT_ID'
_ENV_DATABRICKS_CLIENT_SECRET = 'DATABRICKS_CLIENT_SECRET'
_ENV_DATABRICKS_TOKEN = 'DATABRICKS_TOKEN'


class DatabricksConnector(Connector):
    r"""Establishes a connection to a
    `Databricks <https://www.databricks.com/>`_ database.

    Authentication requires passing either a client ID and client secret, or a
    personal access token, to the connector, either via environment variables
    (``DATABRICKS_CLIENT_ID`` and ``DATABRICKS_CLIENT_SECRET``, or
    ``DATABRICKS_TOKEN``), or via keys in the credentials dictionary
    (``client_id`` and ``client_secret``, or ``token``).

    .. code-block:: python

        import kumoai

        # Either pass `credentials=dict(client_id=..., client_secret=...,
        # token=...) or set the 'DATABRICKS_CLIENT_ID' and
        # 'DATABRICKS_CLIENT_SECRET' (or 'DATABRICKS_TOKEN') environment
        # variables:
        connector = kumoai.connector.DatabricksConnector(
            name="<connector_name>",
            host="<databricks_host_name>",
            cluster_id="<databricks_cluster_id>",
            warehouse_id="<databricks_warehouse_id>",
            catalog="<databricks_catalog_name>",
            credentials=credentials,
        )

        # List all tables:
        print(connector.table_names())

        # Check whether a table is present:
        assert "articles" in connector

        # Fetch a source table (both approaches are equivalent):
        source_table = connector["articles"]
        source_table = connector.table("articles")

    Args:
        name: The name of the connector.
        host: The host name.
        cluster_id: The cluster ID of this warehouse.
        warehouse_id: The warehouse ID of this warehous.
        catalog: The name of the Databricks catalog.
        credentials: The client ID, client secret, and personal access token
            that correspond to this Databricks account.
    """
    def __init__(
            self,
            name: str,
            host: str,
            cluster_id: str,
            warehouse_id: str,
            catalog: str,
            credentials: Optional[Dict[str, str]] = None,
            _bypass_creation: bool = False,  # INTERNAL ONLY.
    ):
        super().__init__()

        self._name = name
        self.host = host
        self.cluster_id = cluster_id
        self.warehouse_id = warehouse_id
        self.catalog = catalog

        if _bypass_creation:
            # TODO(manan, siyang): validate that this connector actually exists
            # in the REST DB:
            return

        # Fully specify credentials, create Kumo connector:
        credentials = credentials or {}
        credentials_args = {
            "client_id":
            credentials.get("client_id", os.getenv(_ENV_DATABRICKS_CLIENT_ID)),
            "client_secret":
            credentials.get("client_secret",
                            os.getenv(_ENV_DATABRICKS_CLIENT_SECRET)),
            "token":
            credentials.get("token", os.getenv(_ENV_DATABRICKS_TOKEN))
        }

        has_pat = credentials_args["token"] is not None
        has_client_id_secret = (credentials_args["client_id"] is not None and
                                credentials_args["client_secret"] is not None)

        if has_pat and has_client_id_secret:
            raise ValueError(
                "Please pass only one of a (Databricks client ID and "
                "Databricks client secret) or a (Databricks PAT).")
        elif not (has_pat or has_client_id_secret):
            raise ValueError(
                f"Please pass valid credentials to create a Databricks "
                f"connector. You can do so either via the 'credentials' "
                f"argument or the {_ENV_DATABRICKS_CLIENT_ID} and "
                f"{_ENV_DATABRICKS_CLIENT_SECRET}, or "
                f"{_ENV_DATABRICKS_TOKEN} environment variables.")

        self._create_connector(credentials_args)  # type: ignore

    @classmethod
    def get_by_name(cls, name: str) -> Self:
        r"""Returns an instance of a named Databricks Connector, including
        those created in the Kumo UI.

        Args:
            name: The name of the existing connector.

        Example:
            >>> import kumoai
            >>> connector = kumoai.DatabricksConnector.get_by_name("name")  # doctest: +SKIP # noqa: E501
        """
        api = global_state.client.connector_api
        resp = api.get(name)
        if resp is None:
            raise ValueError(
                f"There does not exist an existing stored connector with name "
                f"{name}.")
        config = resp.config
        assert isinstance(config, DatabricksConnectorResourceConfig)
        return cls(
            name=config.name,
            host=config.host,
            cluster_id=config.cluster_id,
            warehouse_id=config.warehouse_id,
            catalog=config.catalog,
            credentials=None,
            _bypass_creation=True,
        )

    @override
    @property
    def name(self) -> str:
        r"""Returns the name of this connector."""
        return self._name

    @override
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.DATABRICKS

    @override
    def _source_table_request(
        self,
        table_names: List[str],
    ) -> DatabricksSourceTableRequest:
        return DatabricksSourceTableRequest(
            connector_id=self.name,
            table_names=table_names,
        )

    def _create_connector(self, credentials: Dict[str, str]) -> None:
        r"""Creates and persists a Databricks connector in the REST DB.
        Currently only intended for internal use.

        Args:
            credentials: Fully-specified credentials containing the username
                and password for the Databricks connector.

        Raises:
            RuntimeError: if connector creation failed
        """
        credentials = DatabricksCredentials(
            client_id=credentials["client_id"] or '',
            client_secret=credentials["client_secret"] or '',
            pat=credentials["token"] or '',
        )
        args = CreateConnectorArgs(
            config=DatabricksConnectorResourceConfig(
                name=self.name,
                host=self.host,
                cluster_id=self.cluster_id,
                warehouse_id=self.warehouse_id,
                catalog=self.catalog,
            ),
            credentials=credentials,
        )
        global_state.client.connector_api.create_if_not_exist(args)

    def _delete_connector(self) -> None:
        r"""Deletes a connector in the REST DB. Only intended for internal
        use.
        """
        global_state.client.connector_api.delete_if_exists(self.name)

    # Class properties ########################################################

    @override
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'(name=\"{self.name}\", host=\"{self.host}\", '
                f'cluster_id=\"{self.cluster_id}\", '
                f'warehouse_id=\"{self.warehouse_id}\", '
                f'catalog=\"{self.catalog}\")')
