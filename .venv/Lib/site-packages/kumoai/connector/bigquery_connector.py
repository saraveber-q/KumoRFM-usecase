import os
from typing import Dict, List, Optional

from kumoapi.data_source import (
    BigQueryConnectorResourceConfig,
    BigQueryCredentials,
    CreateConnectorArgs,
    DataSourceType,
)
from kumoapi.source_table import BigQuerySourceTableRequest
from typing_extensions import Self, override

from kumoai import global_state
from kumoai.connector import Connector

_ENV_BIGQUERY_PRIVATE_KEY_ID = 'BIGQUERY_PRIVATE_KEY_ID'
_ENV_BIGQUERY_PRIVATE_KEY = 'BIGQUERY_PRIVATE_KEY'
_ENV_BIGQUERY_CLIENT_ID = 'BIGQUERY_CLIENT_ID'
_ENV_BIGQUERY_CLIENT_EMAIL = 'BIGQUERY_CLIENT_EMAIL'
_ENV_BIGQUERY_TOKEN_URI = 'BIGQUERY_TOKEN_URI'
_ENV_BIGQUERY_AUTH_URI = 'BIGQUERY_AUTH_URI'


class BigQueryConnector(Connector):
    r"""Establishes a connection to a
    `BigQuery <https://cloud.google.com/bigquery>`_ database.

    Authentication requires passing a private key ID, private key string,
    client ID, client email, token URI, and authentication URI to the
    connector, either via environment variables
    (``BIGQUERY_PRIVATE_KEY_ID``, ``BIGQUERY_PRIVATE_KEY``,
    ``BIGQUERY_CLIENT_ID``, ``BIGQUERY_CLIENT_EMAIL``, ``BIGQUERY_TOKEN_URI``,
    ``BIGQUERY_AUTH_URI``), or via keys in the credentials dictionary
    (:obj:`private_key_id`, :obj:`private_key`, :obj:`client_id`,
    :obj:`client_email`, :obj:`token_uri`, :obj:`auth_uri`).

    .. code-block:: python

        import kumoai

        # Either pass `credentials=dict(private_key_id=..., private_key=...,
        # client_id=..., client_email=..., token_uri=..., auth_url=...)` or set
        # the aforementioned environment variables:
        connector = kumoai.BigQueryConnector(
            name="<connector_name>",
            project_id="<bigquery_project_id>",
            dataset_id="<bigquery_dataset_id>",
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
        project_id: The project ID to connect to.
        dataset_id: The dataset ID within the connected project.
        credentials: The private key ID, private key, client ID, client email,
            token URI, and auth URI that correspond to this BigQuery account.
    """
    def __init__(
            self,
            name: str,
            project_id: str,
            dataset_id: str,
            credentials: Optional[Dict[str, str]] = None,
            _bypass_creation: bool = False,  # INTERNAL ONLY.
    ):
        super().__init__()

        self._name = name
        self.project_id = project_id
        self.dataset_id = dataset_id

        if _bypass_creation:
            # TODO(manan, siyang): validate that this connector actually exists
            # in the REST DB:
            return

        # Fully specify credentials, create Kumo connector:
        credentials = credentials or {}
        credentials_args = {
            "private_key_id":
            credentials.get("private_key_id",
                            os.getenv(_ENV_BIGQUERY_PRIVATE_KEY_ID)),
            "private_key":
            credentials.get("private_key",
                            os.getenv(_ENV_BIGQUERY_PRIVATE_KEY)),
            "client_id":
            credentials.get("client_id", os.getenv(_ENV_BIGQUERY_CLIENT_ID)),
            "client_email":
            credentials.get("client_email",
                            os.getenv(_ENV_BIGQUERY_CLIENT_EMAIL)),
            "token_uri":
            credentials.get("token_uri", os.getenv(_ENV_BIGQUERY_TOKEN_URI)),
            "auth_uri":
            credentials.get("auth_uri", os.getenv(_ENV_BIGQUERY_AUTH_URI)),
        }

        self._create_connector(credentials_args)  # type: ignore

    @classmethod
    def get_by_name(cls, name: str) -> Self:
        r"""Returns an instance of a named BigQuery Connector, including
        those created in the Kumo UI.

        Args:
            name: The name of the existing connector.

        Example:
            >>> import kumoai
            >>> connector = kumoai.SnowflakeConnector.get_by_name("name")  # doctest: +SKIP # noqa: E501
        """
        api = global_state.client.connector_api
        resp = api.get(name)
        if resp is None:
            raise ValueError(
                f"There does not exist an existing stored connector with name "
                f"{name}.")
        config = resp.config
        assert isinstance(config, BigQueryConnectorResourceConfig)
        return cls(
            name=config.name,
            project_id=config.project_id,
            dataset_id=config.dataset_id,
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
        return DataSourceType.BIGQUERY

    @override
    def _source_table_request(
        self,
        table_names: List[str],
    ) -> BigQuerySourceTableRequest:
        return BigQuerySourceTableRequest(
            connector_id=self.name,
            table_names=table_names,
        )

    def _create_connector(self, credentials: Dict[str, str]) -> None:
        r"""Creates and persists a BigQuery connector in the REST DB.
        Currently only intended for internal use.

        Args:
            credentials: Fully-specified credentials containing the username
                and password for the BigQuery connector.

        Raises:
            RuntimeError: if connector creation failed
        """
        credentials = BigQueryCredentials(
            private_key_id=credentials["private_key_id"] or '',
            private_key=credentials["private_key"] or '',
            client_id=credentials["client_id"] or '',
            client_email=credentials["client_email"] or '',
            token_uri=credentials["token_uri"] or '',
            auth_uri=credentials["auth_uri"] or '',
        )
        args = CreateConnectorArgs(
            config=BigQueryConnectorResourceConfig(
                name=self.name,
                project_id=self.project_id,
                dataset_id=self.dataset_id,
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
                f'(name=\"{self.name}\",'
                f'project_id=\"{self.project_id}\", '
                f'dataset_id=\"{self.dataset_id}\")')
