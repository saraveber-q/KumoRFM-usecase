import os
from typing import Dict, List, Optional

from kumoapi.data_source import (
    CreateConnectorArgs,
    DataSourceType,
    KeyPair,
    SnowflakeConnectorResourceConfig,
    UsernamePassword,
)
from kumoapi.source_table import SnowflakeSourceTableRequest
from typing_extensions import Self, override

from kumoai import global_state
from kumoai.connector import Connector

_ENV_SNOWFLAKE_USER = 'SNOWFLAKE_USER'
_ENV_SNOWFLAKE_PASSWORD = 'SNOWFLAKE_PASSWORD'
_ENV_SNOWFLAKE_PRIVATE_KEY = 'SNOWFLAKE_PRIVATE_KEY'
_ENV_SNOWFLAKE_PRIVATE_KEY_PASSPHRASE = 'SNOWFLAKE_PRIVATE_KEY_PASSPHRASE'


class SnowflakeConnector(Connector):
    r"""Establishes a connection to a `Snowflake <https://www.snowflake.com/>`_
    database.

    Multiple methods of authentication are available. Username/password
    authentication is supported either via environment variables
    (``SNOWFLAKE_USER`` and ``SNOWFLAKE_PASSWORD``) or via keys in the
    credentials dictionary (:obj:`user` and :obj:`password`).

    .. note::
        Key-pair authentication is coming soon; please contact your Kumo POC if
        you need access.

    .. code-block:: python

        import kumoai

        # Either pass `credentials=dict(user=..., password=...)` or set the
        # 'SNOWFLAKE_USER' and 'SNOWFLAKE_PASSWORD' environment variables:
        connector = kumoai.SnowflakeConnector(
            name="<connector_name>",
            account="<snowflake_account_name>",
            database="<snowflake_database_name>"
            schema_name="<snowflake_schema_name>",
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
        account: The account name.
        warehouse: The name of the warehouse.
        database: The name of the database.
        schema_name: The name of the schema.
        credentials: The username and password corresponding to this Snowflake
            account, if not provided as environment variables.
    """
    def __init__(
            self,
            name: str,
            account: str,
            warehouse: str,
            database: str,
            schema_name: str,
            credentials: Optional[Dict[str, str]] = None,
            _bypass_creation: bool = False,  # INTERNAL ONLY.
    ):
        super().__init__()

        self._name = name
        self.account = account
        self.warehouse = warehouse

        # Snowflake DBs and schemas are all in upper-case:
        self.database = database.upper()
        self.schema_name = schema_name.upper()

        if _bypass_creation:
            # TODO(manan, siyang): validate that this connector actually exists
            # in the REST DB:
            return

        # Fully specify credentials, create Kumo connector:
        credentials = credentials or global_state._snowflake_credentials or {}

        credentials_args = {
            "user": credentials.get("user", os.getenv(_ENV_SNOWFLAKE_USER)),
        }
        password = credentials.get("password",
                                   os.getenv(_ENV_SNOWFLAKE_PASSWORD))
        private_key = credentials.get("private_key",
                                      os.getenv(_ENV_SNOWFLAKE_PRIVATE_KEY))
        private_key_passphrase = credentials.get(
            "private_key_passphrase",
            os.getenv(_ENV_SNOWFLAKE_PRIVATE_KEY_PASSPHRASE))

        if not password and not private_key:
            self._create_native_connector()
            return

        # Don't pass unused credential fields so that _create_connector can
        # decide which credential class (KeyPair or UsernamePassword) to use
        if private_key:
            credentials_args["private_key"] = private_key
            if private_key_passphrase:
                credentials_args[
                    "private_key_passphrase"] = private_key_passphrase
        else:
            credentials_args["password"] = password
        error_name = None
        error_var = None
        if credentials_args["user"] is None:
            error_name = "username"
            error_var = _ENV_SNOWFLAKE_USER
        elif password is None and private_key is None:
            error_name = "password or private key"
            error_var = f"{_ENV_SNOWFLAKE_PASSWORD} or " + \
                        f"{_ENV_SNOWFLAKE_PRIVATE_KEY}"
        if error_name is not None:
            raise ValueError(
                f"Please pass a valid {error_name} to create a Snowflake "
                f"connector. You can do so either via the 'credentials' "
                f"argument or the {error_var} environment variable.")

        self._create_connector(credentials_args)  # type: ignore

    @classmethod
    def get_by_name(cls, name: str) -> Self:
        r"""Returns an instance of a named Snowflake Connector, including
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
        assert isinstance(config, SnowflakeConnectorResourceConfig)
        return cls(
            name=config.name,
            account=config.account,
            warehouse=config.warehouse,
            database=config.database,
            schema_name=config.schema_name,
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
        return DataSourceType.SNOWFLAKE

    @override
    def _source_table_request(
        self,
        table_names: List[str],
    ) -> SnowflakeSourceTableRequest:
        return SnowflakeSourceTableRequest(
            connector_id=self.name,
            table_names=table_names,
        )

    def _create_connector(self, credentials: Dict[str, str]) -> None:
        r"""Creates and persists a Snowflake connector in the REST DB.
        Currently only intended for internal use.

        Args:
            credentials: Fully-specified credentials containing the username
                and password for the Snowflake connector.

        Raises:
            RuntimeError: if connector creation failed
        """
        # TODO(manan, siyang): consider avoiding connector persistence in the
        # REST DB, instead moving towards global connectors. For now, to get
        # a Snowflake experience working smoothly, using the old interface:
        if credentials.get("password") is not None:
            credentials = UsernamePassword(
                username=credentials["user"],
                password=credentials["password"],
            )
        else:
            credentials = KeyPair(
                user=credentials["user"],
                private_key=credentials["private_key"],
                private_key_passphrase=credentials.get(
                    "private_key_passphrase"),
            )
        args = CreateConnectorArgs(
            config=SnowflakeConnectorResourceConfig(
                name=self.name,
                account=self.account,
                warehouse=self.warehouse,
                database=self.database,
                schema_name=self.schema_name,
            ),
            credentials=credentials,
        )
        global_state.client.connector_api.create_if_not_exist(args)

    def _create_native_connector(self) -> None:
        args = CreateConnectorArgs(
            config=SnowflakeConnectorResourceConfig(
                name=self.name,
                account=self.account,
                warehouse=self.warehouse,
                database=self.database,
                schema_name=self.schema_name,
            ),
            credentials=None,
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
                f'(account=\"{self.account}\", database=\"{self.database}\", '
                f'schema=\"{self.schema_name}\")')
