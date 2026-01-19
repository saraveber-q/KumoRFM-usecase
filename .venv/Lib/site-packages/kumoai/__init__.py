import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging

from kumoapi.typing import Dtype, Stype

from kumoai.client.client import KumoClient
from kumoai._logging import initialize_logging, _ENV_KUMO_LOG
from kumoai._singleton import Singleton
from kumoai.futures import create_future, initialize_event_loop
from kumoai.spcs import (
    _get_active_session,
    _get_spcs_token,
    _run_refresh_spcs_token,
)

initialize_logging()
initialize_event_loop()


@dataclass
class GlobalState(metaclass=Singleton):
    r"""Global storage of the state needed to create a Kumo client object. A
    singleton so its initialized state can be referenced elsewhere for free.
    """

    # NOTE fork semantics: CoW on Linux, and re-execed on Windows. So this will
    # likely not work on Windows unless we have special handling for the shared
    # state:
    _url: Optional[str] = None
    _api_key: Optional[str] = None
    _snowflake_credentials: Optional[Dict[str, Any]] = None
    _spcs_token: Optional[str] = None
    _snowpark_session: Optional[Any] = None

    thread_local: threading.local = threading.local()

    def clear(self) -> None:
        if hasattr(self.thread_local, '_client'):
            del self.thread_local._client
        self._url = None
        self._api_key = None
        self._snowflake_credentials = None
        self._spcs_token = None

    def set_spcs_token(self, spcs_token: str) -> None:
        # Set the spcs token in the global state. This will be picked up the
        # next time client() is accessed.
        self._spcs_token = spcs_token

    @property
    def initialized(self) -> bool:
        return self._url is not None and (
            self._api_key is not None or self._snowflake_credentials
            is not None or self._snowpark_session is not None)

    @property
    def client(self) -> KumoClient:
        r"""Accessor for the Kumo client. Note that clients are stored as
        thread-local variables as the requests Session library is not
        guaranteed to be thread-safe.

        For more information, see https://github.com/psf/requests/issues/1871.
        """
        if self._url is None or (self._api_key is None
                                 and self._spcs_token is None
                                 and self._snowpark_session is None):
            raise ValueError(
                "Client creation or authentication failed; please re-create "
                "your client before proceeding.")

        if hasattr(self.thread_local, '_client'):
            # Set the spcs token in the client to ensure it has the latest.
            self.thread_local._client.set_spcs_token(self._spcs_token)
            return self.thread_local._client

        client = KumoClient(self._url, self._api_key, self._spcs_token)
        self.thread_local._client = client
        return client

    @property
    def is_spcs(self) -> bool:
        return (self._api_key is None
                and (self._snowflake_credentials is not None
                     or self._snowpark_session is not None))


global_state: GlobalState = GlobalState()


def init(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    snowflake_credentials: Optional[Dict[str, str]] = None,
    snowflake_application: Optional[str] = None,
    log_level: str = "INFO",
) -> None:
    r"""Initializes and authenticates the API key against the Kumo service.
    Successful authentication is required to use the SDK.

    Example:
        >>> import kumoai
        >>> kumoai.init(url="<api_url>", api_key="<api_key>")  # doctest: +SKIP

    Args:
        url: The Kumo API endpoint. Can also be provided via the
            ``KUMO_API_ENDPOINT`` envronment variable. Will be inferred from
            the provided API key, if not provided.
        api_key: The Kumo API key. Can also be provided via the
            ``KUMO_API_KEY`` environment variable.
        snowflake_credentials: The Snowflake credentials to authenticate
            against the Kumo service. The dictionary should contain the keys
            ``"user"``, ``"password"``, and ``"account"``. This should only be
            provided for SPCS.
        snowflake_application: The Snowflake application.
        log_level: The logging level that Kumo operates under. Defaults to
            INFO; for more information, please see
            :class:`~kumoai.set_log_level`. Can also be set with the
            environment variable ``KUMOAI_LOG``.
    """  # noqa
    # Avoid mutations to the global state after it is set:
    if global_state.initialized:
        print(
            "Client has already been created. To re-initialize Kumo, please "
            "start a new interpreter. No changes will be made to the current "
            "session.")
        return

    set_log_level(os.getenv(_ENV_KUMO_LOG, log_level))

    # Get API key:
    api_key = api_key or os.getenv("KUMO_API_KEY")

    snowpark_session = None
    if snowflake_application:
        if url is not None:
            raise ValueError(
                "Client creation failed: both snowflake_application and url "
                "are specified. If running from a snowflake notebook, specify"
                "only snowflake_application.")
        snowpark_session = _get_active_session()
        if not snowpark_session:
            raise ValueError(
                "Client creation failed: snowflake_application is specified "
                "without an active snowpark session. If running outside "
                "a snowflake notebook, specify a URL and credentials.")
        description = snowpark_session.sql(
            f"DESCRIBE SERVICE {snowflake_application}."
            "USER_SCHEMA.KUMO_SERVICE").collect()[0]
        url = f"http://{description.dns_name}:8888/public_api"

    if api_key is None and not snowflake_application:
        if snowflake_credentials is None:
            raise ValueError(
                "Client creation failed: Neither API key nor snowflake "
                "credentials provided. Please either set the 'KUMO_API_KEY' "
                "or explicitly call `kumoai.init(...)`.")
        if (set(snowflake_credentials.keys())
                != {'user', 'password', 'account'}):
            raise ValueError(
                f"Provided credentials should be a dictionary with keys "
                f"'user', 'password', and 'account'. Only "
                f"{set(snowflake_credentials.keys())} were provided.")

    # Get or infer URL:
    url = url or os.getenv("KUMO_API_ENDPOINT")
    try:
        if api_key:
            url = url or f"http://{api_key.split(':')[0]}.kumoai.cloud/api"
    except KeyError:
        pass
    if url is None:
        raise ValueError(
            "Client creation failed: endpoint URL not provided. Please "
            "either set the 'KUMO_API_ENDPOINT' environment variable or "
            "explicitly call `kumoai.init(...)`.")

    # Assign global state after verification that client can be created and
    # authenticated successfully:
    spcs_token = _get_spcs_token(
        snowflake_credentials
    ) if not api_key and snowflake_credentials else None
    client = KumoClient(url=url, api_key=api_key, spcs_token=spcs_token)
    client.authenticate()
    global_state._url = client._url
    global_state._api_key = client._api_key
    global_state._snowflake_credentials = snowflake_credentials
    global_state._spcs_token = client._spcs_token
    global_state._snowpark_session = snowpark_session

    if not api_key and snowflake_credentials:
        # Refresh token every 10 minutes (expires in 1 hour):
        create_future(_run_refresh_spcs_token(minutes=10))

    logger = logging.getLogger('kumoai')
    log_level = logging.getLevelName(logger.getEffectiveLevel())

    logger.info(
        f"Successfully initialized the Kumo SDK (version {__version__}) "
        f"against deployment {url}, with "
        f"log level {log_level}.")


def set_log_level(level: str) -> None:
    r"""Sets the Kumo logging level, which defines the amount of output that
    methods produce.

    Example:
        >>> import kumoai
        >>> kumoai.set_log_level("INFO")  # doctest: +SKIP

    Args:
        level: the logging level. Can be one of (in order of lowest to highest
            log output) :obj:`DEBUG`, :obj:`INFO`, :obj:`WARNING`,
            :obj:`ERROR`, :obj:`FATAL`, :obj:`CRITICAL`.
    """
    # logging library will ensure `level` is a valid string, and raise a
    # warning if not:
    logging.getLogger('kumoai').setLevel(level)


# Try to initialize purely with environment variables:
if ("pytest" not in sys.modules and "KUMO_API_KEY" in os.environ
        and "KUMO_API_ENDPOINT" in os.environ):
    init()

import kumoai.connector  # noqa
import kumoai.encoder  # noqa
import kumoai.graph  # noqa
import kumoai.pquery  # noqa
import kumoai.trainer  # noqa
import kumoai.utils  # noqa
import kumoai.databricks  # noqa

from kumoai.connector import (  # noqa
    SourceTable, SourceTableFuture, S3Connector, SnowflakeConnector,
    DatabricksConnector, BigQueryConnector, FileUploadConnector, GlueConnector)
from kumoai.graph import Column, Edge, Graph, Table  # noqa
from kumoai.pquery import (  # noqa
    PredictionTableGenerationPlan, PredictiveQuery,
    TrainingTableGenerationPlan, TrainingTable, TrainingTableJob,
    PredictionTable, PredictionTableJob)
from kumoai.trainer import (  # noqa
    ModelPlan, Trainer, TrainingJobResult, TrainingJob,
    BatchPredictionJobResult, BatchPredictionJob)
from kumoai._version import __version__  # noqa

__all__ = [
    'Dtype',
    'Stype',
    'SourceTable',
    'SourceTableFuture',
    'S3Connector',
    'SnowflakeConnector',
    'DatabricksConnector',
    'BigQueryConnector',
    'FileUploadConnector',
    'GlueConnector',
    'Column',
    'Table',
    'Graph',
    'Edge',
    'PredictiveQuery',
    'TrainingTable',
    'TrainingTableJob',
    'TrainingTableGenerationPlan',
    'PredictionTable',
    'PredictionTableJob',
    'PredictionTableGenerationPlan',
    'Trainer',
    'TrainingJobResult',
    'TrainingJob',
    'BatchPredictionJobResult',
    'BatchPredictionJob',
    'ModelPlan',
    '__version__',
]


def in_snowflake_notebook() -> bool:
    try:
        from snowflake.snowpark.context import get_active_session
        import streamlit  # noqa: F401
        get_active_session()
        return True
    except Exception:
        return False


def in_notebook() -> bool:
    if in_snowflake_notebook():
        return True
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if 'google.colab' in str(shell.__class__):
            return True
        return shell.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False
