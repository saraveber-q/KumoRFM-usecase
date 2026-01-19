import asyncio
import os
from functools import reduce
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from snowflake.snowpark import DataFrame, Session


def _get_spcs_token(snowflake_credentials: Dict[str, str]) -> str:
    r"""Fetches a token to access a Kumo application deployed in Snowflake
    Snowpark Container Services (SPCS). This token is valid for 1 hour, after
    which the token must be re-generated.
    """
    # Create a request to the ingress endpoint with authz:
    active_session = _get_active_session()
    if active_session is not None:
        ctx = active_session.connection
    else:
        user = snowflake_credentials["user"]
        password = snowflake_credentials["password"]
        account = snowflake_credentials["account"]
        import snowflake.connector
        ctx = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            session_parameters={
                'PYTHON_CONNECTOR_QUERY_RESULT_FORMAT': 'json'
            },
        )

    # Obtain a session token:
    token_data = ctx._rest._token_request('ISSUE')
    token_extract = token_data['data']['sessionToken']
    return f'\"{token_extract}\"'


def _refresh_spcs_token() -> None:
    r"""Refreshes the SPCS token in global state to avoid expiration."""
    from kumoai import KumoClient, global_state
    if (not global_state.initialized
            or (not global_state._snowflake_credentials
                and not global_state._snowpark_session)):
        raise ValueError(
            "Please initialize the Kumo application with snowflake "
            "credentials before attempting to refresh this token.")
    spcs_token = _get_spcs_token(global_state._snowflake_credentials or {})

    # Verify token validity:
    assert global_state._url is not None
    client = KumoClient(
        url=global_state._url,
        api_key=global_state._api_key,
        spcs_token=spcs_token,
    )
    client.authenticate()

    # Update state:
    global_state.set_spcs_token(spcs_token)


async def _run_refresh_spcs_token(minutes: int) -> None:
    r"""Runs the SPCS token refresh loop every `minutes` minutes."""
    while True:
        await asyncio.sleep(minutes * 60)
        _refresh_spcs_token()


def _get_active_session() -> 'Optional[Session]':
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception:
        return None


def _get_session() -> 'Session':
    import snowflake.snowpark as snowpark

    from kumoai import global_state
    params = global_state._snowflake_credentials
    assert params is not None

    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")
    if not database or not schema:
        raise ValueError("Please set the SNOWFLAKE_DATABASE and "
                         "SNOWFLAKE_SCHEMA environment variables.")
    params['database'] = database
    params['schema'] = schema
    params['client_session_keep_alive'] = True

    return snowpark.Session.builder.configs(params).create()


def _remove_path(session: 'Session', stage_path: str, file_path: str) -> None:
    stage_prefix = '.'.join(stage_path.split('.')[:2])
    name_remove = '.'.join([stage_prefix, file_path])
    session.sql(f"REMOVE {name_remove}").collect()


def _parquet_to_df(path: str) -> 'DataFrame':
    r"""Reads parquet from the given path and returns a snowpark DataFrame."""
    session = _get_session()
    if not path.endswith(os.path.sep):
        path += os.path.sep
    file_list = session.sql(f"LIST {path}").collect()
    for file_row in file_list:
        if file_row.name.endswith('.parquet'):
            continue
        _remove_path(session, path, file_row.name)
    df = session.read.parquet(path)
    return df


def _parquet_dataset_to_df(paths: List[str]) -> 'DataFrame':
    r"""Reads parquet from the given paths and returns a snowpark DataFrame."""
    from snowflake.snowpark import DataFrame
    df_list = [_parquet_to_df(url) for url in paths]
    return reduce(DataFrame.union_all, df_list)
