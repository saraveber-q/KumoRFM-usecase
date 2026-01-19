import asyncio
import concurrent
import logging
from io import StringIO
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    overload,
)

import pandas as pd
from kumoapi.jobs import JobStatus
from kumoapi.source_table import (
    DataSourceType,
    LLMRequest,
    SourceColumn,
    SourceTableDataResponse,
    SourceTableType,
)
from kumoapi.table import TableDefinition
from typing_extensions import override

from kumoai import global_state
from kumoai.client.jobs import LLMJobId
from kumoai.exceptions import HTTPException
from kumoai.futures import KumoFuture, create_future

if TYPE_CHECKING:
    from kumoai.connector import Connector

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 20.0


class SourceTable:
    r"""A source table is a reference to a table stored behind a backing
    :class:`~kumoai.connector.Connector`. It can be used to examine basic
    information about raw data connected to Kumo, including a sample of the
    table's rows, basic statistics, and column data type information.

    Once you are ready to use a table as part of a
    :class:`~kumoai.graph.Graph`, you may create a :class:`~kumoai.graph.Table`
    object from this source table, which includes additional specifying
    information (including column semantic types and column constraint
    information).

    Args:
        name: The name of this table in the backing connector
        connector: The connector containing this table.

    .. note::
        Source tables can also be augmented with large language models to
        introduce contextual embeddings for language features. To do so, please
        consult :meth:`~kumoai.connector.SourceTable.add_llm`.

    Example:
        >>> import kumoai
        >>> connector = kumoai.S3Connector(root_dir='s3://...')  # doctest: +SKIP # noqa: E501
        >>> articles_src = connector['articles']  # doctest: +SKIP
        >>> articles_src = kumoai.SourceTable('articles', connector)  # doctest: +SKIP # noqa: E501
    """
    def __init__(self, name: str, connector: 'Connector') -> None:
        # TODO(manan): existence check, if not too expensive?
        self.name = name
        self.connector = connector

    # Metadata ################################################################

    @property
    def column_dict(self) -> Dict[str, SourceColumn]:
        r"""Returns the names of the columns in this table along with their
        :class:`SourceColumn` information.
        """
        return {col.name: col for col in self.columns}

    @property
    def columns(self) -> List[SourceColumn]:
        r"""Returns a list of the :class:`SourceColumn` metadata of the columns
        in this table.
        """
        resp: SourceTableDataResponse = self.connector._get_table_data(
            table_names=[self.name], sample_rows=0)[0]
        return resp.cols

    # Data Access #############################################################

    def head(self, num_rows: int = 5) -> pd.DataFrame:
        r"""Returns the first :obj:`num_rows` rows of this source table by
        reading data from the backing connector.

        Args:
            num_rows: The number of rows to select. If :obj:`num_rows` is
                larger than the number of available rows, all rows will be
                returned.

        Returns:
            The first :obj:`num_rows` rows of the source table as a
            :class:`~pandas.DataFrame`.
        """
        num_rows = int(num_rows)
        if num_rows <= 0:
            raise ValueError(
                f"'num_rows' must be an integer greater than 0; got {num_rows}"
            )
        try:
            resp = self.connector._get_table_data([self.name], num_rows)[0]

            # TODO(manan, siyang): consider returning `bytes` instead of `json`
            assert resp.sample_rows is not None
            return pd.read_json(StringIO(resp.sample_rows), orient='table')
        except TypeError as e:
            raise RuntimeError(f"Cannot read head of table {self.name}. "
                               "Please restart the kernel and try.") from e

    # Language Models #########################################################

    @overload
    def add_llm(
        self,
        model: str,
        api_key: str,
        template: str,
        output_dir: str,
        output_column_name: str,
        output_table_name: str,
        dimensions: Optional[int],
    ) -> 'SourceTable':
        pass

    @overload
    def add_llm(
        self,
        model: str,
        api_key: str,
        template: str,
        output_dir: str,
        output_column_name: str,
        output_table_name: str,
        dimensions: Optional[int],
        *,
        non_blocking: Literal[False],
    ) -> 'SourceTable':
        pass

    @overload
    def add_llm(
        self,
        model: str,
        api_key: str,
        template: str,
        output_dir: str,
        output_column_name: str,
        output_table_name: str,
        dimensions: Optional[int],
        *,
        non_blocking: Literal[True],
    ) -> 'LLMSourceTableFuture':
        pass

    @overload
    def add_llm(
        self,
        model: str,
        api_key: str,
        template: str,
        output_dir: str,
        output_column_name: str,
        output_table_name: str,
        dimensions: Optional[int] = None,
        *,
        non_blocking: bool,
    ) -> Union['SourceTable', 'LLMSourceTableFuture']:
        pass

    def add_llm(
        self,
        model: str,
        api_key: str,
        template: str,
        output_dir: str,
        output_column_name: str,
        output_table_name: str,
        dimensions: Optional[int] = None,
        *,
        non_blocking: bool = False,
    ) -> Union['SourceTable', 'LLMSourceTableFuture']:
        r"""Experimental method which returns a new source table that
        includes a column computed via an LLM such as OpenAI embedding models.
        Please refer to the example script for more details.

        .. note::

            Current LLM embedding only works for :obj:`SourceTable` in s3.

        .. note::

            Your :obj:`api_key` will be encrypted once we received it and
            it's only decrypted just before we call the OpenAI text embeddings.

        .. note::
            Please keep track of the token usage in the `OpenAI Dashboard
            <https://platform.openai.com/usage/activity>`_. If number of
            tokens in the data exceeds the limit, the backend will raise
            an error and no result will be produced.

        .. warning::

            This method only supports text embedding with data that has less
            than ~6 million tokens. Number of tokens is estimated by following
            `this guide <https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them>`_.

        .. warning::

            This method is still experimental. Please consult with your Kumo
            POC before using it.

        Args:
            model: The LLM model name, *e.g.*, OpenAI's
                :obj:`"text-embedding-3-small"`.
            api_key: The API key to call the LLM service.
            template: A template string to be put into the LLM. For example,
                :obj:`"{A1} and {A2}"` will fuse columns :obj:`A1` and
                :obj:`A2` into a single string.
            output_dir: The S3 directory to store the output.
            output_column_name: The output column name for the LLM.
            output_table_name: The output table name.
            dimensions: The desired LLM embedding dimension.
            non_blocking: Whether making this function non-blocking.

        Example:
            >>> import kumoai
            >>> connector = kumoai.S3Connector(root_dir='s3://...')  # doctest: +SKIP # noqa: E501
            >>> articles_src = connector['articles']  # doctest: +SKIP
            >>> articles_src_future = \
                connector["articles"].add_llm(
                    model="text-embedding-3-small",
                    api_key=YOUR_OPENAI_API_KEY,
                    template=("The product {prod_name} in the {section_name} section"
                              "is categorized as {product_type_name} "
                              "and has following description: {detail_desc}"),
                    output_dir=YOUR_OUTPUT_DIR,
                    output_column_name="embedding_column",
                    output_table_name="articles_emb",
                    dimensions=256,
                    non_blocking=True,
                )
            >>> articles_src_future.status()  # doctest: +SKIP
            >>> articles_src_future.cancel()  # doctest: +SKIP
            >>> articles_src = articles_src_future.result()  # doctest: +SKIP
        """  # noqa
        if global_state.is_spcs:
            raise NotImplementedError("add_llm is not available on Snowflake")
        source_table_type = self._to_api_source_table()
        req = LLMRequest(
            source_table_type=source_table_type,
            template=template,
            model=model,
            model_api_key=api_key,
            output_dir=output_dir,
            output_column_name=output_column_name,
            output_table_name=output_table_name,
            dimensions=dimensions,
        )
        api = global_state.client.llm_job_api
        resp: LLMJobId = api.create(req)
        logger.info(f"LLMJobId: {resp}")
        source_table_future = LLMSourceTableFuture(resp, output_table_name,
                                                   output_dir)
        if non_blocking:
            return source_table_future
        # TODO (zecheng): Add attach for text embedding
        return source_table_future.result()

    # Persistence #############################################################

    def _to_api_source_table(self) -> SourceTableType:
        r"""Cast this source table as an object of type :obj:`SourceTableType`
        for use with the public REST API.
        """
        # TODO(manan): this is stupid, and is necessary because the s3_validate
        # method in Kumo core does not properly return directories. So, we have
        # to explicitly handle this ourselves here...
        try:
            return self.connector._get_table_config(self.name).source_table
        except HTTPException:
            name = self.name.rsplit('.', maxsplit=1)[0]
            out = self.connector._get_table_config(name).source_table
            self.name = name
            return out

    @staticmethod
    def _from_api_table_definition(
            table_definition: TableDefinition) -> 'SourceTable':
        r"""Constructs a :class:`SourceTable` from a
        :class:`kumoapi.table.TableDefinition`.
        """
        from kumoai.connector import (
            BigQueryConnector,
            DatabricksConnector,
            FileUploadConnector,
            GlueConnector,
            S3Connector,
            SnowflakeConnector,
        )
        from kumoai.connector.s3_connector import S3URI
        source_type = table_definition.source_table.data_source_type
        connector: Connector
        if source_type == DataSourceType.S3:
            connector_id = table_definition.source_table.connector_id
            if connector_id in {
                    'parquet_upload_connector', 'csv_upload_connector'
            }:
                # File upload:
                connector = FileUploadConnector(
                    file_type=('parquet' if connector_id ==
                               'parquet_upload_connector' else 'csv'))
                table_name = table_definition.source_table.source_table_name
            else:
                if connector_id is not None:
                    connector = S3Connector(root_dir=None,
                                            _connector_id=connector_id)
                    table_name = (
                        table_definition.source_table.source_table_name)
                else:
                    table_path = S3URI(table_definition.source_table.s3_path)
                    connector = S3Connector(root_dir=table_path.root_dir)
                    # Strip suffix, since Kumo always takes care of that:
                    table_name = table_path.object_name.rsplit(
                        '.', maxsplit=1)[0]
        elif source_type == DataSourceType.SNOWFLAKE:
            connector_api = global_state.client.connector_api
            connector_resp = connector_api.get(
                table_definition.source_table.snowflake_connector_id)
            assert connector_resp is not None
            connector_cfg = connector_resp.config
            connector = SnowflakeConnector(
                name=connector_cfg.name,
                account=connector_cfg.account,
                warehouse=connector_cfg.warehouse,
                database=connector_cfg.database,
                schema_name=connector_cfg.schema_name,
                credentials=None,  # should be in env; do not load from DB.
                _bypass_creation=True,
            )
            table_name = table_definition.source_table.table
        elif source_type == DataSourceType.DATABRICKS:
            connector_api = global_state.client.connector_api
            connector_resp = connector_api.get(
                table_definition.source_table.databricks_connector_id)
            assert connector_resp is not None
            connector_cfg = connector_resp.config
            connector = DatabricksConnector(
                name=connector_cfg.name,
                host=connector_cfg.host,
                cluster_id=connector_cfg.cluster_id,
                warehouse_id=connector_cfg.warehouse_id,
                catalog=connector_cfg.catalog,
                credentials=None,  # should be in env; do not load from DB.
                _bypass_creation=True,
            )
            table_name = table_definition.source_table.table
        elif source_type == DataSourceType.BIGQUERY:
            connector_api = global_state.client.connector_api
            connector_resp = connector_api.get(
                table_definition.source_table.bigquery_connector_id)
            assert connector_resp is not None
            connector_cfg = connector_resp.config
            connector = BigQueryConnector(
                name=connector_cfg.name,
                project_id=connector_cfg.project_id,
                dataset_id=connector_cfg.dataset_id,
                credentials=None,  # should be in env; do not load from DB.
                _bypass_creation=True,
            )
            table_name = table_definition.source_table.table_name
        elif source_type == DataSourceType.GLUE:
            connector_api = global_state.client.connector_api
            connector_resp = connector_api.get(
                table_definition.source_table.glue_connector_id)
            assert connector_resp is not None
            connector_cfg = connector_resp.config
            connector = GlueConnector(
                name=connector_cfg.name,
                account=connector_cfg.account,
                region=connector_cfg.region,
                database=connector_cfg.database,
                _bypass_creation=True,
            )
            table_name = table_definition.source_table.table
        else:
            raise NotImplementedError()

        return SourceTable(table_name, connector)

    # Class properties ########################################################

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'


class SourceTableFuture(KumoFuture[SourceTable]):
    r"""A representation of an on-going :class:`SourceTable` generation
    process.
    """
    def __init__(
        self,
        job_id: LLMJobId,
        table_name: str,
        output_dir: str,
    ) -> None:
        self.job_id = job_id
        self._fut: concurrent.futures.Future = create_future(
            _poll(job_id, table_name, output_dir))

    @override
    def result(self) -> SourceTable:
        return self._fut.result()

    @override
    def future(self) -> 'concurrent.futures.Future[SourceTable]':
        return self._fut

    def status(self) -> JobStatus:
        return _get_status(self.job_id)


class LLMSourceTableFuture(SourceTableFuture):
    r"""A representation of an on-going :class:`SourceTable`
    generation process for LLM. This class inherits from the
    :class:`SourceTableFuture` with some functions specific
    to LLM job.
    """
    def __init__(
        self,
        job_id: LLMJobId,
        table_name: str,
        output_dir: str,
    ) -> None:
        super().__init__(job_id, table_name, output_dir)

    def cancel(self) -> JobStatus:
        r"""Cancel the LLM job."""
        api = global_state.client.llm_job_api
        return api.cancel(self.job_id)


def _get_status(job_id: str) -> JobStatus:
    api = global_state.client.llm_job_api
    resource: JobStatus = api.get(job_id)
    return resource


async def _poll(job_id: str, table_name: str, output_dir: str) -> SourceTable:
    status = _get_status(job_id)
    while not status.is_terminal:
        await asyncio.sleep(_DEFAULT_INTERVAL_S)
        status = _get_status(job_id)

    if status != JobStatus.DONE:
        raise RuntimeError(f"LLM job {job_id} failed with "
                           f"job status {status}.")

    from kumoai.connector import S3Connector
    connector = S3Connector(root_dir=output_dir)

    return connector[table_name]
