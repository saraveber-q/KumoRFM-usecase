from typing import List

from kumoapi.source_table import (
    DataSourceType,
    FileType,
    S3SourceTableRequest,
    SourceTableConfigRequest,
    SourceTableConfigResponse,
)
from typing_extensions import override

from kumoai import global_state
from kumoai.connector.base import Connector
from kumoai.connector.utils import delete_uploaded_table, upload_table


class FileUploadConnector(Connector):
    r"""Defines a connector to files directly uploaded to Kumo, either as
    'parquet' or 'csv' (non-partitioned) data.

    To get started with file upload, please first upload a table with
    the :meth:`upload` method in the :class:`FileUploadConnector` class.
    You can then access
    this table behind the file upload connector as follows:

    .. code-block:: python

        import kumoai

        # Create the file upload connector:
        connector = kumoai.FileUploadConnector(file_type="parquet")

        # Upload the table; assume it is stored at `/data/users.parquet`
        connector.upload(name="users", path="/data/users.parquet")

        # Check that the file upload connector has a `users` table:
        assert connector.has_table("users")

    Args:
        file_type: The file type of uploaded data. Can be either ``"csv"``
            or ``"parquet"``.
    """
    def __init__(self, file_type: str) -> None:
        r"""Creates the connector to uploaded files of type
        :obj:`file_type`.
        """
        assert file_type.lower() in {'parquet', 'csv'}
        self._file_type = file_type.lower()

    @property
    def name(self) -> str:
        return f'{self._file_type}_upload_connector'

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.S3

    @property
    def file_type(self) -> FileType:
        return (FileType.PARQUET
                if self._file_type == 'parquet' else FileType.CSV)

    def _get_table_config(self, table_name: str) -> SourceTableConfigResponse:
        req = SourceTableConfigRequest(connector_id=self.name,
                                       table_name=table_name,
                                       source_type=self.source_type,
                                       file_type=None)
        return global_state.client.source_table_api.get_table_config(req)

    @override
    def _source_table_request(self,
                              table_names: List[str]) -> S3SourceTableRequest:
        return S3SourceTableRequest(s3_root_dir="", connector_id=self.name,
                                    table_names=table_names, file_type=None)

    def upload(
        self,
        name: str,
        path: str,
        auto_partition: bool = True,
        partition_size_mb: int = 250,
    ) -> None:
        r"""Upload a table to Kumo from a local or remote path.

        Supports ``s3://``, ``gs://``, ``abfs://``, ``abfss://``, and ``az://``

        Tables uploaded this way can be accessed from this
        ``FileUploadConnector`` using the provided name, e.g.,
        ``connector_obj["my_table"]``.

        Local files
        -----------
        - Accepts one ``.parquet`` or ``.csv`` file (must match this
          connector’s ``file_type``).
        - If the file is > 1 GiB and ``auto_partition=True``, it is split
          into ~``partition_size_mb`` MiB parts and uploaded under a common
          prefix so the connector can read them as one table.

        Remote paths
        ------------
        - **Single file** (``.parquet``/``.csv``): validated and uploaded via
          multipart PUT. Files > 1 GiB are rejected — re-shard to ~200 MiB
          and upload the directory instead.
        - **Directory**: must contain only one format (all Parquet or all CSV)
          matching this connector’s ``file_type``. Files are validated
          (consistent schema; CSV headers sanitized) and uploaded in parallel
          with memory-safe budgeting.

        .. warning::
           For local uploads, input must be a single CSV or Parquet file
           (matching the connector type). For remote uploads, mixed
           CSV/Parquet directories are not supported. Remote single files
           larger than 1 GiB are not supported.

        Examples:
        ---------
        .. code-block:: python

            import kumoai
            conn = kumoai.FileUploadConnector(file_type="parquet")

            # Local: small file
            conn.upload(name="users", path="/data/users.parquet")

            # Local: large file (auto-partitions)
            conn.upload(
                name="txns",
                path="/data/large_txns.parquet",
            )

            # Local: disable auto-partitioning (raises if > 1 GiB)
            conn.upload(
                name="users",
                path="/data/users.parquet",
                auto_partition=False,
            )

            # CSV connector
            csv_conn = kumoai.FileUploadConnector(file_type="csv")
            csv_conn.upload(name="sales", path="/data/sales.csv")

            # Remote: single file (<= 1 GiB)
            conn.upload(name="logs", path="s3://bkt/path/logs.parquet")

            # Remote: directory of shards (uniform format)
            csv_conn.upload(name="events", path="gs://mybkt/events_csv/")

        Args:
            name:
                Table name to create in Kumo; access later via this connector.
            path:
                Local path or remote URL to a ``.parquet``/``.csv`` file or a
                directory (uniform format). The format must match this
                connector’s ``file_type``.
            auto_partition:
                Local-only. If ``True`` and the local file is > 1 GiB, split
                into ~``partition_size_mb`` MiB parts.
            partition_size_mb:
                Local-only. Target partition size (100–1000 MiB) when
                ``auto_partition`` is ``True``.
        """
        upload_table(name=name, path=path, auto_partition=auto_partition,
                     partition_size_mb=partition_size_mb,
                     file_type=self._file_type)

    def delete(
        self,
        name: str,
    ) -> None:
        r"""Synchronously deletes a previously uploaded table from the Kumo
        data plane.

        .. code-block:: python

            # Assume we have uploaded a `.parquet` table named `users`, and a
            # `FileUploadConnector` has been created called `connector`, and
            # we want to delete this table from Kumo:
            connector.delete(name="users")

        Args:
            name: The name of the table to be deleted. This table must have
                previously been uploaded with a call to
                :meth:`~kumoai.connector.FileUploadConnector.upload`.
        """
        if not self.has_table(name):
            raise ValueError(f"The table '{name}' does not exist in {self}. "
                             f"Please check the existence of the source data.")

        delete_uploaded_table(name, self._file_type)
