from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import Future
from functools import cached_property
from typing import List, Optional, Tuple, Union

import pandas as pd
from kumoapi.common import JobStatus
from kumoapi.jobs import (
    ArtifactExportRequest,
    CustomTrainingTable,
    GenerateTrainTableJobResource,
    GenerateTrainTableRequest,
    JobStatusReport,
    SourceTableType,
    TrainingTableOutputConfig,
    TrainingTableSpec,
    WriteMode,
)
from kumoapi.source_table import S3SourceTable
from tqdm.auto import tqdm
from typing_extensions import Self, override

from kumoai import global_state
from kumoai.artifact_export import (
    ArtifactExportJob,
    ArtifactExportResult,
    TrainingTableExportConfig,
)
from kumoai.client.jobs import (
    GenerateTrainTableJobAPI,
    GenerateTrainTableJobID,
)
from kumoai.connector import S3Connector, SourceTable
from kumoai.databricks import to_db_table_name
from kumoai.formatting import pretty_print_error_details
from kumoai.futures import KumoProgressFuture, create_future
from kumoai.jobs import JobInterface

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 20


class TrainingTable:
    r"""A training table in the Kumo platform. A training table can be
    initialized from a job ID of a completed training table generation job.

    .. code-block:: python

        import kumoai

        # Create a Training Table from a training table generation job. Note
        # that the job ID passed here must be in a completed state:
        training_table = kumoai.TrainingTable("gen-traintable-job-...")

        # Read the training table as a Pandas DataFrame:
        training_df = training_table.data_df()

        # Get URLs to download the training table:
        training_download_urls = training_table.data_urls()

        # Add weight column to the training table:
        # see `kumo-sdk.examples.datasets.weighted_train_table.py`
        # for a more detailed example
        # 1. Export train table
        connector = kumo.S3Connector("s3_path")
        training_table.export(TrainingTableExportConfig(
            output_types={'training_table'},
            output_connector=connector,
            output_table_name="<any_name>"))
        # 2. Assume the weight column was added to the train table
        # and it was saved to the same S3 path as "<mod_name>"
        training_table.update(SourceTable("<mod_table>", connector),
                              TrainingTableSpec(weight_col="weight"))

    Args:
        job_id: ID of the training table generation job which generated this
            training table.
    """
    def __init__(self, job_id: GenerateTrainTableJobID):
        self.job_id = job_id
        status = _get_status(job_id).status
        self._custom_train_table: Optional[CustomTrainingTable] = None
        if status != JobStatus.DONE:
            raise ValueError(
                f"Job {job_id} is not yet complete (status: {status}). If you "
                f"would like to create a future (waiting for training table "
                f"generation success), please use `TrainingTableJob`.")

    def data_urls(self) -> List[str]:
        r"""Returns a list of URLs that can be used to view generated
        training table data. The list will contain more than one element
        if the table is partitioned; paths will be relative to the location of
        the Kumo data plane.
        """
        api: GenerateTrainTableJobAPI = (
            global_state.client.generate_train_table_job_api)
        return api._get_table_data(self.job_id, presigned=True, raw_path=True)

    def data_df(self) -> pd.DataFrame:
        r"""Returns a :class:`~pandas.DataFrame` object representing the
        generated training data.

        .. warning::

            This method will load the full training table into memory as a
            :class:`~pandas.DataFrame` object. If you are working on a machine
            with limited resources, please use
            :meth:`~kumoai.pquery.TrainingTable.data_urls` instead to download
            the data and perform analysis per-partition.
        """
        urls = self.data_urls()
        if global_state.is_spcs:
            from kumoai.spcs import _parquet_dataset_to_df

            # TODO(dm): return type hint is wrong
            return _parquet_dataset_to_df(self.data_urls())

        try:
            return pd.concat([pd.read_parquet(x) for x in urls])
        except Exception as e:
            raise ValueError(
                f"Could not create a Pandas DataFrame object from data paths "
                f"{urls}. Please construct the DataFrame manually.") from e

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(job_id={self.job_id})'

    def _to_s3_api_source_table(self,
                                source_table: SourceTable) -> S3SourceTable:
        assert isinstance(source_table.connector, S3Connector)
        source_type = source_table._to_api_source_table()
        root_dir: str = source_table.connector.root_dir  # type: ignore
        if root_dir[-1] != os.sep:
            root_dir = root_dir + os.sep
        return S3SourceTable(
            s3_path=root_dir,
            source_table_name=source_table.name,
            file_type=source_type.file_type,
        )

    def export(
        self,
        output_config: TrainingTableExportConfig,
        non_blocking: bool = True,
    ) -> Union[ArtifactExportJob, ArtifactExportResult]:
        r"""Export the training table to the connector.
        specified in the output config. Use the exported table to
        add a weight column then use `update` to update the training table.

        Args:
            output_config: The output configuration to write the training
                table.
            non_blocking: If ``True``, the method will return a future object
                `ArtifactExportJob` representing the export job.
                If ``False``, the method will block until the export job is
                complete and return `ArtifactExportResult`.
        """
        assert output_config.output_connector is not None
        assert output_config.output_types == {'training_table'}
        output_table_name = to_db_table_name(output_config.output_table_name)
        assert output_table_name is not None
        s3_path = None
        connector_id = None
        table_name = ""
        write_mode = WriteMode.OVERWRITE

        if isinstance(output_config.output_connector, S3Connector):
            assert output_config.output_connector.root_dir is not None
            s3_path = output_config.output_connector.root_dir
            s3_path = os.path.join(s3_path, output_table_name)
        else:
            connector_id = output_config.output_connector.name
            table_name = output_table_name
            if output_config.connector_specific_config:
                write_mode = output_config.connector_specific_config.write_mode

        api = global_state.client.artifact_export_api
        output_config = TrainingTableOutputConfig(
            s3_path=s3_path,
            connector_id=connector_id,
            table_name=table_name,
            write_mode=write_mode,
        )

        request = ArtifactExportRequest(job_id=self.job_id,
                                        training_table_output=output_config)
        job_id = api.create(request)
        if non_blocking:
            return ArtifactExportJob(job_id)
        return ArtifactExportJob(job_id).attach()

    def validate_custom_table(
        self,
        source_table_type: SourceTableType,
        train_table_mod: TrainingTableSpec,
    ) -> None:
        r"""Validates the modified training table.

        Args:
            source_table_type: The source table to be used as the modified
                training table.
            train_table_mod: The modification specification.

        Raises:
            ValueError: If the modified training table is invalid.

        """
        api: GenerateTrainTableJobAPI = (
            global_state.client.generate_train_table_job_api)
        response = api.validate_custom_train_table(self.job_id,
                                                   source_table_type,
                                                   train_table_mod)
        if not response.ok:
            raise ValueError("Invalid weighted train table",
                             response.error_message)

    def update(
        self,
        source_table: SourceTable,
        train_table_mod: TrainingTableSpec,
        validate: bool = True,
    ) -> Self:
        r"""Sets the `source_table` as the modified training table.

        .. note::
            The only allowed modification is the addition of weight column
            Any other modification might lead to unintentded ERRORS while
            training.
            Further negative/NA weight values are not supported.

        The custom training table is ingested during trainer.fit()
        and is used as the training table.

        Args:
            source_table: The source table to be used as the modified training
                table.
            train_table_mod: The modification specification.
            validate: Whether to validate the modified training table. This can
                be slow for large tables.
        """
        if isinstance(source_table.connector, S3Connector):
            # Special handling for s3 as `source_table._to_api_source_table`
            # concatenates root_dir and file name. But the backend expects
            # these to be separate.
            source_table_type = self._to_s3_api_source_table(source_table)
        else:
            source_table_type = source_table._to_api_source_table()
        if validate:
            self.validate_custom_table(source_table_type, train_table_mod)
        self._custom_train_table = CustomTrainingTable(
            source_table=source_table_type, table_mod_spec=train_table_mod,
            validated=validate)
        return self


# Training Table Future #######################################################


class TrainingTableJob(JobInterface[GenerateTrainTableJobID,
                                    GenerateTrainTableRequest,
                                    GenerateTrainTableJobResource],
                       KumoProgressFuture[TrainingTable]):
    r"""A representation of an ongoing training table generation job in the
    Kumo platform.

    .. code-block:: python

        import kumoai

        # See `PredictiveQuery` documentation:
        pquery = kumoai.PredictiveQuery(...)

        # If a training table is generated in nonblocking mode, the response
        # will be of type `TrainingTableJob`:
        training_table_job = pquery.generate_training_table(non_blocking=True)

        # You can also construct a `TrainingTableJob` from a job ID, e.g.
        # one that is present in the Kumo Jobs page:
        training_table_job = kumoai.TrainingTableJob("trainingjob-...")

        # Get the status of the job:
        print(training_table_job.status())

        # Attach to the job, and poll progress updates:
        training_table_job.attach()

        # Cancel the job:
        training_table_job.cancel()

        # Wait for the job to complete, and return a `TrainingTable`:
        training_table_job.result()

    Args:
        job_id: ID of the training table generation job.
    """
    @override
    @staticmethod
    def _api() -> GenerateTrainTableJobAPI:
        return global_state.client.generate_train_table_job_api

    def __init__(
        self,
        job_id: GenerateTrainTableJobID,
    ) -> None:
        self.job_id = job_id

    @cached_property
    def _fut(self) -> Future[TrainingTable]:
        return create_future(_poll(self.job_id))

    @override
    @property
    def id(self) -> GenerateTrainTableJobID:
        r"""The unique ID of this training table generation process."""
        return self.job_id

    @override
    def result(self) -> TrainingTable:
        return self._fut.result()

    @override
    def future(self) -> Future[TrainingTable]:
        return self._fut

    @override
    def status(self) -> JobStatusReport:
        r"""Returns the status of a running training table generation job."""
        return _get_status(self.job_id)

    @override
    def _attach_internal(self, interval_s: float = 20.0) -> TrainingTable:
        assert interval_s >= 4.0
        print(f"Attaching to training table generation job {self.job_id}. "
              f"Tracking this job in the Kumo UI is coming soon. To detach "
              f"from this job, please enter Ctrl+C (the job will continue to "
              f"run, and you can re-attach anytime).")

        def _get_progress() -> Optional[Tuple[int, int]]:
            progress = self._api().get_progress(self.job_id)
            if len(progress) == 0:
                return None
            expected_iter = progress['num_expected_iterations']
            completed_iter = progress['num_finished_iterations']
            return (expected_iter, completed_iter)

        # Print progress bar:
        print("Training table generation is in progress. If your task is "
              "temporal, progress per timeframe will be loaded shortly.")

        # Wait for either timeframes to become available, or the job is done:
        progress = _get_progress()
        while progress is None:
            progress = _get_progress()
            # Not a temporal task, and it's done:
            if self.status().status.is_terminal:
                return self.result()
            time.sleep(interval_s)

        # Wait for timeframes to become available:
        progress = _get_progress()
        assert progress is not None
        total, prog = progress
        pbar = tqdm(total=total, unit="timeframe",
                    desc="Generating Training Table")
        pbar.update(pbar.n - prog)
        while not self.done():
            progress = _get_progress()
            assert progress is not None
            total, prog = progress
            pbar.reset(total)
            pbar.update(prog)
            time.sleep(interval_s)
        pbar.update(pbar.total)
        pbar.close()

        # Future is done:
        return self.result()

    def cancel(self) -> None:
        r"""Cancels a running training table generation job, and raises an
        error if cancellation failed.
        """
        return self._api().cancel(self.job_id)

    @override
    def load_config(self) -> GenerateTrainTableRequest:
        r"""Load the full configuration for this training table generation job.

        Returns:
            GenerateTrainTableRequest: Complete configuration including plan,
            pquery_id, graph_snapshot_id, etc.
        """
        return self._api().get_config(self.job_id)


def _get_status(job_id: str) -> JobStatusReport:
    api = global_state.client.generate_train_table_job_api
    resource: GenerateTrainTableJobResource = api.get(job_id)
    return resource.job_status_report


async def _poll(job_id: str) -> TrainingTable:
    # TODO(manan): make asynchronous natively with aiohttp:
    status = _get_status(job_id).status
    while not status.is_terminal:
        await asyncio.sleep(_DEFAULT_INTERVAL_S)
        status = _get_status(job_id).status

    if status != JobStatus.DONE:
        api = global_state.client.generate_train_table_job_api
        error_details = api.get_job_error(job_id)
        error_str = pretty_print_error_details(error_details)
        raise RuntimeError(
            f"Training table generation for job {job_id} failed with "
            f"job status {status}. Encountered below error(s):"
            f'{error_str}')

    return TrainingTable(job_id)
