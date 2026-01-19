from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from datetime import datetime
from functools import cached_property
from typing import List, Optional, Union

import pandas as pd
from kumoapi.common import JobStatus
from kumoapi.jobs import (
    GeneratePredictionTableJobResource,
    GeneratePredictionTableRequest,
    JobStatusReport,
)
from typing_extensions import override

from kumoai import global_state
from kumoai.client.jobs import (
    GeneratePredictionTableJobAPI,
    GeneratePredictionTableJobID,
)
from kumoai.connector.s3_connector import S3URI
from kumoai.formatting import pretty_print_error_details
from kumoai.futures import KumoFuture, create_future
from kumoai.jobs import JobInterface

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 20


class PredictionTable:
    r"""A prediction table in the Kumo platform. A prediction table can
    either be initialized from a job ID of a completed prediction table
    generation job, or a path on a supported object store (S3 for a SaaS or
    Databricks deployment, and Snowflake session storage for Snowflake).

    .. warning::
        Custom prediction table is an experimental feature; please work
        with your Kumo POC to ensure you are using it correctly!


    .. code-block:: python

        import kumoai

        # Create a Prediction Table from a prediction table generation job.
        # Note that the job ID passed here must be in a completed state:
        prediction_table = kumoai.PredictionTable("gen-predtable-job-...")

        # Read the prediction table as a Pandas DataFrame:
        prediction_df = prediction_table.data_df()

        # Get URLs to download the prediction table:
        prediction_download_urls = prediction_table.data_urls()

    Args:
        job_id: ID of the prediction table generation job which
            generated this prediction table. If a custom table data path is
            specified, this parameter should be left as ``None``.
        table_data_path: S3 path of the table data location, for which Kumo
            must at least have read access. If a job ID is specified, this
            parameter should be left as ``None``.
    """
    def __init__(
        self,
        job_id: Optional[GeneratePredictionTableJobID] = None,
        table_data_path: Optional[str] = None,
    ) -> None:
        # Validation:
        if not (job_id or table_data_path):
            raise ValueError(
                "A PredictionTable must either be initialized with a table "
                "data path, or a job ID of a completed prediction table "
                "generation job.")
        if job_id and table_data_path:
            raise ValueError(
                "Please either pass a table data path, or a job ID of a "
                "completed prediction table generation job; passing both "
                "is not allowed.")

        # Custom path:
        self.table_data_uri: Optional[Union[str, S3URI]] = None
        if table_data_path is not None:
            if table_data_path.startswith('dbfs:/'):
                raise ValueError(
                    "Files from Databricks UC Volumes are not supported")
            if global_state.is_spcs:
                if table_data_path.startswith('s3://'):
                    raise ValueError(
                        "SPCS does not support S3 paths for prediction tables."
                    )
                # TODO(zeyuan): support custom stage path on SPCS:
                self.table_data_uri = table_data_path
            else:
                self.table_data_uri = S3URI(table_data_path).validate()

        # Job ID:
        self.job_id = job_id
        if job_id:
            status = _get_status(job_id).status
            if status != JobStatus.DONE:
                raise ValueError(
                    f"Job {job_id} is not yet complete (status: {status}). If "
                    f"you would like to create a future (waiting for "
                    f"prediction table generation success), please use "
                    f"`PredictionTableJob`.")

    def data_urls(self) -> List[str]:
        r"""Returns a list of URLs that can be used to view generated
        prediction table data; if a custom data path was passed, this path is
        simply returned.

        The list will contain more than one element if the table is
        partitioned; paths will be relative to the location of the Kumo data
        plane.
        """
        api = global_state.client.generate_prediction_table_job_api
        if not self.job_id:
            # Custom prediction table:
            if global_state.is_spcs:
                assert isinstance(self.table_data_uri, str)
                return [self.table_data_uri]
            else:
                assert isinstance(self.table_data_uri, S3URI)
                return [self.table_data_uri.uri]
        return api.get_table_data(self.job_id, presigned=True)

    def data_df(self) -> pd.DataFrame:
        r"""Returns a Pandas DataFrame object representing the generated
        or custom-specified prediction table data.

        .. warning::

            This method will load the full prediction table into memory as a
            :class:`~pandas.DataFrame` object. If you are working on a machine
            with limited resources, please use
            :meth:`~kumoai.pquery.PredictionTable.data_urls` instead to
            download the data and perform analysis per-partition.
        """
        if global_state.is_spcs:
            from kumoai.spcs import _parquet_dataset_to_df

            # TODO(dm): return type hint is wrong
            return _parquet_dataset_to_df(self.data_urls())
        else:
            urls = self.data_urls()
            try:
                return pd.concat([pd.read_parquet(x) for x in urls])
            except Exception as e:
                raise ValueError(
                    f"Could not create a Pandas DataFrame object from data "
                    f"paths {urls}. Please construct the DataFrame manually."
                ) from e

    @property
    def anchor_time(self) -> Optional[datetime]:
        r"""Returns the anchor time corresponding to the generated prediction
        table data, if the data was not custom-specified.
        """
        if self.job_id is None:
            logger.warning(
                "Fetching the anchor time is not supported for a custom "
                "prediction table (path: %s)", self.table_data_uri)
            return None
        api = global_state.client.generate_prediction_table_job_api
        return api.get_anchor_time(self.job_id)


# Prediction Table Future #####################################################


class PredictionTableJob(JobInterface[GeneratePredictionTableJobID,
                                      GeneratePredictionTableRequest,
                                      GeneratePredictionTableJobResource],
                         KumoFuture[PredictionTable]):
    r"""A representation of an ongoing prediction table generation job in the
    Kumo platform.

    .. code-block:: python

        import kumoai

        # See `PredictiveQuery` documentation:
        pquery = kumoai.PredictiveQuery(...)

        # If a prediction table is generated in nonblocking mode, the response
        # will be of type `PredictionTableJob`:
        prediction_table_job = pquery.generate_prediction_table(non_blocking=True)

        # You can also construct a `PredictionTableJob` from a job ID, e.g.
        # one that is present in the Kumo Jobs page:
        prediction_table_job = kumoai.PredictionTableJob("gen-predtable-job-...")

        # Get the status of the job:
        print(prediction_table_job.status())

        # Cancel the job:
        prediction_table_job.cancel()

        # Wait for the job to complete, and return a `PredictionTable`:
        prediction_table_job.result()

    Args:
        job_id: ID of the prediction table generation job.
    """  # noqa

    @override
    @staticmethod
    def _api() -> GeneratePredictionTableJobAPI:
        return global_state.client.generate_prediction_table_job_api

    def __init__(
        self,
        job_id: GeneratePredictionTableJobID,
    ) -> None:
        self.job_id = job_id
        self.job: Optional[GeneratePredictionTableJobResource] = None

    @cached_property
    def _fut(self) -> Future:
        return create_future(self._poll())

    @override
    @property
    def id(self) -> GeneratePredictionTableJobID:
        r"""The unique ID of this prediction table generation process."""
        return self.job_id

    @override
    def result(self) -> PredictionTable:
        return self._fut.result()

    @override
    def future(self) -> Future[PredictionTable]:
        return self._fut

    @override
    def status(self) -> JobStatusReport:
        r"""Returns the status of a running prediction table generation job."""
        return self._poll_job().job_status_report

    def cancel(self) -> None:
        r"""Cancels a running prediction table generation job, and raises an
        error if cancellation failed.
        """
        return self._api().cancel(self.job_id)

    # TODO(manan): make asynchronous natively with aiohttp:
    def _poll_job(self) -> GeneratePredictionTableJobResource:
        # Skip polling if job is already in terminal state.
        if not self.job or not self.job.job_status_report.status.is_terminal:
            self.job = self._api().get(self.job_id)
        return self.job

    async def _poll(self) -> PredictionTable:
        while not self.status().status.is_terminal:
            await asyncio.sleep(_DEFAULT_INTERVAL_S)
        status = self.status().status
        if status != JobStatus.DONE:
            error_details = self._api().get_job_error(self.job_id)
            error_str = pretty_print_error_details(error_details)
            raise RuntimeError(
                f"Prediction table generation for job {self.job_id} failed "
                f"with job status {status}. Encountered below"
                f" errors: {error_str}")
        return PredictionTable(self.job_id)

    @override
    def load_config(self) -> GeneratePredictionTableRequest:
        r"""Load the full configuration for this
        prediction table generation job.

        Returns:
            GeneratePredictionTableRequest:
                Complete configuration including plan,
            pquery_id, graph_snapshot_id, etc.
        """
        return self._api().get_config(self.job_id)


def _get_status(job_id: str) -> JobStatusReport:
    api = global_state.client.generate_prediction_table_job_api
    resource: GeneratePredictionTableJobResource = api.get(job_id)
    return resource.job_status_report
