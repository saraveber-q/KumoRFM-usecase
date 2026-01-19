import asyncio
import concurrent
import concurrent.futures
import time
from datetime import datetime, timezone
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from urllib.parse import urlparse, urlunparse

import pandas as pd
from kumoapi.common import JobStatus
from kumoapi.data_snapshot import GraphSnapshotID
from kumoapi.jobs import (
    ArtifactExportRequest,
    AutoTrainerProgress,
    BaselineEvaluationMetrics,
    BaselineJobRequest,
    BatchPredictionJobSummary,
    BatchPredictionRequest,
    JobStatusReport,
    ModelEvaluationMetrics,
    PredictionProgress,
    TrainingJobRequest,
)
from kumoapi.model_plan import ModelPlan
from kumoapi.online_serving import (
    OnlinePredictionOptions,
    OnlineServingEndpointRequest,
)
from kumoapi.task import TaskType
from tqdm.auto import tqdm
from typing_extensions import override

from kumoai import global_state
from kumoai.artifact_export import (
    ArtifactExportJob,
    ArtifactExportResult,
    OutputConfig,
)
from kumoai.client.jobs import (
    BaselineJobAPI,
    BaselineJobID,
    BaselineJobResource,
    BatchPredictionJobAPI,
    BatchPredictionJobID,
    BatchPredictionJobResource,
    TrainingJobAPI,
    TrainingJobID,
    TrainingJobResource,
)
from kumoai.databricks import to_db_table_name
from kumoai.futures import KumoProgressFuture, create_future
from kumoai.jobs import JobInterface
from kumoai.trainer.online_serving import OnlineServingEndpointFuture
from kumoai.trainer.util import (
    build_prediction_output_config,
    validate_output_arguments,
)

if TYPE_CHECKING:
    from kumoai.pquery import (
        PredictionTable,
        PredictionTableJob,
        PredictiveQuery,
        TrainingTable,
        TrainingTableJob,
    )


class BaselineJobResult:
    r"""Represents a completed baseline job.

    A :class:`BaselineJobResult` object can either be obtained by creating a
    :class:`~kumoai.trainer.BaselineJob` object and calling the
    :meth:`~kumoai.trainer.BaselineJob.result` method to await the job's
    completion, or by directly creating the object. The former approach is
    recommended, as it includes verification that the job finished succesfully.

    Example:
        >>> import kumoai  # doctest: +SKIP
        >>> job_future = kumoai.BaselineJob(id=...)  # doctest: +SKIP
        >>> job = job_future.result()  # doctest: +SKIP
    """
    def __init__(self, job_id: BaselineJobID) -> None:
        self.job_id = job_id

        # A cached completed, finalized job resource:
        self._job_resource: Optional[BaselineJobResource] = None

    def metrics(self) -> Dict[str, BaselineEvaluationMetrics]:
        r"""Returns the metrics associated with this completed training job,
        or raises an exception if metrics cannot be obtained.
        """
        return self._get_job_resource(
            require_completed=True).result.eval_metrics

    def _get_job_resource(self,
                          require_completed: bool) -> BaselineJobResource:
        if self._job_resource:
            return self._job_resource

        try:
            api = global_state.client.baseline_job_api
            resource: BaselineJobResource = api.get(self.job_id)
        except Exception as e:
            raise RuntimeError(
                f"Baseline job {self.job_id} was not found in the Kumo "
                f"database. Please contact Kumo for further assistance. "
            ) from e

        if not require_completed:
            return resource

        status = resource.job_status_report.status
        if not status.is_terminal:
            raise RuntimeError(
                f"Baseline job {self.job_id} has not yet completed. Please "
                f"create a `BaselineJob` class and await its completion "
                f"before attempting to view metrics.")

        if status != JobStatus.DONE:
            # Should never happen, the future will not resolve:
            raise ValueError(
                f"Baseline job {self.job_id} completed with status {status}, "
                f"and was therefore unable to produce metrics. Please "
                f"re-train the job until it successfully completes.")

        self._job_resource = resource
        return self._job_resource


class TrainingJobResult:
    r"""Represents a completed training job.

    A :class:`TrainingJobResult` object can either be obtained by creating a
    :class:`~kumoai.trainer.TrainingJob` object and calling the
    :meth:`~kumoai.trainer.TrainingJob.result` method to await the job's
    completion, or by directly creating the object. The former approach is
    recommended, as it includes verification that the job finished succesfully.

    .. code-block:: python

        import kumoai

        training_job = kumoai.TrainingJob("trainingjob-...")

        # Wait for a training job's completion, and get its result:
        training_job_result = training_job.result()

        # Alternatively, create the result directly, but be sure that the job
        # is completed:
        training_job_result = kumoai.TrainingJobResult("trainingjob-...")

        # Get associated objects:
        pquery = training_job_result.predictive_query
        training_table = training_job_result.training_table

        # Get holdout data:
        holdout_df = training_job_result.holdout_df()

    Example:
        >>> import kumoai  # doctest: +SKIP
        >>> job_future = kumoai.TrainingJob(id=...)  # doctest: +SKIP
        >>> job = job_future.result()  # doctest: +SKIP
    """
    def __init__(self, job_id: TrainingJobID) -> None:
        self.job_id = job_id

        # A cached completed, finalized job resource:
        self._job_resource: Optional[TrainingJobResource] = None

    @property
    def id(self) -> TrainingJobID:
        r"""The unique ID of this training job."""
        return self.job_id

    @property
    def model_plan(self) -> ModelPlan:
        r"""Returns the model plan associated with this training job."""
        return self._get_job_resource(
            require_completed=False).config.model_plan

    @property
    def training_table(self) -> Union['TrainingTableJob', 'TrainingTable']:
        r"""Returns the training table associated with this training job,
        either as a :class:`~kumoai.pquery.TrainingTable` or a
        :class:`~kumoai.pquery.TrainingTableJob` depending on the status of
        the training table generation job.
        """
        from kumoai.pquery import TrainingTableJob
        training_table_job_id = self._get_job_resource(
            require_completed=False).config.train_table_job_id
        if training_table_job_id is None:
            raise RuntimeError(
                f"Unable to access the training table generation job ID for "
                f"job {self.job_id}. Did this job fail before generating its "
                f"training table?")
        fut = TrainingTableJob(training_table_job_id)
        if fut.status().status == JobStatus.DONE:
            return fut.result()
        return fut

    @property
    def predictive_query(self) -> 'PredictiveQuery':
        r"""Returns the :class:`~kumoai.pquery.PredictiveQuery` object that
        defined the training table for this training job.
        """
        from kumoai.pquery import PredictiveQuery
        return PredictiveQuery.load_from_training_job(self.job_id)

    @property
    def tracking_url(self) -> str:
        r"""Returns a tracking URL pointing to the UI display of this training
        job.
        """
        tracking_url = self._get_job_resource(
            require_completed=False).job_status_report.tracking_url
        return _rewrite_tracking_url(tracking_url)

    def metrics(self) -> ModelEvaluationMetrics:
        r"""Returns the metrics associated with this completed training job,
        or raises an exception if metrics cannot be obtained.
        """
        return self._get_job_resource(
            require_completed=True).result.eval_metrics

    def holdout_url(self) -> str:
        r"""Returns a URL for downloading or reading the holdout dataset.

        If Kumo is deployed as a SaaS application, the returned URL will be a
        presigned AWS S3 URL with a TTL of 1 hour. If Kumo is deployed as a
        Snowpark Container Services application, the returned URL will be a
        Snowflake stage path that can be directly accessed within a Snowflake
        worksheet.
        """
        api: TrainingJobAPI = global_state.client.training_job_api
        return api.holdout_data_url(self.job_id, presigned=True)

    def holdout_df(self) -> pd.DataFrame:
        r"""Reads the holdout dataset (parquet file) as pandas DataFrame.

        .. note::
            Please note that this function may be memory-intensive, depending
            on the size of your holdout dataframe. Please exercise caution.
        """
        holdout_url = self.holdout_url()

        if global_state.is_spcs:
            from kumoai.spcs import _get_session

            # TODO(dm): return type hint is wrong
            return _get_session().read.parquet(holdout_url)

        if holdout_url.startswith("dbfs:"):
            raise ValueError(f"holdout_df is unsupported for "
                             f"Databricks UC Volume path {holdout_url}")

        return pd.read_parquet(holdout_url)

    def launch_online_serving_endpoint(
        self,
        pred_options: OnlinePredictionOptions = OnlinePredictionOptions(),
        snapshot_id: Optional[GraphSnapshotID] = None,
    ) -> OnlineServingEndpointFuture:
        self._get_job_resource(require_completed=True)
        pquery = self.predictive_query
        task_type = pquery.get_task_type()
        if task_type == TaskType.BINARY_CLASSIFICATION:
            if not pred_options.binary_classification_threshold:
                raise ValueError(
                    'Missing binary_classification_threshold option')
        if (not task_type.is_classification
                and task_type != TaskType.REGRESSION):
            raise ValueError(
                f'{task_type} does not yet support online serving')

        endpoint_id = global_state.client.online_serving_endpoint_api.create(
            OnlineServingEndpointRequest(self.id, pred_options, snapshot_id))
        return OnlineServingEndpointFuture(endpoint_id)

    def _get_job_resource(self,
                          require_completed: bool) -> TrainingJobResource:
        if self._job_resource:
            return self._job_resource

        try:
            api = global_state.client.training_job_api
            resource: TrainingJobResource = api.get(self.job_id)
        except Exception as e:
            raise RuntimeError(
                f"Training job {self.job_id} was not found in the Kumo "
                f"database. Please contact Kumo for further assistance. "
            ) from e

        if not require_completed:
            return resource

        status = resource.job_status_report.status
        if not status.is_terminal:
            raise RuntimeError(
                f"Training job {self.job_id} has not yet completed. Please "
                f"create a `TrainingJob` class and await its completion "
                f"before attempting to view metrics.")

        if status != JobStatus.DONE:
            # Should never happen, the future will not resolve:
            raise ValueError(
                f"Training job {self.job_id} completed with status {status}, "
                f"and was therefore unable to produce metrics. Please "
                f"re-train the job until it successfully completes.")

        self._job_resource = resource
        return self._job_resource


class BatchPredictionJobResult:
    r"""Represents a completed batch prediction job.

    A :class:`BatchPredictionJobResult` object can either be obtained by
    creating a :class:`~kumoai.trainer.BatchPredictionJob` object and calling
    the :meth:`~kumoai.trainer.BatchPredictionJob.result` method to await the
    job's completion, or by directly creating the object. The former approach
    is recommended, as it includes verification that the job finished
    succesfully.

    .. code-block:: python

        import kumoai

        prediction_job = kumoai.BatchPredictionJob("bp-job-...")

        # Wait for a batch prediction job's completion, and get its result:
        prediction_job_result = prediction_job.result()

        # Alternatively, create the result directly, but be sure that the job
        # is completed:
        prediction_job_result = kumoai.BatchPredictionJobResult("bp-job-...")

        # Get associated objects:
        prediction_table = prediction_job_result.prediction_table

        # Get prediction data (in-memory):
        predictions_df = training_job.predictions_df()

        # Export prediction data to any output connector:
        prediction_job_result.export(
            output_type = ...,
            output_connector = ...,
            output_table_name = ...,
        )
    """  # noqa: E501

    def __init__(self, job_id: BatchPredictionJobID) -> None:
        self.job_id = job_id
        self._job_resource: Optional[BatchPredictionJobResource] = None

    @property
    def id(self) -> BatchPredictionJobID:
        r"""The unique ID of this batch prediction job."""
        return self.job_id

    @property
    def tracking_url(self) -> str:
        r"""Returns a tracking URL pointing to the UI display of this batch
        prediction job.
        """
        tracking_url = self._get_job_resource(
            require_completed=False).job_status_report.tracking_url
        return _rewrite_tracking_url(tracking_url)

    def summary(self) -> BatchPredictionJobSummary:
        r"""Returns summary statistics associated with the batch prediction
        job's output, or raises an exception if summary statistics cannot be
        obtained.
        """
        return self._get_job_resource(require_completed=True).result

    @property
    def prediction_table(
            self) -> Union['PredictionTableJob', 'PredictionTable']:
        r"""Returns the prediction table associated with this prediction job,
        either as a :class:`~kumoai.pquery.PredictionTable` or a
        :class:`~kumoai.pquery.PredictionTableJob` depending on the status
        of the prediction table generation job.
        """
        from kumoai.pquery import PredictionTableJob
        prediction_table_job_id = self._get_job_resource(
            require_completed=False).config.pred_table_job_id
        if prediction_table_job_id is None:
            raise RuntimeError(
                f"Unable to access the prediction table generation job ID for "
                f"job {self.job_id}. Did this job fail before generating its "
                f"prediction table, or use a custom prediction table?")
        fut = PredictionTableJob(prediction_table_job_id)
        if fut.status().status == JobStatus.DONE:
            return fut.result()
        return fut

    def export(
        self,
        output_config: OutputConfig,
        non_blocking: bool = True,
    ) -> Union['ArtifactExportJob', 'ArtifactExportResult']:
        r"""Export the prediction output or the embedding output to the
        specific output location.

        Args:
            output_config: The output configuration to be used.
            non_blocking: If ``True``, the method will return a future object
                `ArtifactExportJob` representing the export job.
                If ``False``, the method will block until the export job is
                complete and return `ArtifactExportResult`.
        """
        output_table_name = to_db_table_name(output_config.output_table_name)
        validate_output_arguments(
            (output_config.output_types),
            output_config.output_connector,
            output_table_name,
        )
        if output_config.output_types is not None and len(
                output_config.output_types) > 1:
            raise ValueError(
                f'Each export request can only support one output_type, '
                f'received {output_config.output_types}. If you want to make '
                'multiple output_type exports, please make separate export() '
                'calls.')
        prediction_output_config = build_prediction_output_config(
            list(output_config.output_types)[0],
            output_config.output_connector,
            output_table_name,
            output_config.output_metadata_fields,
            output_config,
        )

        api = global_state.client.artifact_export_api
        request = ArtifactExportRequest(
            job_id=self.id, prediction_output=prediction_output_config)
        job_id = api.create(request)
        if non_blocking:
            return ArtifactExportJob(job_id)
        return ArtifactExportJob(job_id).attach()

    def predictions_urls(self) -> List[str]:
        r"""Returns a list of URLs for downloading or reading the predictions.

        If Kumo is deployed as a SaaS application, the returned URLs will be
        presigned AWS S3 URLs. If Kumo is deployed as a Snowpark Container
        Services application, the returned URLs will be Snowflake stage paths
        that can be directly accessed within a Snowflake worksheet. If Kumo is
        deployed as a Databricks application, Databricks UC volume paths.
        """
        api: BatchPredictionJobAPI = (
            global_state.client.batch_prediction_job_api)
        return api.get_batch_predictions_url(self.job_id)

    def predictions_df(self) -> pd.DataFrame:
        r"""Returns a :class:`~pandas.DataFrame` object representing the
        generated predictions.

        .. warning::

            This method will load the full prediction output into memory as a
            :class:`~pandas.DataFrame` object. If you are working on a machine
            with limited resources, please use
            :meth:`~kumoai.trainer.BatchPredictionResult.predictions_urls`
            instead to download the data and perform analysis per-partition.
        """
        urls = self.predictions_urls()
        try:
            return pd.concat(pd.read_parquet(x) for x in urls)
        except Exception as e:
            raise ValueError(
                f"Could not create a Pandas DataFrame object from data paths "
                f"{urls}. Please construct the DataFrame manually.") from e

    def embeddings_urls(self) -> List[str]:
        r"""Returns a list of URLs for downloading or reading the embeddings.

        If Kumo is deployed as a SaaS application, the returned URLs will be
        presigned AWS S3 URLs. If Kumo is deployed as a Snowpark Container
        Services application, the returned URLs will be Snowflake stage paths
        that can be directly accessed within a Snowflake worksheet. If Kumo is
        deployed as a Databricks application, Databricks UC volume paths.
        """
        api: BatchPredictionJobAPI = (
            global_state.client.batch_prediction_job_api)
        return api.get_batch_embeddings_url(self.job_id)

    def embeddings_df(self) -> pd.DataFrame:
        r"""Returns a :class:`~pandas.DataFrame` object representing the
        generated embeddings.

        .. warning::

            This method will load the full prediction output into memory as a
            :class:`~pandas.DataFrame` object. If you are working on a machine
            with limited resources, please use
            :meth:`~kumoai.trainer.BatchPredictionResult.embeddings_urls`
            instead to download the data and perform analysis per-partition.
        """
        urls = self.embeddings_urls()
        try:
            return pd.concat(pd.read_parquet(x) for x in urls)
        except Exception as e:
            raise ValueError(
                f"Could not create a Pandas DataFrame object from data paths "
                f"{urls}. Please construct the DataFrame manually.") from e

    def _get_job_resource(
            self, require_completed: bool) -> BatchPredictionJobResource:
        if self._job_resource:
            return self._job_resource

        try:
            api = global_state.client.batch_prediction_job_api
            resource: BatchPredictionJobResource = api.get(self.job_id)
        except Exception as e:
            raise RuntimeError(
                f"Batch prediction job {self.job_id} was not found in the "
                f"Kumo database. Please contact Kumo for further assistance. "
            ) from e

        if not require_completed:
            return resource

        status = resource.job_status_report.status
        if not status.is_terminal:
            raise RuntimeError(
                f"Batch prediction job {self.job_id} has not yet completed. "
                f"Please create a `BatchPredictionJob` class and await "
                "its completion before attempting to view stats.")

        if status != JobStatus.DONE:
            validation_resp = resource.job_status_report.validation_response
            validation_message = ""
            if validation_resp:
                validation_message = validation_resp.message()
            if len(validation_message) > 0:
                validation_message = f"Details: {validation_message}"

            raise ValueError(
                f"Batch prediction job {self.job_id} completed with status "
                f"{status}, and was therefore unable to produce metrics. "
                f"{validation_message}")

        self._job_resource = resource
        return resource


# Training Job Future #########################################################


class TrainingJob(JobInterface[TrainingJobID, TrainingJobRequest,
                               TrainingJobResource],
                  KumoProgressFuture[TrainingJobResult]):
    r"""Represents an in-progress training job.

    A :class:`TrainingJob` object can either be created as the result of
    :meth:`~kumoai.trainer.Trainer.fit` with ``non_blocking=True``, or
    directly with a training job ID (*e.g.* of a job created asynchronously in
    a different environment).


    .. code-block:: python

        import kumoai

        # See `Trainer` documentation:
        trainer = kumoai.Trainer(...)

        # If a Trainer is `fit` in nonblocking mode, the response will be of
        # type `TrainingJob`:
        training_job = trainer.fit(..., non_blocking=True)

        # You can also construct a `TrainingJob` from a job ID, e.g. one that
        # is present in the Kumo Jobs page:
        training_job = kumoai.TrainingJob("trainingjob-...")

        # Get the status of the job:
        print(training_job.status())

        # Attach to the job, and poll progress updates:
        training_job.attach()
        # Training: 70%|█████████    | [300s<90s, trial=4, train_loss=1.056, val_loss=0.682, val_mae=35.709, val_mse=7906.239, val_rmse=88.917

        # Cancel the job:
        training_job.cancel()

        # Wait for the job to complete, and return a `TrainingJobResult`:
        training_job.result()

    Args:
        job_id: The training job ID to await completion of.
    """  # noqa

    @override
    @staticmethod
    def _api() -> TrainingJobAPI:
        return global_state.client.training_job_api

    def __init__(self, job_id: TrainingJobID) -> None:
        self.job_id = job_id

    @cached_property
    def _fut(self) -> concurrent.futures.Future:
        return create_future(_poll_training(self.job_id))

    @override
    @property
    def id(self) -> TrainingJobID:
        r"""The unique ID of this training job."""
        return self.job_id

    @override
    def result(self) -> TrainingJobResult:
        return self._fut.result()

    @override
    def future(self) -> 'concurrent.futures.Future[TrainingJobResult]':
        return self._fut

    @property
    def tracking_url(self) -> str:
        r"""Returns a tracking URL pointing to the UI that can be used to
        monitor the status of an ongoing or completed job.
        """
        return _rewrite_tracking_url(self.status().tracking_url)

    @override
    def _attach_internal(
        self,
        interval_s: float = 20.0,
    ) -> TrainingJobResult:
        r"""Allows a user to attach to a running training job, and view its
        progress inline.

        Args:
            interval_s (float): Time interval (seconds) between polls, minimum
                value allowed is 4 seconds.

        Example:
            >>> job_future = kumoai.TrainingJob(job_id="...")  # doctest: +SKIP
            >>> job_future.attach()  # doctest: +SKIP
            Attaching to training job <id>. To track this job...
            Training: 70%|█████████    | [300s<90s, trial=4, train_loss=1.056, val_loss=0.682, val_mae=35.709, val_mse=7906.239, val_rmse=88.917
        """  # noqa
        assert interval_s >= 4.0
        print(f"Attaching to training job {self.job_id}. To track this job in "
              f"the Kumo UI, please visit {self.tracking_url}. To detach from "
              f"this job, please enter Ctrl+C: the job will continue to run, "
              f"and you can re-attach anytime by calling the `attach()` "
              f"method on the `TrainingJob` object. For example: "
              f"kumoai.TrainingJob(\"{self.job_id}\").attach()")

        # TODO(manan): this is not perfect, the `asyncio.sleep` in the poller
        # may cause a "DONE" status to be printed for up to
        # interval_s*`timeout` seconds before the future resolves.
        # That's probably fine:
        if self.done():
            return self.result()

        # For every non-training stage, just show the stage and status:
        print("Waiting for job to start.")
        current_status = JobStatus.NOT_STARTED
        while current_status == JobStatus.NOT_STARTED:
            report = self.status()
            current_status = report.status
            current_stage = report.event_log[-1].stage_name
            time.sleep(interval_s)

        prev_stage = current_stage
        print(f"Current stage: {current_stage}. In progress...", end="",
              flush=True)
        while not self.done():
            # Print status of stage:
            if current_stage != prev_stage:
                print(" Done.")
                print(f"Current stage: {current_stage}. In progress...",
                      end="", flush=True)
            if current_stage == "Training":
                _time = self.progress().estimated_training_time
                if _time and _time != 0:
                    break
            time.sleep(interval_s)
            report = self.status()
            prev_stage = current_stage
            current_stage = report.event_log[-1].stage_name

        # We are not on Training:
        if self.done():
            return self.result()

        # We are training: print a progress bar
        progress = self.progress()
        bar_format = '{desc}: {percentage:3.0f}%|{bar}|{unit} '
        total = int(progress.estimated_training_time)
        elapsed = int(progress.elapsed_training_time)
        pbar = tqdm(desc="Training", unit="% done", bar_format=bar_format,
                    total=total, dynamic_ncols=True)
        pbar.update(elapsed)

        while not self.done():
            progress = self.progress()
            trial_no = min(progress.completed_trials + 1,
                           progress.total_trials)

            if f'{max(trial_no-1, 0)}' in progress.trial_progress:
                trial_metrics = progress.trial_progress[
                    f'{max(trial_no-1, 0)}'].metrics
            elif f'{max(trial_no-2, 0)}' in progress.trial_progress:
                trial_metrics = progress.trial_progress[
                    f'{max(trial_no-2, 0)}'].metrics
            else:
                trial_metrics = {}

            # If we don't have metrics, wait until we do:
            if len(trial_metrics) == 0:
                continue

            # Show all metrics:
            # TODO(manan): only show tune metric, trial, epoch, and loss:
            last_epoch_metrics = trial_metrics[sorted(
                trial_metrics.keys())[-1]]
            train_metrics_s = ", ".join([
                f"{key_name}={key_val:.3f}" for key_name, key_val in
                last_epoch_metrics.train_metrics.items()
            ])
            val_metrics_s = ", ".join([
                f"{key_name}={key_val:.3f}" for key_name, key_val in
                last_epoch_metrics.validation_metrics.items()
            ])

            # Update numbers:
            delta = int(progress.elapsed_training_time - pbar.n)
            total = int(progress.estimated_training_time)
            pbar.update(delta)
            pbar.total = total
            if pbar.n > pbar.total:
                pbar.total = pbar.n

            # NOTE we use `unit` here as a hack, instead of `set_postfix`,
            # since `tqdm` defaults to adding a comma before the postfix
            # (https://github.com/tqdm/tqdm/issues/712)
            pbar.unit = (f"[{pbar.n}s<{pbar.total-pbar.n}s, trial={trial_no}, "
                         f"{train_metrics_s}, {val_metrics_s}]")
            pbar.refresh()
            time.sleep(interval_s)
        pbar.update(pbar.total - pbar.n)
        pbar.close()

        # Future is done:
        return self.result()

    def progress(self) -> AutoTrainerProgress:
        r"""Returns the progress of an ongoing or completed training job."""
        return self._api().get_progress(self.job_id)

    @override
    def status(self) -> JobStatusReport:
        r"""Returns the status of a running training job."""
        return _get_training_status(self.job_id)

    def cancel(self) -> bool:
        r"""Cancels a running training job, and returns ``True`` if
        cancellation succeeded.

        Example:
            >>> job_future = kumoai.TrainingJob(job_id="...")  # doctest: +SKIP
            >>> job_future.cancel()  # doctest: +SKIP
        """  # noqa
        return self._api().cancel(self.job_id).is_cancelled

    @override
    def load_config(self) -> TrainingJobRequest:
        r"""Load the full configuration for this training job.

        Returns:
            TrainingJobRequest: Complete configuration including model_plan,
            pquery_id, graph_snapshot_id, train_table_job_id, etc.
        """
        return self._api().get_config(self.job_id)


def _get_training_status(job_id: str) -> JobStatusReport:
    api = global_state.client.training_job_api
    resource: TrainingJobResource = api.get(job_id)
    return resource.job_status_report


async def _poll_training(job_id: str) -> TrainingJobResult:
    # TODO(manan): make asynchronous natively with aiohttp:
    status = _get_training_status(job_id).status
    while not status.is_terminal:
        await asyncio.sleep(10)
        status = _get_training_status(job_id).status

    # TODO(manan, siyang): improve
    if status != JobStatus.DONE:
        api = global_state.client.training_job_api
        job_resource = api.get(job_id)
        validation_resp = (job_resource.job_status_report.validation_response)

        validation_message = ""
        if validation_resp:
            validation_message = validation_resp.message()
        if len(validation_message) > 0:
            validation_message = f"Details: {validation_message}"

        raise RuntimeError(f"Training job {job_id} completed with job status "
                           f"{status}. {validation_message}")

    # TODO(manan): improve
    return TrainingJobResult(job_id=job_id)


# Batch Prediction Job Future #################################################


class BatchPredictionJob(JobInterface[BatchPredictionJobID,
                                      BatchPredictionRequest,
                                      BatchPredictionJobResource],
                         KumoProgressFuture[BatchPredictionJobResult]):
    r"""Represents an in-progress batch prediction job.

    A :class:`BatchPredictionJob` object can either be created as the
    result of :meth:`~kumoai.trainer.Trainer.predict` with
    ``non_blocking=True``, or directly with a batch prediction job ID (*e.g.*
    of a job created asynchronously in a different environment).

    .. code-block:: python

        import kumoai

        # See `Trainer` documentation:
        trainer = kumoai.Trainer(...)

        # If a Trainer `predict` is called in nonblocking mode, the response
        # will be of type `BatchPredictionJob`:
        prediction_job = trainer.predict(..., non_blocking=True)

        # You can also construct a `BatchPredictionJob` from a job ID, e.g. one
        # that is present in the Kumo Jobs page:
        prediction_job = kumoai.BatchPredictionJob("bp-job-...")

        # Get the status of the job:
        print(prediction_job.status())

        # Attach to the job, and poll progress updates:
        prediction_job.attach()
        # Attaching to batch prediction job <id>. To track this job...
        # Predicting (job_id=..., start=..., elapsed=..., status=...). Stage: ...

        # Cancel the job:
        prediction_job.cancel()

        # Wait for the job to complete, and return a `BatchPredictionJobResult`:
        prediction_job.result()

    Args:
        job_id: The batch prediction job ID to await completion of.
    """  # noqa

    @override
    @staticmethod
    def _api() -> BatchPredictionJobAPI:
        return global_state.client.batch_prediction_job_api

    def __init__(self, job_id: BatchPredictionJobID) -> None:
        self.job_id = job_id

    @cached_property
    def _fut(self) -> concurrent.futures.Future:
        return create_future(_poll_batch_prediction(self.job_id))

    @override
    @property
    def id(self) -> BatchPredictionJobID:
        r"""The unique ID of this batch prediction job."""
        return self.job_id

    @override
    def result(self) -> BatchPredictionJobResult:
        return self._fut.result()

    @override
    def future(self) -> 'concurrent.futures.Future[BatchPredictionJobResult]':
        return self._fut

    @property
    def tracking_url(self) -> str:
        r"""Returns a tracking URL pointing to the UI that can be used to
        monitor the status of an ongoing or completed job.
        """
        return _rewrite_tracking_url(self.status().tracking_url)

    @override
    def _attach_internal(
        self,
        interval_s: float = 20.0,
    ) -> BatchPredictionJobResult:
        r"""Allows a user to attach to a running batch prediction job, and view
        its progress inline.

        Args:
            interval_s (float): Time interval (seconds) between polls, minimum
                value allowed is 4 seconds.

        """
        assert interval_s >= 4.0
        print(f"Attaching to batch prediction job {self.job_id}. To track "
              f"this job in the Kumo UI, please visit {self.tracking_url}. To "
              f"detach from this job, please enter Ctrl+C (the job will "
              f"continue to run, and you can re-attach anytime).")
        # TODO(manan): this is not perfect, the `asyncio.sleep` in the poller
        # may cause a "DONE" status to be printed for up to
        # interval_s*`timeout` seconds before the future resolves.
        # That's probably fine:
        if self.done():
            return self.result()

        print("Waiting for job to start.")
        current_status = JobStatus.NOT_STARTED
        while current_status == JobStatus.NOT_STARTED:
            report = self.status()
            current_status = report.status
            current_stage = report.event_log[-1].stage_name
            time.sleep(interval_s)

        prev_stage = current_stage
        print(f"Current stage: {current_stage}. In progress...", end="",
              flush=True)
        while not self.done():
            # Print status of stage:
            if current_stage != prev_stage:
                print(" Done.")
                print(f"Current stage: {current_stage}. In progress...",
                      end="", flush=True)
                if current_stage == "Predicting":
                    _time = self.progress().estimated_prediction_time
                    if _time and _time != 0:
                        break

            time.sleep(interval_s)
            report = self.status()
            prev_stage = current_stage
            current_stage = report.event_log[-1].stage_name

        # We are not on Batch Prediction:
        if self.done():
            return self.result()

        # We are predicting: print a progress bar
        bar_format = '{desc}: {percentage:3.0f}%|{bar} '
        total_iterations, elapsed = 0, 0
        pbar = tqdm(desc="Predicting", unit="% done", bar_format=bar_format,
                    total=100, dynamic_ncols=True)
        pbar.update(elapsed)

        while not self.done():
            progress = self.progress()
            if progress is None:
                time.sleep(interval_s)
                continue
            total_iterations = progress.total_iterations
            completed_iterations = progress.completed_iterations
            pbar.update(
                (completed_iterations - elapsed) / total_iterations * 100)
            elapsed = completed_iterations
            elapsed_pct = completed_iterations / total_iterations
            pbar.refresh()
            time.sleep(interval_s)

        pbar.update(1.0 - elapsed_pct)
        pbar.close()

        # Future is done:
        return self.result()

    def progress(self) -> PredictionProgress:
        r"""Returns the progress of an ongoing or completed batch prediction
        job.
        """
        return self._api().get_progress(self.job_id)

    @override
    def status(self) -> JobStatusReport:
        r"""Returns the status of a running batch prediction job."""
        return _get_batch_prediction_status(self.job_id)

    def cancel(self) -> bool:
        r"""Cancels a running batch prediction job, and returns ``True`` if
        cancellation succeeded.
        """
        return self._api().cancel(self.job_id).is_cancelled

    @override
    def load_config(self) -> BatchPredictionRequest:
        r"""Load the full configuration for this batch prediction job.

        Returns:
            BatchPredictionRequest: Complete
            configuration including predict_options,
            outputs, model_training_job_id, etc.
        """
        return self._api().get_config(self.job_id)


def _get_batch_prediction_job(job_id: str) -> BatchPredictionJobResource:
    api = global_state.client.batch_prediction_job_api
    return api.get(job_id)


def _get_batch_prediction_status(job_id: str) -> JobStatusReport:
    api = global_state.client.batch_prediction_job_api
    resource: BatchPredictionJobResource = api.get(job_id)
    return resource.job_status_report


async def _poll_batch_prediction(job_id: str) -> BatchPredictionJobResult:
    # TODO(manan): make asynchronous natively with aiohttp:
    job_resource = _get_batch_prediction_job(job_id)
    status = job_resource.job_status_report.status
    while not status.is_terminal:
        await asyncio.sleep(10)
        job_resource = _get_batch_prediction_job(job_id)
        status = job_resource.job_status_report.status

    # TODO(manan, siyang): improve
    if status != JobStatus.DONE:
        validation_resp = job_resource.job_status_report.validation_response
        validation_message = ""
        if validation_resp:
            validation_message = validation_resp.message()
        if len(validation_message) > 0:
            validation_message = f"Details: {validation_message}"

        raise ValueError(
            f"Batch prediction job {job_id} completed with status "
            f"{status}, and was therefore unable to produce metrics. "
            f"{validation_message}")

    # TODO(manan): improve
    return BatchPredictionJobResult(job_id=job_id)


# Baseline Job Future #################################################


class BaselineJob(JobInterface[BaselineJobID, BaselineJobRequest,
                               BaselineJobResource],
                  KumoProgressFuture[BaselineJobResult]):
    r"""Represents an in-progress baseline job.

    A :class:`BaselineJob` object can either be created as the result of
    :meth:`~kumoai.trainer.BaselineTrainer.run` with ``non_blocking=True``, or
    directly with a baseline job ID (*e.g.* of a job created asynchronously in
    a different environment).

    Args:
        job_id: The baseline job ID to await completion of.

    Example:
        >>> import kumoai  # doctest: +SKIP
        >>> id = "some_baseline_job_id"
        >>> job_future = kumoai.BaselineJob(id)  # doctest: +SKIP
        >>> job_future.attach()  # doctest: +SKIP
        Attaching to baseline job <id>. To track this job...
    """  # noqa

    @override
    @staticmethod
    def _api() -> BaselineJobAPI:
        return global_state.client.baseline_job_api

    def __init__(self, job_id: BaselineJobID) -> None:
        self.job_id = job_id

    @cached_property
    def _fut(self) -> concurrent.futures.Future:
        return create_future(_poll_baseline(self.job_id))

    @override
    @property
    def id(self) -> BaselineJobID:
        r"""The unique ID of this training job."""
        return self.job_id

    @override
    def result(self) -> BaselineJobResult:
        return self._fut.result()

    @override
    def future(self) -> 'concurrent.futures.Future[BaselineJobResult]':
        return self._fut

    @property
    def tracking_url(self) -> str:
        r"""Returns a tracking URL pointing to the UI that can be used to
        monitor the status of an ongoing or completed job.
        """
        return ""

    @override
    def _attach_internal(
        self,
        interval_s: float = 20.0,
    ) -> BaselineJobResult:
        r"""Allows a user to attach to a running baseline job, and view its
        progress inline.

        Args:
            interval_s (float): Time interval (seconds) between polls, minimum
                value allowed is 4 seconds.

        Example:
            >>> job_future = kumoai.BaselineJob(job_id="...")  # doctest: +SKIP
            >>> job_future.attach()  # doctest: +SKIP
            Attaching to baseline job <id>. To track this job...
        """  # noqa
        assert interval_s >= 4.0
        print(f"Attaching to baseline job {self.job_id}."
              f"To detach from "
              f"this job, please enter Ctrl+C (the job will continue to run, "
              f"and you can re-attach anytime).")

        while not self.done():
            report = self.status()
            status = report.status
            latest_event = report.event_log[-1]
            stage = latest_event.stage_name
            detail = ", " + latest_event.detail if latest_event.detail else ""

            start = report.start_time
            now = datetime.now(timezone.utc)
            print(f"Baseline job (job_id={self.job_id} start={start}, elapsed="
                  f"{now-start}, status={status}). Stage: {stage}{detail}")
            time.sleep(interval_s)

        # Future is done:
        return self.result()

    @override
    def status(self) -> JobStatusReport:
        r"""Returns the status of a running baseline job."""
        return _get_baseline_status(self.job_id)

    @override
    def load_config(self) -> BaselineJobRequest:
        r"""Load the full configuration for this baseline job.

        Returns:
            BaselineJobRequest: Complete configuration including metrics,
            pquery_id, graph_snapshot_id, etc.
        """
        return self._api().get_config(self.job_id)


def _get_baseline_status(job_id: str) -> JobStatusReport:
    api = global_state.client.baseline_job_api
    resource: BaselineJobResource = api.get(job_id)
    return resource.job_status_report


async def _poll_baseline(job_id: str) -> BaselineJobResult:
    status = _get_baseline_status(job_id).status
    while not status.is_terminal:
        await asyncio.sleep(10)
        status = _get_baseline_status(job_id).status

    if status != JobStatus.DONE:
        raise RuntimeError(
            f"Baseline job {job_id} failed with job status {status}.")

    return BaselineJobResult(job_id=job_id)


def _rewrite_tracking_url(tracking_url: str) -> str:
    r"""Rewrites tracking URL to account for deployment subdomains."""
    # TODO(manan): improve...
    if 'http' not in tracking_url:
        return tracking_url
    parsed_base = urlparse(global_state.client._url)
    parsed_tracking = urlparse(tracking_url)
    tracking_url = urlunparse((
        parsed_base.scheme,
        parsed_base.netloc,
        parsed_tracking.path,
        parsed_tracking.params,
        parsed_tracking.query,
        parsed_tracking.fragment,
    ))
    return tracking_url
