import logging
from dataclasses import field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated, Literal

from kumoapi.common import JobSource, JobStatus, StrEnum, ValidationResponse
from kumoapi.data_snapshot import GraphSnapshotID
from kumoapi.distilled_model_plan import DistilledModelPlan
from kumoapi.model_plan import (
    ModelPlan,
    PredictionTableGenerationPlan,
    TrainingTableGenerationPlan,
)
from kumoapi.source_table import SourceTableType
from kumoapi.train import TrainingTableSpec

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    time_taken: Optional[float] = None
    train_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[
        str,
        Optional[float],  # Support for `None` in case metrics are N/A.
    ] = field(default_factory=dict)


@dataclass
class TrainerProgress:
    start_time: Optional[datetime] = None
    num_epochs_completed: int = 0
    elapsed_time: float = 0.0
    estimated_train_time: Optional[float] = None
    # TODO Make this a list
    metrics: Dict[int, EpochMetrics] = field(default_factory=dict)
    # The final best validation model performance after training is finished:
    final_validation_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.num_epochs_completed != len(self.metrics):
            raise ValueError("Number of epochs and metrics size do not match")


@dataclass
class AutoTrainerProgress:
    start_time: Optional[datetime] = None
    total_trials: int = 0
    completed_trials: int = 0
    elapsed_training_time: float = 0.0
    # Provides an estimated amount of time for the training to complete.
    # This is generally an upper bound and the estimate gets more accurate
    # as time progresses.
    estimated_training_time: Optional[float] = None
    # Map of trial ID to training progress for all running/completed trials
    trial_progress: Dict[str, TrainerProgress] = field(default_factory=dict)


@dataclass
class PredictionProgress:
    start_time: Optional[datetime] = None
    total_iterations: int = 0
    completed_iterations: int = 0
    elapsed_prediction_time: timedelta = timedelta(seconds=0)
    # Provides an estimated amount of time for the prediction to complete.
    # This is generally an upper bound and the estimate gets more accurate
    # as time progresses.
    estimated_prediction_time: Optional[timedelta] = None


class JobType(StrEnum):
    GENERATE_TRAIN_TABLE_JOB = "GENERATE_TRAIN_TABLE_JOB"
    GENERATE_PREDICTION_TABLE_JOB = "GENERATE_PREDICTION_TABLE_JOB"
    TRAINING_JOB = "TRAINING_JOB"
    BATCH_PREDICTION_JOB = "BATCH_PREDICTION_JOB"


# Execution status of a Training or Batch Prediction job.
# Log entry to recorded detailed events throughout multi-step job execution.
@dataclass
class JobEventLogEntry:
    # Name of current stage (step).
    stage_name: str
    last_updated_at: datetime
    detail: Optional[str] = None


@dataclass
class JobStatusReport:
    status: JobStatus

    # URL to the Kumo web UI page that allows human to track and monitor job
    # progress, and also view the job summary after the job finishes.
    tracking_url: str

    start_time: datetime
    end_time: Optional[datetime] = None  # Present when status is not RUNNING

    # Informational job execution event log for logging/debugging purpose.
    event_log: List[JobEventLogEntry] = field(default_factory=list)

    # Errors associated with this job mostly caused by failures in
    # async workflows
    validation_response: Optional[ValidationResponse] = None


@dataclass
class Metric:
    name: str
    value: Optional[float]


@dataclass
class ModelEvaluationMetrics:
    # Eval metrics on the test(holdout) data split.
    test_metrics: List[Metric] = field(default_factory=list)

    # Eval metrics on the validation data split.
    validation_metrics: List[Metric] = field(default_factory=list)

    # Eval metrics on the training data split.
    training_metrics: List[Metric] = field(default_factory=list)


@dataclass
class BaselineEvaluationMetrics:
    # Eval metrics on the test(holdout) data split.
    test_metrics: List[Metric] = field(default_factory=list)

    # Eval metrics on the validation data split.
    validation_metrics: Optional[List[Metric]] = None


@dataclass
class BaselineJobSummary:
    """Summary report of a successful query baseline job."""
    total_elapsed_time: timedelta

    # Model eval metrics are available when job status is DONE.
    eval_metrics: Optional[Dict[str, BaselineEvaluationMetrics]] = None


@dataclass
class TrainingJobSummary:
    """Summary report of a successful query training job."""
    # Model eval metrics are available when job status is DONE.
    eval_metrics: ModelEvaluationMetrics

    # TODO(siyang): other stats/info such as cost (GPU hours), etc.
    total_elapsed_time: timedelta
    automl_experiments_completed: int


@dataclass
class CustomTrainingTable:
    """Specifies the custom training table to be used for training.

    Args:
        source_table: The source table to the custom training table.
        table_mod_spec: The modifications made to the original training table.
        validated: Whether the custom training table has been validated
            against the original training table.
    """
    source_table: SourceTableType
    table_mod_spec: TrainingTableSpec
    validated: bool = False


@dataclass
class JobRequestBase:
    """Common job launch request options applicable to all job types."""

    # Custom key-value pair tags to be associated with the job.
    # Tags are useful for grouping, searching and managing jobs (and models).
    #
    # Requirements:
    # 1. Key may be at most 64 characters long, and may only contain
    #    alphanumeric, dot, underscore and dash characters.
    # 2. Value may be at most 256 characters long.
    job_tags: Dict[str, str]


@dataclass
class JobResourceBase:
    """Common info/metadata for job resource of any kind"""
    job_id: str
    job_status_report: JobStatusReport

    # Time when job was created (launched)
    created_at: datetime

    # All tags attached to this job, including both system-defined and
    # custom-defined tags at job launch time, as well as additional tag updates
    # made (if any) after job was launched.
    tags: Dict[str, str]


@dataclass
class GenerateTrainTableRequest(JobRequestBase):
    """POST request body to create a generate-train-table job."""
    pquery_id: str
    plan: 'TrainingTableGenerationPlan'
    graph_snapshot_id: Optional[GraphSnapshotID]


@dataclass
class GenerateTrainTableJobResource(JobResourceBase):
    config: GenerateTrainTableRequest
    source: JobSource
    user: Optional[str] = None


@dataclass
class GeneratePredictionTableRequest(JobRequestBase):
    pquery_id: str
    plan: 'PredictionTableGenerationPlan'
    graph_snapshot_id: Optional[GraphSnapshotID]


@dataclass
class GeneratePredictionTableJobResource(JobResourceBase):
    config: GeneratePredictionTableRequest
    source: JobSource
    user: Optional[str] = None


@dataclass
class BaselineJobRequest(JobRequestBase):
    """POST request body to create a baseline job."""
    pquery_id: str
    metrics: List[str]

    # Optional, a specific Graph data snapshot to use in this baseline job.
    graph_snapshot_id: Optional[GraphSnapshotID] = None

    # Optionally we can specify the ID of generate-train-table job that was
    # created with the same pquery_name.
    # If not specified (by default), a generate-train-table job will be
    # launched with default plan.
    train_table_job_id: Optional[str] = None


@dataclass
class DistillationJobRequest(JobRequestBase):
    """POST request body to create a job to train a distilled model
    for online serving."""
    pquery_id: str
    # Job Id of a trained deep GNN model used to generate embeddings
    # to train the distilled model.
    base_training_job_id: str
    distilled_model_plan: DistilledModelPlan

    # See TrainingJobRequest for documentation on train_table_job_id,
    # graph_snapshot_id, and custom_train_table fields.
    train_table_job_id: Optional[str] = None
    graph_snapshot_id: Optional[GraphSnapshotID] = None
    custom_train_table: Optional[CustomTrainingTable] = None


@dataclass
class DistillationJobResource(JobResourceBase):
    config: DistillationJobRequest
    result: Optional[TrainingJobSummary] = None


@dataclass
class TrainingJobRequest(JobRequestBase):
    """POST request body to create a training job."""
    pquery_id: str

    # Required field without default?  Or optional field default to None?
    model_plan: 'ModelPlan' = field(default_factory=ModelPlan)

    # Optional, a specific Graph data snapshot to use in this training job.
    graph_snapshot_id: Optional[GraphSnapshotID] = None

    # Optionally we can specify the ID of generate-train-table job that was
    # created with the same pquery_name.
    # If not specified (by default), a generate-train-table job will be
    # launched with default plan.
    train_table_job_id: Optional[str] = None
    # Depreciated in favor of `custom_train_table`.
    train_table_override: Optional[SourceTableType] = None

    # Used for triggering baselines jobs along with training job in v2 ui.
    enable_baselines: bool = False

    # Used to specify a train table that is modified
    # post generation via PQ. Currently only supports adding
    # a wieght colum.
    custom_train_table: Optional[CustomTrainingTable] = None

    # Training will start from the best model from
    # the below training job.
    warm_start_job_id: Optional[str] = None


@dataclass
class BaselineJobResource(JobResourceBase):
    config: BaselineJobRequest
    # Present if job status is DONE.
    result: Optional[BaselineJobSummary] = None


@dataclass
class TrainingJobResource(JobResourceBase):
    config: TrainingJobRequest

    # Present if job status is DONE.
    result: Optional[TrainingJobSummary] = None


@dataclass
class BatchPredictionOptions:
    # Required if prediction task is to perform binary classification.
    binary_classification_threshold: Optional[float] = None

    # On classification tasks, for each entity, we will only return predictions
    # for the K classes with the highest predicted values for the entity.
    # If empty, predict all class. This field is ignored for regression tasks.
    num_classes_to_return: Optional[int] = None

    # No.of workers to use when generating batch predictions. When set to a
    # value greater than 1, the prediction table is partitioned into smaller
    # parts and processed in parallel.
    #
    # Default: 1 - Sequential inference over the prediction table.
    num_workers: int = 1


class TrainingTableArtifactType(StrEnum):
    FULL_TRAIN_TABLE = "FULL_TRAIN_TABLE"
    # TODO: Add more artifact types like train table split


class PredictionArtifactType(StrEnum):
    """Specifies what kind of batch predictions should be generated.
    The user may specify multiple types of predictions to be computed, and
    each one will be output to a separate file.
    """
    PREDICTIONS = "PREDICTIONS"
    EMBEDDINGS = "EMBEDDINGS"


class PredictionStorageType(StrEnum):
    S3 = "S3"
    SNOWFLAKE = "SNOWFLAKE"
    DATABRICKS = "DATABRICKS"
    BIGQUERY = "BIGQUERY"


# Metadata fields that can be optionally selected and included as additional
# columns in the output table.
class MetadataField(StrEnum):
    ANCHOR_TIMESTAMP = 'ANCHOR_TIMESTAMP'
    JOB_TIMESTAMP = 'JOB_TIMESTAMP'


class WriteMode(StrEnum):
    OVERWRITE = "OVERWRITE"
    APPEND = "APPEND"


@dataclass
class SnowflakePredictionOutput:
    artifact_type: PredictionArtifactType
    connector_id: str
    table_name: str
    storage_type: Literal[
        PredictionStorageType.SNOWFLAKE] = PredictionStorageType.SNOWFLAKE
    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)
    # Whether to APPEND or OVERWRITE the data in the existing table. If using
    # APPEND, it is strongly recommended to use JOB_TIMESTAMP as extra_fields.
    write_mode: WriteMode = WriteMode.OVERWRITE


@dataclass
class S3PredictionOutput:
    artifact_type: PredictionArtifactType
    file_path: str
    storage_type: Literal[PredictionStorageType.S3] = PredictionStorageType.S3
    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)


@dataclass
class DatabricksPredictionOutput:
    artifact_type: PredictionArtifactType
    connector_id: str
    table_name: str
    storage_type: Literal[
        PredictionStorageType.DATABRICKS] = PredictionStorageType.DATABRICKS
    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)


@dataclass
class BigQueryPredictionOutput:
    storage_type: Literal[PredictionStorageType.BIGQUERY]
    artifact_type: PredictionArtifactType
    connector_id: str
    table_name: str
    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)

    # There are two ways to write to Bigquery:
    # 1. Write directly to the table_name. This is the default behavior
    # with staging_table_name=None. The output is *APPENDED* to the table_name.
    # It's a safe way to not overwrite the existing data in the table_name.
    # 2. Write to a staging table and then copy the staging table to the
    # table_name. After the staging table is written, the table_name will be
    # *OVERWRITTEN* with the staging table. This is done by replacing the
    # table_name with the staging table in an atomic operation. This is
    # recommended when the table_name already exists and we want to overwrite
    # the existing data in the table_name while ensuring that the table_name
    # is never corrupted.
    staging_table_name: Optional[str] = None
    # The default write mode is overwrite and if `staging_table_name`` is None,
    # if default to `table_name_{staging}`.
    write_mode: WriteMode = WriteMode.OVERWRITE


PredictionOutputConfig = Annotated[Union[SnowflakePredictionOutput,
                                         S3PredictionOutput,
                                         DatabricksPredictionOutput,
                                         BigQueryPredictionOutput],
                                   Field(discriminator='storage_type')]


@dataclass
class BatchPredictionRequest(JobRequestBase):
    """POST request body to create a Batch Prediction job."""
    # ID of a (successful) modeling job.
    model_training_job_id: str

    predict_options: BatchPredictionOptions
    outputs: List[PredictionOutputConfig] = field(default_factory=list)

    # Optional, a specific Graph data snapshot to use in this training job.
    graph_snapshot_id: Optional[GraphSnapshotID] = None

    # Optionally we can specify the ID of generate-pred-table job that
    # generates the prediction table. Only one of the following two fields can
    # be specified. If not specified (by default), a generate-pred-table job
    # will be launched with default plan.
    #
    # ID of an in-progress or successfully completed Generate-Prediction-Table
    # Job.
    pred_table_job_id: Optional[str] = None
    # File(or directory) path of the prediction table, usually custom-generated
    # by the user.
    pred_table_path: Optional[str] = None

    # Whether to enable explanations for the Batch Prediction job
    explanations: bool = False

    def __post_init__(self):
        if self.pred_table_job_id and self.pred_table_path:
            raise ValueError(
                'Only one of "pred_table_job_id" or "pred_table_path" fields '
                'can be set, not both.')


@dataclass
class BatchPredictionJobSummary:
    """Summary of a successful batch prediction job."""
    num_entities_predicted: int
    # TODO: Add more stats


@dataclass
class BatchPredictionJobResource(JobResourceBase):
    config: BatchPredictionRequest
    # Present if job status is DONE.
    result: Optional[BatchPredictionJobSummary] = None


@dataclass
class TrainingTableOutputConfig:
    table_name: str = ""  # leave empty for s3
    artifact_type: TrainingTableArtifactType =\
        TrainingTableArtifactType.FULL_TRAIN_TABLE

    # For non-s3 connector specify connector_id
    # For s3 connector, specify s3_path
    connector_id: Optional[str] = None
    s3_path: Optional[str] = None

    # Write mode is applicable only for non-s3 connector.
    # Whether to APPEND or OVERWRITE the data in the existing table. If using
    # APPEND, it is strongly recommended to use JOB_TIMESTAMP as extra_fields.
    write_mode: WriteMode = WriteMode.OVERWRITE

    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)

    def __post_init__(self):
        if self.connector_id is None and self.s3_path is None:
            raise ValueError(
                'At least one of "connector_id" or "s3_path" fields '
                'must be set.')
        elif self.connector_id and self.s3_path:
            raise ValueError('Only one of "connector_id" or "s3_path" fields '
                             'can be set, not both.')

        if self.connector_id and self.table_name == "":
            raise ValueError('Table name must be set if connector_id is set')


@dataclass
class ArtifactExportRequest:
    """POST request body to create an artifact export job.

    Args:
        job_id: ID of a job that generates the artifact.
        prediction_output: Optional prediction output configuration. For
            uploading prediction/embedding artifacts.
        training_table_output: Optional training table output configuration.
            For uploading training table artifacts.
    """
    # ID of a job that generates the artifact.
    job_id: str

    prediction_output: Optional[PredictionOutputConfig] = None

    training_table_output: Optional[TrainingTableOutputConfig] = None

    # TODO: Support more artifacts type

    def __post_init__(self):
        if (self.prediction_output is None
                and self.training_table_output is None):
            raise ValueError('At least one of "prediction_output" or '
                             '"training_table_output" fields must be set.')
        elif self.prediction_output and self.training_table_output:
            raise ValueError(
                'Only one of "prediction_output" or '
                '"training_table_output" fields can be set, not both.')


@dataclass
class ArtifactExportResponse:
    # ID of a artifact export job.
    job_id: str


@dataclass
class CancelTrainingJobResponse:
    is_cancelled: bool


@dataclass
class CancelBatchPredictionJobResponse:
    is_cancelled: bool


@dataclass
class GetPredictionsDfUrlResponse:
    """
    Response class for /prediction_jobs/{job_id}/get_prediction_df_urls.
    """
    # List of presigned URLs, each entry corresponding to one parquet partition
    prediction_partitions: List[str]


@dataclass
class GetEmbeddingsDfUrlResponse:
    """
    Response class for /prediction_jobs/{job_id}/get_embedding_df_urls.
    """
    # List of presigned URLs, each entry corresponding to one parquet partition
    embedding_partitions: List[str]


class ErrorType(Enum):
    """
    Enumeration of different error/info response types returned to the
    user.
    """
    # Mostly errors which prevent user from proceeding with an
    # operation.
    ERROR = 0
    # Mostly warnings/info which does not prevent user from
    # proceeding but could be good insights.
    INFO = 1


@dataclass
class ErrorCTA:
    """
    Class representing potential actionable items for the users.
    """
    # While name is mostly useful for the UI, it can be use to
    # format error responses in SDK APIs as well.
    name: str
    # The url associated with the the click to action.
    url: str


@dataclass
class ErrorDetail:
    """
    Each error in workflow execution is associated with one instance
    of :class:ErrorDetail.
    """
    type: ErrorType
    description: str
    title: Optional[str]
    # Captures potential actionable items based on the error.
    cta: Optional[ErrorCTA]


@dataclass
class ErrorDetails:
    """
    Response class for errors and warnings returned by long running
    jobs.
    """
    items: List[ErrorDetail]
