import logging
from typing import (
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

from kumoapi.jobs import (
    BatchPredictionOptions,
    BatchPredictionRequest,
    JobStatus,
    MetadataField,
    PredictionOutputConfig,
    TrainingJobRequest,
    TrainingJobResource,
)
from kumoapi.model_plan import ModelPlan

from kumoai import global_state
from kumoai.artifact_export.config import OutputConfig
from kumoai.client.jobs import (
    GeneratePredictionTableJobID,
    TrainingJobAPI,
    TrainingJobID,
)
from kumoai.connector.base import Connector
from kumoai.connector.s3_connector import S3URI
from kumoai.databricks import to_db_table_name
from kumoai.graph import Graph
from kumoai.pquery.prediction_table import PredictionTable, PredictionTableJob
from kumoai.pquery.training_table import TrainingTable, TrainingTableJob
from kumoai.trainer.job import (
    BatchPredictionJob,
    BatchPredictionJobResult,
    TrainingJob,
    TrainingJobResult,
)
from kumoai.trainer.util import (
    build_prediction_output_config,
    validate_output_arguments,
)

logger = logging.getLogger(__name__)


class Trainer:
    r"""A trainer supports creating a Kumo machine learning model on a
    :class:`~kumoai.pquery.PredictiveQuery`. It is primarily oriented around
    two methods: :meth:`~kumoai.trainer.Trainer.fit`, which accepts a
    :class:`~kumoai.graph.Graph` and :class:`~kumoai.pquery.TrainingTable` and
    produces a :class:`~kumoai.trainer.TrainingJobResult`, and
    :meth:`~kumoai.trainer.Trainer.predict`, which accepts a
    :class:`~kumoai.graph.Graph` and :class:`~kumoai.pquery.PredictionTable`
    and produces a :class:`~kumoai.trainer.BatchPredictionJobResult`.

    A :class:`~kumoai.trainer.Trainer` can also be loaded from a training job,
    with :meth:`~kumoai.trainer.Trainer.load`.

    .. code-block:: python

        import kumoai

        # See `Graph` and `PredictiveQuery` documentation:
        graph = kumoai.Graph(...)
        pquery = kumoai.PredictiveQuery(graph=graph, query=...)

        # Create a `Trainer` object, using a suggested model plan given the
        # predictive query:
        model_plan = pquery.suggest_model_plan()
        trainer = kumoai.Trainer(model_plan)

        # Create a training table from the predictive query:
        training_table = pquery.generate_training_table()

        # Fit a model:
        training_job = trainer.fit(
            graph = graph,
            training_table = training_table,
        )

        # Create a prediction table from the predictive query:
        prediction_table = pquery.generate_prediction_table()

        # Generate predictions:
        prediction_job = trainer.predict(
            graph = graph,
            prediction_table = prediction_table,
            # other arguments here...
        )

        # Load a trained query to generate predictions:
        pquery = kumoai.PredictiveQuery.load_from_training_job("trainingjob-...")
        trainer = kumoai.Trainer.load("trainingjob-...")
        prediction_job = trainer.predict(
            pquery.graph,
            pquery.generate_prediction_table(),
        )

    Args:
        model_plan: A model plan that the trainer should follow when fitting a
            Kumo model to a predictive query. This model plan can either be
            generated from a predictive query, with
            :meth:`~kumoai.pquery.PredictiveQuery.suggest_model_plan`, or can
            be manually specified.
    """  # noqa: E501

    def __init__(self, model_plan: ModelPlan) -> None:
        self._model_plan: Optional[ModelPlan] = model_plan

        # Cached from backend:
        self._training_job_id: Optional[TrainingJobID] = None

    @property
    def model_plan(self) -> Optional[ModelPlan]:
        return self._model_plan

    @model_plan.setter
    def model_plan(self, model_plan: ModelPlan) -> None:
        self._model_plan = model_plan

    # Metadata ################################################################

    @property
    def is_trained(self) -> bool:
        r"""Returns ``True`` if this trainer instance has successfully been
        trained (and is therefore ready for prediction); ``False`` otherwise.
        """
        if not self._training_job_id:
            return False
        try:
            api = global_state.client.training_job_api
            res: TrainingJobResource = api.get(self._training_job_id)
        except Exception as e:  # noqa
            logger.exception(
                "Failed to fetch training status for training job with ID %s",
                self._training_job_id, exc_info=e)
            return False
        return res.job_status_report.status == JobStatus.DONE

    # Fit / predict ###########################################################

    @overload
    def fit(
        self,
        graph: Graph,
        train_table: Union[TrainingTable, TrainingTableJob],
    ) -> TrainingJobResult:
        pass

    @overload
    def fit(
        self,
        graph: Graph,
        train_table: Union[TrainingTable, TrainingTableJob],
        *,
        non_blocking: Literal[False],
    ) -> TrainingJobResult:
        pass

    @overload
    def fit(
        self,
        graph: Graph,
        train_table: Union[TrainingTable, TrainingTableJob],
        *,
        non_blocking: Literal[True],
    ) -> TrainingJob:
        pass

    @overload
    def fit(
        self,
        graph: Graph,
        train_table: Union[TrainingTable, TrainingTableJob],
        *,
        non_blocking: bool,
    ) -> Union[TrainingJob, TrainingJobResult]:
        pass

    def fit(
        self,
        graph: Graph,
        train_table: Union[TrainingTable, TrainingTableJob],
        *,
        non_blocking: bool = False,
        custom_tags: Mapping[str, str] = {},
        warm_start_job_id: Optional[TrainingJobID] = None,
    ) -> Union[TrainingJob, TrainingJobResult]:
        r"""Fits a model to the specified graph and training table, with the
        strategy defined by this :class:`Trainer`'s :obj:`model_plan`.

        Args:
            graph: The :class:`~kumoai.graph.Graph` object that represents the
                tables and relationships that Kumo will learn from.
            train_table: The :class:`~kumoai.pquery.TrainingTable`, or
                in-progress :class:`~kumoai.pquery.TrainingTableJob`, that
                represents the training data produced by a
                :class:`~kumoai.pquery.PredictiveQuery` on :obj:`graph`.
            non_blocking: Whether this operation should return immediately
                after launching the training job, or await completion of the
                training job.
            custom_tags: Additional, customer defined k-v tags to be associated
                with the job to be launched. Job tags are useful for grouping
                and searching jobs.
            warm_start_job_id: Optional job ID of a completed training job to
                warm start from. Initializes the new model with the best
                weights from the specified job, using its model
                architecture, column processing, and neighbor sampling
                configurations.

        Returns:
            Union[TrainingJobResult, TrainingJob]:
                If ``non_blocking=False``, returns a training job object. If
                ``non_blocking=True``, returns a training job future object.
        """
        # TODO(manan, siyang): remove soon:
        job_id = train_table.job_id
        assert job_id is not None

        train_table_job_api = global_state.client.generate_train_table_job_api
        pq_id = train_table_job_api.get(job_id).config.pquery_id
        assert pq_id is not None

        custom_table = None
        if isinstance(train_table, TrainingTable):
            custom_table = train_table._custom_train_table

        # NOTE the backend implementation currently handles sequentialization
        # between a training table future and a training job; that is, if the
        # training table future is still executing, the backend will wait on
        # the job ID completion before executing a training job. This preserves
        # semantics for both futures, ensures that Kumo works as expected if
        # used only via REST API, and allows us to avoid chaining calllbacks
        # in an ugly way here:
        api = global_state.client.training_job_api
        self._training_job_id = api.create(
            TrainingJobRequest(
                dict(custom_tags),
                pquery_id=pq_id,
                model_plan=self._model_plan,
                graph_snapshot_id=graph.snapshot(non_blocking=non_blocking),
                train_table_job_id=job_id,
                custom_train_table=custom_table,
                warm_start_job_id=warm_start_job_id,
            ))

        out = TrainingJob(job_id=self._training_job_id)
        if non_blocking:
            return out
        return out.attach()

    def predict(
        self,
        graph: Graph,
        prediction_table: Union[PredictionTable, PredictionTableJob],
        output_types: Optional[Set[str]] = None,
        output_connector: Optional[Connector] = None,
        output_table_name: Optional[Union[str, Tuple]] = None,
        output_metadata_fields: Optional[List[MetadataField]] = None,
        output_config: Optional[OutputConfig] = None,
        training_job_id: Optional[TrainingJobID] = None,
        binary_classification_threshold: Optional[float] = None,
        num_classes_to_return: Optional[int] = None,
        num_workers: int = 1,
        non_blocking: bool = False,
        custom_tags: Mapping[str, str] = {},
    ) -> Union[BatchPredictionJob, BatchPredictionJobResult]:
        """Using the trained model specified by :obj:`training_job_id` (or
        the model corresponding to the last invocation of
        :meth:`~kumoai.trainer.Trainer.fit`, if not present), predicts the
        future values of the entities in :obj:`prediction_table` leveraging
        current information from :obj:`graph`.

        Args:
            graph: The :class:`~kumoai.graph.Graph` object that represents the
                tables and relationships that Kumo will use to make
                predictions.
            prediction_table: The :class:`~kumoai.pquery.PredictionTable`, or
                in-progress :class:`~kumoai.pquery.PredictionTableJob`, that
                represents the prediction data produced by a
                :class:`~kumoai.pquery.PredictiveQuery` on :obj:`graph`. This
                table may also be custom-generated.
            output_config: Output configuration defining properties of the
                generated batch prediction outputs. This includes:

                - output_types: The types of outputs that should be produced by
                  the prediction job. Can include either ``'predictions'``,
                  ``'embeddings'``, or both.
                - output_connector: The output data source that Kumo should
                  write batch predictions to, if it is None, produce
                  local download output only.
                - output_table_name: The name of the table in the output data
                  source that Kumo should write batch predictions to. In the
                  case of a Databricks connector, this should be a tuple of
                  two strings, the schema name and the output prediction
                  table name.
                - output_metadata_fields: Any additional metadata fields to
                  include as new columns in the produced ``'predictions'``
                  output. Currently, allowed options are ``JOB_TIMESTAMP``
                  and ``ANCHOR_TIMESTAMP``.
                - connector_specific_config: Custom connector specific output
                  configuration, such as whether to append or overwrite
                  existing tables.

            output_types: *(Deprecated)* The types of outputs that should be
                produced by the prediction job. Can include either
                ``'predictions'``, ``'embeddings'``, or both. Use
                :obj:`output_config` instead.
            output_connector: *(Deprecated)* The output data source that Kumo
                should write batch predictions to, if it is None, produce local
                download output only. Use :obj:`output_config` instead.
            output_table_name: *(Deprecated)* The name of the table in the
                output data source that Kumo should write batch predictions to.
                In the case of a Databricks connector, this should be a tuple
                of two strings: the schema name and the output prediction
                table name. Use :obj:`output_config` instead.
            output_metadata_fields: *(Deprecated)* Any additional metadata
                fields to include as new columns in the produced
                ``'predictions'`` output. Currently, allowed options are
                ``JOB_TIMESTAMP`` and ``ANCHOR_TIMESTAMP``. Use
                :obj:`output_config` instead.
            training_job_id: The job ID of the training job whose model will be
                used for making predictions.
            binary_classification_threshold: If this model corresponds to
                a binary classification task, the threshold for which higher
                values correspond to ``1``, and lower values correspond to
                ``0``.
            num_classes_to_return: If this model corresponds to a ranking task,
                the number of classes to return in the prediction output.
            num_workers: Number of workers to use when generating batch
                predictions. When set to a value greater than 1, the prediction
                table is partitioned into smaller parts and processed in
                parallel. The default is 1, which implies sequential inference
                over the prediction table.
            non_blocking: Whether this operation should return immediately
                after launching the batch prediction job, or await
                completion of the batch prediction job.
            custom_tags: Additional, customer defined k-v tags to be associated
                with the job to be launched. Job tags are useful for grouping
                and searching jobs.

        Returns:
            Union[BatchPredictionJob, BatchPredictionJobResult]:
                If ``non_blocking=False``, returns a training job object. If
                ``non_blocking=True``, returns a training job future object.
        """
        if (output_types is not None or output_connector is not None
                or output_table_name is not None
                or output_metadata_fields is not None):
            raise ValueError(
                'Specifying output_types, output_connector, '
                'output_metadata_fields '
                'and output_table_name as direct inputs to predict() is '
                'deprecated. Please use output_config to specify these '
                'parameters.')
        assert output_config is not None
        # Be able to pass output_config as a dictionary
        if isinstance(output_config, dict):
            output_config = OutputConfig(**output_config)
        output_table_name = to_db_table_name(output_config.output_table_name)
        validate_output_arguments(
            output_config.output_types,
            output_config.output_connector,
            output_table_name,
        )

        # Create outputs:
        outputs: List[PredictionOutputConfig] = []
        for output_type in output_config.output_types:
            if output_config.output_connector is None:
                # Predictions are generated to the Kumo dataplane, and can
                # only be exported via the UI for now:
                pass
            else:
                outputs.append(
                    build_prediction_output_config(
                        output_type,
                        output_config.output_connector,
                        output_table_name,
                        output_config.output_metadata_fields,
                        output_config,
                    ))

        training_job_id = training_job_id or self._training_job_id
        if training_job_id is None:
            raise ValueError(
                "Cannot run batch prediction without a specified or saved "
                "training job ID. Please either call `fit(...)` or pass a "
                "job ID of a completed training job to proceed.")

        pred_table_job_id: Optional[GeneratePredictionTableJobID] = \
            prediction_table.job_id
        pred_table_data_path = None
        if pred_table_job_id is None:
            assert isinstance(prediction_table, PredictionTable)
            if isinstance(prediction_table.table_data_uri, S3URI):
                pred_table_data_path = prediction_table.table_data_uri.uri
            else:
                pred_table_data_path = prediction_table.table_data_uri

        api = global_state.client.batch_prediction_job_api
        # Remove to resolve https://github.com/kumo-ai/kumo/issues/24250
        # from kumoai.pquery.predictive_query import PredictiveQuery
        # pquery = PredictiveQuery.load_from_training_job(training_job_id)
        # if pquery.get_task_type() == TaskType.BINARY_CLASSIFICATION:
        #     if binary_classification_threshold is None:
        #         logger.warning(
        # "No binary classification threshold provided. "
        # "Using default threshold of 0.5.")
        #         binary_classification_threshold = 0.5
        job_id, response = api.maybe_create(
            BatchPredictionRequest(
                dict(custom_tags),
                model_training_job_id=training_job_id,
                predict_options=BatchPredictionOptions(
                    binary_classification_threshold=(
                        binary_classification_threshold),
                    num_classes_to_return=num_classes_to_return,
                    num_workers=num_workers,
                ),
                outputs=outputs,
                graph_snapshot_id=graph.snapshot(non_blocking=non_blocking),
                pred_table_job_id=pred_table_job_id,
                pred_table_path=pred_table_data_path,
            ))

        message = response.message()
        if not response.ok:
            raise RuntimeError(f"Prediction failed. {message}")
        elif not response.empty():
            logger.warning("Prediction produced the following warnings: %s",
                           message)
        assert job_id is not None

        self._batch_prediction_job_id = job_id
        out = BatchPredictionJob(job_id=self._batch_prediction_job_id)
        if non_blocking:
            return out
        return out.attach()

    # Persistence #############################################################

    @classmethod
    def _load_from_job(cls, job: TrainingJobResource) -> 'Trainer':
        trainer = cls(job.config.model_plan)
        trainer._training_job_id = job.job_id
        return trainer

    @classmethod
    def load(cls, job_id: TrainingJobID) -> 'Trainer':
        r"""Creates a :class:`~kumoai.trainer.Trainer` instance from a training
        job ID.
        """
        api: TrainingJobAPI = global_state.client.training_job_api
        job = api.get(job_id)
        return cls._load_from_job(job)

    # TODO(siyang): load trainer by searching training job via tags.
    @classmethod
    def load_from_tags(cls, tags: Mapping[str, str]) -> 'Trainer':
        r"""Creates a :class:`~kumoai.trainer.Trainer` instance from a set of
        job tags. If multiple jobs match the list of tags, only one will be
        selected.
        """
        api = global_state.client.training_job_api
        jobs = api.list(limit=1, additional_tags=tags)
        if not jobs:
            raise RuntimeError(f'No successful training job found for {tags}')
        assert len(jobs) == 1
        return cls._load_from_job(jobs[0])
