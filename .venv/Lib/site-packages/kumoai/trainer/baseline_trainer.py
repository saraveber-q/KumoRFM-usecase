from typing import List, Mapping, Optional, Union

from kumoapi.jobs import BaselineJobRequest

from kumoai import global_state
from kumoai.client.jobs import BaselineJobID
from kumoai.graph import Graph
from kumoai.pquery.training_table import TrainingTable, TrainingTableJob
from kumoai.trainer.job import BaselineJob, BaselineJobResult


class BaselineTrainer:
    r"""A baseline trainer supports creating a Kumo baseline model on a
    :class:`~kumoai.pquery.PredictiveQuery`. It is primarily oriented around
    :meth:`~kumoai.trainer.Trainer.run`, which accepts a
    :class:`~kumoai.graph.Graph` and :class:`~kumoai.pquery.TrainingTable` and
    produces a :class:`~kumoai.trainer.BaselineJobResult`.

    Args:
        metrics List[str]: A list to metrics that baseline model will be
            evaluated on.

    Example:
        >>> import kumoai  # doctest: +SKIP
        >>> pquery = kumoai.PredictiveQuery(...)  # doctest: +SKIP
        >>> trainer = kumoai.BaselineTrainer(metrics=metrics)  # doctest: +SKIP

    .. # noqa: E501
    """
    def __init__(self, metrics: List[str]) -> None:
        self._metrics: List[str] = metrics

        # Cached from backend:
        self._baseline_job_id: Optional[BaselineJobID] = None

    def run(
        self,
        graph: Graph,
        train_table: Union[TrainingTable, TrainingTableJob],
        *,
        non_blocking: bool = False,
        custom_tags: Mapping[str, str] = {},
    ) -> Union[BaselineJob, BaselineJobResult]:
        """Runs a baseline to the specified graph and training table.

        Args:
            graph (Graph): The :class:`~kumoai.graph.Graph` object that
                represents the tables and relationships that baseline model
                is running against.
            train_table (Union[TrainingTable, TrainingTableJob]): The
                :class:`~kumoai.pquery.TrainingTable`, or in-progress
                :class:`~kumoai.pquery.TrainingTableJob` that represents
                the training data produced by a
                :class:`~kumoai.pquery.PredictiveQuery` on :obj:`graph`.
            non_blocking (bool): Whether this operation should return
                immediately after launching the baseline job, or await
                completion of the baseline job. Defaults to False.
            custom_tags (Mapping[str, str], optional): Customer defined k-v
                tags to be associated with the job to be launched. Job tags
                are useful for grouping and searching jobs.. Defaults to {}.

        Returns:
            Union[BaselineJob, BaselineJobResult]:
                If ``non_blocking=False``, returns a baseline job object. If
                ``non_blocking=True``, returns a baseline job future object.
        """
        job_id = train_table.job_id
        assert job_id is not None

        train_table_job_api = global_state.client.generate_train_table_job_api
        pq_id = train_table_job_api.get(job_id).config.pquery_id
        assert pq_id is not None

        # NOTE the backend implementation currently handles sequentialization
        # between a training table future and a baseline job; that is, if the
        # training table future is still executing, the backend will wait on
        # the job ID completion before executing a baseline job. This preserves
        # semantics for both futures, ensures that Kumo works as expected if
        # used only via REST API, and allows us to avoid chaining calllbacks
        # in an ugly way here:
        api = global_state.client.baseline_job_api
        self._baseline_job_id = api.create(
            BaselineJobRequest(
                job_tags=dict(custom_tags),
                pquery_id=pq_id,
                metrics=self._metrics,
                graph_snapshot_id=graph.snapshot(non_blocking=non_blocking),
                train_table_job_id=job_id,
            ))
        out = BaselineJob(job_id=self._baseline_job_id)
        if non_blocking:
            return out
        return out.attach()
