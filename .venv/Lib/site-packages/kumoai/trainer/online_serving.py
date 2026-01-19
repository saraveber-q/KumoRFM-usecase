import asyncio
import concurrent
import concurrent.futures
import logging
from datetime import datetime, timezone
from typing import Optional, Union

from kumoapi.json_serde import to_json_dict
from kumoapi.online_serving import (
    NodeId,
    OnlinePredictionRequest,
    OnlinePredictionResponse,
    OnlinePredictionResult,
    OnlineServingEndpointRequest,
    OnlineServingEndpointResource,
    OnlineServingStatus,
    OnlineServingStatusCode,
    OnlineServingUpdate,
    RealtimeFeatures,
    TimestampNanos,
)
from typing_extensions import override

from kumoai import global_state
from kumoai.client.jobs import TrainingJobID
from kumoai.client.online import (
    OnlineServingEndpointAPI,
    OnlineServingEndpointID,
)
from kumoai.client.utils import parse_response
from kumoai.futures import KumoFuture, create_future
from kumoai.graph.graph import Graph

logger = logging.getLogger(__name__)


class OnlineServingEndpoint:
    """Represents a Kumo online serving endpoint that serves online `predict`
    requests.
    """
    def __init__(self, endpoint_url: str):
        self._endpoint_url = endpoint_url
        # Use the same global session with API key in header.
        self._session = global_state.client._session
        self._endpoint_id = self._endpoint_url.split('/')[-1]
        self._predict_url = f'{endpoint_url}/predict'
        logger.info('Initialized OnlineServingEndpoint at: %s', endpoint_url)

    def predict(
        self,
        fkey: NodeId,
        *,
        time: Union[datetime, TimestampNanos, None] = None,
        realtime_features: Optional[RealtimeFeatures] = None,
    ) -> OnlinePredictionResult:
        """Performs online inference for a single entity key using the
        currently deployed model and feature graph.

        This method sends a low-latency prediction request to the live
        endpoint. It supports injecting optional
        real-time features and controlling the anchor time used for temporal
        feature lookups.

        Parameters:
            fkey (NodeId):
                The entity key (e.g., user ID, item ID) to run inference on.

            time (datetime | TimestampNanos | None, optional):
                The effective timestamp for feature lookup and model
                prediction.  If not provided, the current server time will be
                used.

            realtime_features (Optional[RealtimeFeatures], optional):
                Additional real-time features to inject into the feature graph
                for this prediction request.
                These can complement batch-generated features, useful for
                contextual signals like current session state, real-time data,
                etc.

        Returns: The prediction result from the deployed model. The return type
        is a union type depending on the model task type.
        """
        timestamp_nanos = time
        if isinstance(time, datetime):
            timestamp_nanos = int(time.timestamp() * 10**9)
        resp = self._session.post(
            self._predict_url, json=to_json_dict(
                OnlinePredictionRequest(fkey, timestamp_nanos,
                                        realtime_features)))
        resp.raise_for_status()
        return parse_response(OnlinePredictionResponse, resp).result

    def ping(self) -> str:
        resp = self._session.get(f'{self._endpoint_url}/probe_liveness')
        resp.raise_for_status()
        return resp.text

    def update(
        self,
        *,
        refresh_graph_data: bool = True,
        graph_override: Optional[Graph] = None,
        new_model_id: Optional[TrainingJobID] = None,
    ) -> 'OnlineServingEndpointUpdateFuture':
        """Triggers an asynchronous update to the online serving endpoint using
        a blue-green deployment strategy.

        This method allows clients to deploy a new version of the endpoint with
        updated model weights, refreshed feature
        data, or a complete graph override. The update is applied in the
        background without interrupting availability,
        and will swap traffic to the new deployment once it is fully ready.

        Parameters:
            refresh_graph_data (bool, optional):
                Whether to reload feature data from the latest available
                source. Defaults to True.

            graph_override (Optional[Graph], optional):
                If provided, overrides the existing feature graph with the
                given one.
                This is useful for testing or dynamic reconfiguration of the
                feature pipeline.

            new_model_id (Optional[TrainingJobID], optional):
                If specified, deploys a new model with the given training job
                ID.  This model will replace the currently serving model after
                update is complete.

        Returns:
            OnlineServingEndpointUpdateFuture:
                A future-like object that can be used to check the progress and
                result of the update operation.

        Example: (aysnchronously send email notification when update is done)
            >>> fut = endpoint.update(new_model_id="model_202504")
            >>> fut.future().add_done_callback(send_email_notification)
        """
        res = _endpoint_api().get_if_exists(self._endpoint_id)
        assert res

        if not refresh_graph_data and not new_model_id:
            raise ValueError(
                'Expect to update online endpoint by loading a new model '
                'and/or refreshed graph data.')

        model_id = new_model_id or res.config.model_training_job_id
        if refresh_graph_data:
            graph_snapshot_id = None
        if graph_override:
            graph_snapshot_id = graph_override.snapshot(
                force_refresh=refresh_graph_data, non_blocking=True)
        else:
            graph_snapshot_id = (None if refresh_graph_data else
                                 res.config.graph_snapshot_id)

        updated = _endpoint_api().update(
            self._endpoint_id,
            OnlineServingEndpointRequest(model_id, res.config.predict_options,
                                         graph_snapshot_id))

        return OnlineServingEndpointUpdateFuture(self._endpoint_id,
                                                 noop=not updated)

    def destroy(self) -> None:
        _endpoint_api().delete(self._endpoint_id)


class OnlineServingEndpointFuture(KumoFuture[OnlineServingEndpoint]):
    def __init__(self, id: OnlineServingEndpointID) -> None:
        self._id = id
        self._fut: concurrent.futures.Future[
            OnlineServingEndpoint] = create_future(_poll_endpoint_ready(id))

    @property
    def id(self) -> OnlineServingEndpointID:
        r"""The unique ID of this batch prediction job."""
        return self._id

    @override
    def result(self) -> OnlineServingEndpoint:
        return self._fut.result()

    @override
    def future(self) -> 'concurrent.futures.Future[OnlineServingEndpoint]':
        return self._fut


class OnlineServingEndpointUpdateFuture(KumoFuture[OnlineServingUpdate]):
    def __init__(self, id: OnlineServingEndpointID, noop: bool):
        if noop:
            res = _endpoint_api().get_if_exists(id)
            assert res
            fut = concurrent.futures.Future[OnlineServingEndpoint]()
            fut.set_result(
                OnlineServingUpdate(
                    prev_config=res.config, target_config=res.config,
                    update_started_at=datetime.now(timezone.utc),
                    update_status=OnlineServingStatus(
                        OnlineServingStatusCode.READY,
                        datetime.now(timezone.utc))))
        else:
            fut = create_future(_poll_update_ready(id))
        self._fut = fut

    @override
    def result(self) -> OnlineServingUpdate:
        return self._fut.result()

    @override
    def future(self) -> 'concurrent.futures.Future[OnlineServingUpdate]':
        return self._fut


async def _get_endpoint_resource(
        id: OnlineServingEndpointID) -> OnlineServingEndpointResource:
    api = global_state.client.online_serving_endpoint_api
    # TODO(manan): make asynchronous natively with aiohttp:
    res = await asyncio.get_running_loop().run_in_executor(
        None, api.get_if_exists, id)
    assert res
    return res


async def _poll_endpoint_ready(
        id: OnlineServingEndpointID) -> OnlineServingEndpoint:
    while True:
        res = await _get_endpoint_resource(id)
        status = res.status.status_code
        if status == OnlineServingStatusCode.IN_PROGRESS:
            await asyncio.sleep(10)
        else:
            break

    if status == OnlineServingStatusCode.FAILED:
        raise ValueError(f"Failed to launch online endpoint id={id}, "
                         f"failure message: {res.status.failure_message}")

    assert status == OnlineServingStatusCode.READY
    endpoint = OnlineServingEndpoint(res.endpoint_url)
    logger.info('OnlineServingEndpoint is ready, ping: %s', endpoint.ping())
    return endpoint


async def _poll_update_ready(
        id: OnlineServingEndpointID) -> OnlineServingUpdate:
    while True:
        res = await _get_endpoint_resource(id)
        if res.update.update_status == OnlineServingStatusCode.IN_PROGRESS:
            await asyncio.sleep(10)
        else:
            break

    return res.update


def _endpoint_api() -> OnlineServingEndpointAPI:
    return global_state.client.online_serving_endpoint_api
