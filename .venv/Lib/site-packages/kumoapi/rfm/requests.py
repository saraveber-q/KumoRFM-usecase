from typing import Any, Dict, List, Optional

from pydantic.dataclasses import dataclass
from typing_extensions import Self

from kumoapi.common import ValidationResponse
from kumoapi.graph import GraphDefinition
from kumoapi.model_plan import RunMode
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.rfm import Context, Explanation, PQueryDefinition


@dataclass
class RFMValidateQueryRequest:
    query: str
    graph_definition: GraphDefinition


@dataclass
class RFMValidateQueryResponse:
    query_definition: PQueryDefinition
    validation_response: ValidationResponse


@dataclass
class RFMParseQueryRequest:
    query: str
    graph_definition: GraphDefinition


@dataclass
class RFMParseQueryResponse:
    query: ValidatedPredictiveQuery
    validation_response: ValidationResponse


@dataclass
class RFMPredictRequest:
    context: Context
    run_mode: RunMode
    use_prediction_time: bool = False
    query: str = ""

    def to_protobuf(self) -> Any:
        import kumoapi.rfm.protos.request_pb2 as _request_pb2

        request_pb2: Any = _request_pb2

        msg = request_pb2.PredictRequest()
        self.context.fill_protobuf_(msg.context)
        msg.run_mode = getattr(request_pb2.RunMode, self.run_mode.upper())
        msg.use_prediction_time = self.use_prediction_time
        msg.query = self.query

        return msg

    def serialize(self) -> bytes:
        import kumoapi.rfm.protos.request_pb2 as _request_pb2

        request_pb2: Any = _request_pb2

        msg = request_pb2.PredictRequest()
        self.context.fill_protobuf_(msg.context)
        msg.run_mode = getattr(request_pb2.RunMode, self.run_mode.upper())
        msg.use_prediction_time = self.use_prediction_time
        msg.query = self.query

        return self.to_protobuf().SerializeToString()

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        import kumoapi.rfm.protos.request_pb2 as _request_pb2

        request_pb2: Any = _request_pb2

        msg = request_pb2.PredictRequest()
        msg.ParseFromString(data)

        return cls(
            context=Context.from_protobuf(msg.context),
            run_mode=RunMode(request_pb2.RunMode.Name(msg.run_mode).lower()),
            use_prediction_time=bool(msg.use_prediction_time),
            query=str(msg.query),
        )


@dataclass
class RFMExplanationResponse:
    prediction: dict[str, Any]
    summary: str
    details: Explanation


@dataclass
class RFMPredictResponse:
    prediction: dict[str, Any]


@dataclass
class RFMEvaluateRequest:
    context: Context
    run_mode: RunMode
    metrics: Optional[List[str]] = None
    use_prediction_time: bool = False

    def __post_init__(self) -> None:
        if self.metrics is not None and len(self.metrics) == 0:
            self.metrics = None

    def to_protobuf(self) -> Any:
        import kumoapi.rfm.protos.request_pb2 as _request_pb2

        request_pb2: Any = _request_pb2

        msg = request_pb2.EvaluateRequest()
        self.context.fill_protobuf_(msg.context)
        msg.run_mode = getattr(request_pb2.RunMode, self.run_mode.upper())
        if self.metrics is not None:
            msg.metrics.extend(self.metrics)
        msg.use_prediction_time = self.use_prediction_time

        return msg

    def serialize(self) -> bytes:
        return self.to_protobuf().SerializeToString()

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        import kumoapi.rfm.protos.request_pb2 as _request_pb2

        request_pb2: Any = _request_pb2

        msg = request_pb2.EvaluateRequest()
        msg.ParseFromString(data)

        return cls(
            context=Context.from_protobuf(msg.context),
            run_mode=RunMode(request_pb2.RunMode.Name(msg.run_mode).lower()),
            metrics=list(msg.metrics),
            use_prediction_time=bool(msg.use_prediction_time),
        )


@dataclass
class RFMEvaluateResponse:
    metrics: Dict[str, Optional[float]]


@dataclass
class ListPromptsResponse:
    prompts: list[str]


@dataclass
class PromptContentResponse:
    name: str
    content: str
