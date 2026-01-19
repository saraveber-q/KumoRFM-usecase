import copy
from dataclasses import fields, make_dataclass
from http import HTTPStatus
from typing import List, Optional, Tuple, Union, get_args, get_origin

from kumoapi.common import ValidationResponse
from kumoapi.json_serde import to_json_dict
from kumoapi.model_plan import (
    MissingType,
    ModelPlan,
    ModelPlanInfo,
    PlanMixin,
    SuggestModelPlanRequest,
    TrainingTableGenerationPlan,
)
from kumoapi.pquery import PQueryResource, QueryType
from kumoapi.task import TaskType
from kumoapi.typing import WITH_PYDANTIC_V2
from pydantic.dataclasses import dataclass

from kumoai.client import KumoClient
from kumoai.client.endpoints import PQueryEndpoints
from kumoai.client.utils import (
    Returns,
    parse_id_response,
    parse_response,
    raise_on_error,
)

GraphID = str
PQueryID = str


class PQueryAPI:
    r"""Typed API definition for Kumo Predictive Query definition."""
    def __init__(self, client: KumoClient) -> None:
        self._client = client

    def create(self, pquery: PQueryResource) -> PQueryID:
        r"""Creates a Predictive Query resource object in Kumo."""
        resp = self._client._request(PQueryEndpoints.create,
                                     json=to_json_dict(pquery))
        raise_on_error(resp)
        return parse_id_response(resp)

    def get_if_exists(self, name_or_id: str) -> Optional[PQueryResource]:
        r"""Fetches a Predictive Query given its name(id)."""
        resp = self._client._request(PQueryEndpoints.get.with_id(name_or_id))
        if resp.status_code == HTTPStatus.NOT_FOUND:
            return None

        raise_on_error(resp)
        return parse_response(PQueryResource, resp)

    def list(self, *,
             name_pattern: Optional[str] = None) -> List[PQueryResource]:
        params = {}
        if name_pattern:
            params['name_pattern'] = name_pattern
        resp = self._client._request(PQueryEndpoints.list, params=params)
        raise_on_error(resp)
        return parse_response(List[PQueryResource], resp)

    def delete(self, name: str) -> None:
        resp = self._client._request(PQueryEndpoints.delete.with_id(name))
        raise_on_error(resp)

    def infer_task_type(
        self,
        pquery_string: str,
        graph_id: GraphID,
    ) -> Tuple[TaskType, str]:
        resp = self._client._request(
            PQueryEndpoints.infer_task_type,
            params={
                'pquery_string': pquery_string,
                'graph_id': graph_id
            },
        )
        raise_on_error(resp)
        return parse_response(Returns[Tuple[TaskType, str]], resp)

    def validate(self, pquery: PQueryResource) -> ValidationResponse:
        r"""Validates a Predictive Query resource object in Kumo."""
        resp = self._client._request(PQueryEndpoints.validate,
                                     json=to_json_dict(pquery))
        raise_on_error(resp)
        return parse_response(Returns[ValidationResponse], resp)

    def suggest_training_table_plan(
        self,
        req: SuggestModelPlanRequest,
    ) -> TrainingTableGenerationPlan:
        resp = self._client._request(
            PQueryEndpoints.suggest_training_table_plan,
            json=to_json_dict(req),
        )
        raise_on_error(resp)
        return parse_response(TrainingTableGenerationPlan, resp)

    def suggest_model_plan(
        self,
        req: SuggestModelPlanRequest,
    ) -> ModelPlan:
        resp = self._client._request(
            PQueryEndpoints.suggest_model_plan,
            json=to_json_dict(req),
        )
        raise_on_error(resp)
        plan_info = parse_response(ModelPlanInfo, resp)

        return filter_model_plan(
            plan_info.model_plan,
            plan_info.task_type,
            plan_info.query_type,
            plan_info.has_train_table_weight_col,
        )


def filter_model_plan(
    plan: ModelPlan,
    task_type: TaskType,
    query_type: QueryType,
    has_train_table_weight_col: bool,
) -> ModelPlan:
    r"""Filters the model plan ``plan`` down to its valid fields only, given
    ``task_type`` and ``query_type``.
    """
    new_section_fields = []
    new_sections = []
    for section_field in fields(plan):
        section = getattr(plan, section_field.name)

        new_opt_fields = []
        new_opts = []
        for field in fields(section):
            if WITH_PYDANTIC_V2:
                metadata = field.default.json_schema_extra  # type: ignore
            else:
                metadata = field.default.extra['metadata']  # type: ignore
            if not section.is_valid_option(
                    name=field.name,
                    metadata=metadata,
                    task_type=task_type,
                    query_type=query_type,
                    has_train_table_weight_col=has_train_table_weight_col,
            ):
                continue

            # Remove `MissingType` from type annotation:
            if WITH_PYDANTIC_V2:
                hidden = metadata["hidden"]
            else:
                hidden = metadata.hidden
            if not hidden and get_origin(field.type) is Union:
                _types = tuple(_type for _type in get_args(field.type)
                               if _type is not MissingType)
                assert len(_types) > 0
                _type = Union[_types] if len(_types) > 1 else _types[0]
            else:
                _type = field.type

            default = copy.copy(field.default)
            if not WITH_PYDANTIC_V2:
                # Pydantic v1
                from pydantic.fields import Undefined
                default.default = Undefined  # type: ignore
            else:
                try:
                    # Pydantic v2 - Undefined moved to pydantic_core
                    from pydantic_core import PydanticUndefined
                    default.default = PydanticUndefined  # type: ignore
                except ImportError:
                    # Fallback: Neither v1 nor v2 Undefined available
                    # In Pydantic v2, not setting a default is equivalent to
                    # Undefined
                    pass

            # Forward compatibility - Remove any newly introduced arguments not
            # returned yet by the backend:
            value = getattr(section, field.name)
            if value != MissingType.VALUE:
                new_opt_fields.append((field.name, _type, default))
                new_opts.append(value)

        Section = dataclass(
            config=dict(validate_assignment=True),
            repr=False,
        )(make_dataclass(
            type(section).__name__,
            new_opt_fields,
            bases=(PlanMixin, ),
            repr=False,
        ))
        new_section_fields.append((section_field.name, Section))
        new_sections.append(Section(*new_opts))

    Plan = dataclass(
        config=dict(validate_assignment=True),
        repr=False,
    )(make_dataclass(
        'ModelPlan',
        new_section_fields,
        bases=(ModelPlan, ),
        repr=False,
    ))
    return Plan(*new_sections)
