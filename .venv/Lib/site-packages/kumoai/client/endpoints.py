from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Optional


class HTTPMethod(Enum):
    r"""HTTP methods supported by the API."""
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass(frozen=True)
class Endpoint:
    r"""Represents an API endpoint with its path and HTTP method."""
    path: Optional[str] = field(default=None)
    method: HTTPMethod = HTTPMethod.GET

    def validate(self) -> None:
        pass

    def get_path(self) -> str:
        if self.path is None:
            # This should be in validate but is here for type checker
            raise ValueError("Endpoint requires a path")
        return self.path


@dataclass(frozen=True)
class IDEndpoint(Endpoint):
    r"""Represents an API endpoint with an additional id/name in its path."""
    template_path: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        if self.template_path is None:
            raise ValueError("template_path must be set explicitly")
        if "{id}" not in self.template_path:
            raise ValueError("IDEndpoint path must contain '{{id}}': "
                             f"got '{self.template_path}'")

    def with_id(self, resource_id: str) -> 'IDEndpoint':
        assert self.template_path is not None
        return IDEndpoint(template_path=self.template_path,
                          path=self.template_path.format(id=resource_id),
                          method=self.method)

    def validate(self) -> None:
        if self.path is None:
            raise ValueError(
                "IDEndpoint requires with_id() to be called with a resource id"
            )

    @classmethod
    def from_base(cls, base: str, method: HTTPMethod) -> 'IDEndpoint':
        r"""Alternate constructor to build a template_path by appending /{id}
        to base.
        """
        template = f"{base}/{{id}}"
        return cls(template_path=template, method=method)


class ConnectorEndpoints:
    BASE: Final[str] = "/connectors"

    create = Endpoint(BASE, HTTPMethod.POST)
    list = Endpoint(BASE, HTTPMethod.GET)
    get = IDEndpoint.from_base(BASE, HTTPMethod.GET)
    delete = IDEndpoint.from_base(BASE, HTTPMethod.DELETE)

    start_file_upload = Endpoint(f"{BASE}/utils/start_file_upload",
                                 HTTPMethod.POST)
    delete_uploaded_file = Endpoint(f"{BASE}/utils/delete_uploaded_file",
                                    HTTPMethod.POST)
    complete_file_upload = Endpoint(f"{BASE}/utils/complete_file_upload",
                                    HTTPMethod.POST)


class PQueryEndpoints:
    BASE: Final[str] = '/predictive_queries'

    create = Endpoint(BASE, HTTPMethod.POST)
    list = Endpoint(BASE, HTTPMethod.GET)
    get = IDEndpoint.from_base(BASE, HTTPMethod.GET)
    delete = IDEndpoint.from_base(BASE, HTTPMethod.DELETE)

    infer_task_type = Endpoint(f"{BASE}/infer_task_type", HTTPMethod.POST)
    validate = Endpoint(f"{BASE}/validate", HTTPMethod.POST)
    suggest_training_table_plan = Endpoint(
        f"{BASE}/train_table_generation_plan", HTTPMethod.POST)
    suggest_model_plan = Endpoint(f"{BASE}/model_plan", HTTPMethod.POST)


class SourceTableEndpoints:
    BASE: Final[str] = "/source_tables"

    get = IDEndpoint.from_base(BASE, HTTPMethod.GET)

    list_tables = Endpoint(f"{BASE}/list_tables", HTTPMethod.POST)
    validate_table = Endpoint(f"{BASE}/validate_table", HTTPMethod.POST)
    get_table_data = Endpoint(f"{BASE}/get_table_data", HTTPMethod.POST)
    get_table_config = Endpoint(f"{BASE}/get_table_config", HTTPMethod.POST)


class GraphEndpoints:
    BASE: Final[str] = "/graphs"
    SNAPSHOTS: Final[str] = "/graphsnapshots"

    create = Endpoint(BASE, HTTPMethod.POST)
    get = IDEndpoint.from_base(BASE, HTTPMethod.GET)

    validate = Endpoint(f"{BASE}/validate", HTTPMethod.POST)
    infer_links = Endpoint(f"{BASE}/infer_links", HTTPMethod.POST)
    create_snapshot = Endpoint(SNAPSHOTS, HTTPMethod.POST)
    get_snapshot = IDEndpoint.from_base(SNAPSHOTS, HTTPMethod.GET)
    get_edge_stats = IDEndpoint.from_base(f"{SNAPSHOTS}/edge_health",
                                          method=HTTPMethod.GET)


class OnlineServingEndpoints:
    BASE: Final[str] = "/online_serving_endpoints"

    create = Endpoint(BASE, HTTPMethod.POST)
    get = IDEndpoint.from_base(BASE, HTTPMethod.GET)
    list = Endpoint(BASE, HTTPMethod.GET)
    update = IDEndpoint.from_base(BASE, HTTPMethod.PATCH)
    delete = IDEndpoint.from_base(BASE, HTTPMethod.DELETE)


class TableEndpoints:
    BASE: Final[str] = "/tables"
    SNAPSHOTS: Final[str] = "/tablesnapshots"

    create = Endpoint(BASE, HTTPMethod.POST)
    get = IDEndpoint.from_base(BASE, HTTPMethod.GET)

    create_snapshot = Endpoint(SNAPSHOTS, HTTPMethod.POST)
    get_snapshot = IDEndpoint.from_base(SNAPSHOTS, HTTPMethod.GET)
    validate = Endpoint(f"{BASE}/validate", HTTPMethod.POST)
    infer_metadata = Endpoint(f"{BASE}/infer_metadata", HTTPMethod.POST)


class RFMEndpoints:
    BASE: Final[str] = "/rfm"

    predict = Endpoint(f"{BASE}/predict", HTTPMethod.POST)
    explain = Endpoint(f"{BASE}/explain", HTTPMethod.POST)
    evaluate = Endpoint(f"{BASE}/evaluate", HTTPMethod.POST)
    validate_query = Endpoint(f"{BASE}/validate_query", HTTPMethod.POST)
    parse_query = Endpoint(f"{BASE}/parse_query", HTTPMethod.POST)
