from http import HTTPStatus
from typing import Any, Dict, Optional

from kumoapi.common import ValidationResponse
from kumoapi.data_snapshot import TableSnapshotID, TableSnapshotResource
from kumoapi.json_serde import to_json_dict
from kumoapi.table import (
    TableDefinition,
    TableMetadataRequest,
    TableMetadataResponse,
    TableResource,
    TableValidationRequest,
)

from kumoai.client import KumoClient
from kumoai.client.endpoints import TableEndpoints
from kumoai.client.utils import (
    parse_id_response,
    parse_response,
    raise_on_error,
)

TableID = str


class TableAPI:
    r"""Typed API definition for Kumo table definition."""
    def __init__(self, client: KumoClient) -> None:
        self._client = client

    def create_table(
        self,
        table_def: TableDefinition,
        *,
        name_alias: Optional[str] = None,
        force_rename: bool = False,
    ) -> TableID:
        r"""Creates a Table (metadata definition) resource object in Kumo."""
        params: Dict[str, Any] = {'force_rename': force_rename}
        if name_alias:
            params['name_alias'] = name_alias
        resp = self._client._request(
            TableEndpoints.create,
            params=params,
            json=to_json_dict(table_def),
        )
        raise_on_error(resp)
        return parse_id_response(resp)

    def get_table_if_exists(
        self,
        table_id_or_name: str,
    ) -> Optional[TableResource]:
        r"""Fetches a connector given its ID."""
        resp = self._client._request(
            TableEndpoints.get.with_id(table_id_or_name))
        if resp.status_code == HTTPStatus.NOT_FOUND:
            return None

        raise_on_error(resp)
        return parse_response(TableResource, resp)

    def create_snapshot(
        self,
        table_definition: TableDefinition,
        *,
        refresh_source: bool = False,
    ) -> TableSnapshotID:
        params: Dict[str, Any] = {'refresh_source': refresh_source}
        resp = self._client._request(TableEndpoints.create_snapshot,
                                     params=params,
                                     json=to_json_dict(table_definition))
        raise_on_error(resp)
        return parse_id_response(resp)

    def get_snapshot(
        self,
        snapshot_id: TableSnapshotID,
    ) -> TableSnapshotResource:
        resp = self._client._request(
            TableEndpoints.get_snapshot.with_id(snapshot_id))
        raise_on_error(resp)
        return parse_response(TableSnapshotResource, resp)

    def infer_metadata(
        self,
        request: TableMetadataRequest,
    ) -> TableMetadataResponse:
        response = self._client._request(TableEndpoints.infer_metadata,
                                         json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(TableMetadataResponse, response)

    def validate_table(
        self,
        request: TableValidationRequest,
    ) -> ValidationResponse:
        response = self._client._request(TableEndpoints.validate,
                                         json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(ValidationResponse, response)
