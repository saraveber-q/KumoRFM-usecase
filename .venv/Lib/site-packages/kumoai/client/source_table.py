from typing import List

from kumoapi.json_serde import to_json_dict
from kumoapi.source_table import (
    SourceTableConfigRequest,
    SourceTableConfigResponse,
    SourceTableDataRequest,
    SourceTableDataResponse,
    SourceTableListRequest,
    SourceTableListResponse,
    SourceTableValidateRequest,
    SourceTableValidateResponse,
)

from kumoai.client import KumoClient
from kumoai.client.endpoints import SourceTableEndpoints
from kumoai.client.utils import parse_response, raise_on_error


class SourceTableAPI:
    r"""Typed API definition for Kumo source tables."""
    def __init__(self, client: KumoClient) -> None:
        self._client = client

    def validate_table(
            self, request: SourceTableValidateRequest
    ) -> SourceTableValidateResponse:
        response = self._client._request(SourceTableEndpoints.validate_table,
                                         json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(SourceTableValidateResponse, response)

    def list_tables(
            self, request: SourceTableListRequest) -> SourceTableListResponse:
        response = self._client._request(SourceTableEndpoints.list_tables,
                                         json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(SourceTableListResponse, response)

    def get_table_data(
            self, request: SourceTableDataRequest) -> SourceTableDataResponse:
        response = self._client._request(SourceTableEndpoints.get_table_data,
                                         json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(List[SourceTableDataResponse], response)

    def get_table_config(
            self,
            request: SourceTableConfigRequest) -> SourceTableConfigResponse:
        response = self._client._request(SourceTableEndpoints.get_table_config,
                                         json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(SourceTableConfigResponse, response)
