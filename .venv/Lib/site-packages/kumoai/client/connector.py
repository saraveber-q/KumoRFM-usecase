from http import HTTPStatus
from typing import List, Optional

from kumoapi.data_source import (
    CompleteFileUploadRequest,
    ConnectorResponse,
    CreateConnectorArgs,
    DataSourceType,
    DeleteUploadedFileRequest,
    StartFileUploadRequest,
    StartFileUploadResponse,
)
from kumoapi.json_serde import to_json_dict

from kumoai.client import KumoClient
from kumoai.client.endpoints import ConnectorEndpoints
from kumoai.client.utils import parse_response, raise_on_error
from kumoai.exceptions import HTTPException


class ConnectorAPI:
    r"""Typed API definition for Kumo connectors."""
    def __init__(self, client: KumoClient) -> None:
        self._client = client

    def start_file_upload(
            self, req: StartFileUploadRequest) -> StartFileUploadResponse:
        res = self._client._request(
            ConnectorEndpoints.start_file_upload,
            json=to_json_dict(req, insecure=True),
        )
        raise_on_error(res)
        return parse_response(StartFileUploadResponse, res)

    def delete_file_upload(self, req: DeleteUploadedFileRequest) -> None:
        res = self._client._request(
            ConnectorEndpoints.delete_uploaded_file,
            json=to_json_dict(req, insecure=True),
        )
        raise_on_error(res)

    def complete_file_upload(self, req: CompleteFileUploadRequest) -> None:
        res = self._client._request(
            ConnectorEndpoints.complete_file_upload,
            json=to_json_dict(req, insecure=True),
        )
        raise_on_error(res)

    def create(self, create_connector_args: CreateConnectorArgs) -> None:
        r"""Creates a connector in Kumo."""
        resp = self._client._request(
            ConnectorEndpoints.create,
            json=to_json_dict(create_connector_args, insecure=True),
        )
        raise_on_error(resp)

    def create_if_not_exist(
        self,
        create_connector_args: CreateConnectorArgs,
    ) -> bool:
        r"""Creates a connector in Kumo if the connector does not exist.

        Returns:
            :obj:`True` if the connector is newly created, :obj:`False`
            otherwise.
        """
        _id = create_connector_args.config.name
        existing_connector = self.get(_id)
        if existing_connector:
            if existing_connector.config == create_connector_args.config:
                return False
            raise HTTPException(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                f"Connector {_id} already exists, but has a differing "
                f"configuration. Input: {create_connector_args.config}, "
                f"existing: {existing_connector.config}",
            )
        self.create(create_connector_args)
        return existing_connector is None

    def get(self, connector_id: str) -> Optional[ConnectorResponse]:
        r"""Fetches a connector given its ID."""
        resp = self._client._request(
            ConnectorEndpoints.get.with_id(connector_id))
        if resp.status_code == HTTPStatus.NOT_FOUND:
            return None

        raise_on_error(resp)
        return parse_response(ConnectorResponse, resp)

    def list(
        self, data_source_type: Optional[DataSourceType] = None
    ) -> List[ConnectorResponse]:
        r"""Lists connectors for a given data source type."""
        params = {
            'data_source_type': data_source_type
        } if data_source_type else {}

        resp = self._client._request(ConnectorEndpoints.list, params=params)
        raise_on_error(resp)
        return parse_response(List[ConnectorResponse], resp)

    def delete_if_exists(self, connector_id: str) -> bool:
        r"""Deletes a connector if it exists."""
        resp = self._client._request(
            ConnectorEndpoints.delete.with_id(connector_id))
        if resp.status_code == HTTPStatus.NOT_FOUND:
            return False
        raise_on_error(resp)
        return True
