from typing import TYPE_CHECKING, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from kumoai.client.endpoints import Endpoint, HTTPMethod

if TYPE_CHECKING:
    from kumoai.client.connector import ConnectorAPI
    from kumoai.client.graph import GraphAPI
    from kumoai.client.jobs import (
        ArtifactExportJobAPI,
        BaselineJobAPI,
        BatchPredictionJobAPI,
        GeneratePredictionTableJobAPI,
        GenerateTrainTableJobAPI,
        LLMJobAPI,
        TrainingJobAPI,
    )
    from kumoai.client.online import OnlineServingEndpointAPI
    from kumoai.client.pquery import PQueryAPI
    from kumoai.client.source_table import SourceTableAPI
    from kumoai.client.table import TableAPI

API_VERSION = 'v1'


class KumoClient:
    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        spcs_token: Optional[str] = None,
        verify_ssl: bool = True,
    ) -> None:
        r"""Creates a client against the Kumo public API, provided a URL of
        the endpoint and an authentication token.

        Args:
            url: the public API endpoint URL.
            api_key: the public API authentication token.
            spcs_token: the SPCS token used for authentication to access the
                Kumo API endpoint.
            verify_ssl: whether to verify SSL certificates. Set to False to
                skip SSL certificate verification (equivalent to curl -k).
        """
        self._url = url
        self._api_url = f"{url}/{API_VERSION}"
        self._api_key = api_key
        self._spcs_token = spcs_token
        self._verify_ssl = verify_ssl

        retry_strategy = Retry(
            total=10,  # Maximum number of retries
            connect=3,  # How many connection-related errors to retry on
            read=3,  # How many times to retry on read errors
            status=5,  # How many times to retry on bad status codes (below)
            # Status codes to retry on.
            status_forcelist=[408, 429, 500, 502, 503, 504],
            # Exponential backoff factor: 2, 4, 8, 16 seconds delay)
            backoff_factor=2.0,
        )
        http_adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount('http://', http_adapter)
        session.mount('https://', http_adapter)
        self._session = session
        if self._api_key:
            self._session.headers.update({"X-API-Key": self._api_key})
        elif self._spcs_token:
            self._session.headers.update(
                {'Authorization': f'Snowflake Token={self._spcs_token}'})

    def authenticate(self) -> None:
        """Raises an exception if authentication fails."""
        try:
            self._session.get(self._url + '/v1/connectors',
                              verify=self._verify_ssl).raise_for_status()
        except Exception:
            raise ValueError(
                "Client authentication failed. Please check if you "
                "have a valid API key/credentials.")

    def set_spcs_token(self, spcs_token: str) -> None:
        r"""Sets the SPCS token for the client and updates the session
        headers.
        """
        self._spcs_token = spcs_token
        self._session.headers.update(
            {'Authorization': f'Snowflake Token={self._spcs_token}'})

    @property
    def artifact_export_api(self) -> 'ArtifactExportJobAPI':
        r"""Returns the artifact export API."""
        from kumoai.client.jobs import ArtifactExportJobAPI
        return ArtifactExportJobAPI(self)

    @property
    def connector_api(self) -> 'ConnectorAPI':
        r"""Returns the typed connector API."""
        from kumoai.client.connector import ConnectorAPI
        return ConnectorAPI(self)

    @property
    def source_table_api(self) -> 'SourceTableAPI':
        r"""Returns the typed source table API."""
        from kumoai.client.source_table import SourceTableAPI
        return SourceTableAPI(self)

    @property
    def table_api(self) -> 'TableAPI':
        r"""Returns the typed Kumo Table (snapshot) API."""
        from kumoai.client.table import TableAPI
        return TableAPI(self)

    @property
    def graph_api(self) -> 'GraphAPI':
        r"""Returns the typed Graph (metadata and snapshot) API."""
        from kumoai.client.graph import GraphAPI
        return GraphAPI(self)

    @property
    def pquery_api(self) -> 'PQueryAPI':
        r"""Returns the typed Predictive Query API."""
        from kumoai.client.pquery import PQueryAPI
        return PQueryAPI(self)

    @property
    def training_job_api(self) -> 'TrainingJobAPI':
        r"""Returns the typed Training Job API."""
        from kumoai.client.jobs import TrainingJobAPI
        return TrainingJobAPI(self)

    @property
    def batch_prediction_job_api(self) -> 'BatchPredictionJobAPI':
        from kumoai.client.jobs import BatchPredictionJobAPI
        return BatchPredictionJobAPI(self)

    @property
    def generate_train_table_job_api(self) -> 'GenerateTrainTableJobAPI':
        r"""Returns the typed Generate-Train-Table Job API."""
        from kumoai.client.jobs import GenerateTrainTableJobAPI
        return GenerateTrainTableJobAPI(self)

    @property
    def generate_prediction_table_job_api(
            self) -> 'GeneratePredictionTableJobAPI':
        from kumoai.client.jobs import GeneratePredictionTableJobAPI
        return GeneratePredictionTableJobAPI(self)

    @property
    def llm_job_api(self) -> 'LLMJobAPI':
        from kumoai.client.jobs import LLMJobAPI
        return LLMJobAPI(self)

    @property
    def baseline_job_api(self) -> 'BaselineJobAPI':
        r"""Returns the typed Training Job API."""
        from kumoai.client.jobs import BaselineJobAPI
        return BaselineJobAPI(self)

    @property
    def online_serving_endpoint_api(self) -> 'OnlineServingEndpointAPI':
        from kumoai.client.online import OnlineServingEndpointAPI
        return OnlineServingEndpointAPI(self)

    def _request(self, endpoint: Endpoint, **kwargs: Any) -> requests.Response:
        r"""Send a HTTP request to the specified endpoint."""
        endpoint_str = endpoint.get_path()
        if endpoint.method == HTTPMethod.GET:
            return self._get(endpoint_str, **kwargs)
        elif endpoint.method == HTTPMethod.POST:
            return self._post(endpoint_str, **kwargs)
        elif endpoint.method == HTTPMethod.PATCH:
            return self._patch(endpoint_str, **kwargs)
        elif endpoint.method == HTTPMethod.DELETE:
            return self._delete(endpoint_str, **kwargs)
        else:
            raise ValueError(f"Unsupported HTTP method: {endpoint.method}")

    def _get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        r"""Send a GET request to the specified endpoint, with keyword
        arguments, returned objects, and exceptions raised corresponding to
        :meth:`requests.Session.get`.
        """
        url = self._format_endpoint_url(endpoint)
        return self._session.get(url=url, verify=self._verify_ssl, **kwargs)

    def _post(self, endpoint: str, **kwargs: Any) -> requests.Response:
        r"""Send a POST request to the specified endpoint, with keyword
        arguments, returned objects, and exceptions raised corresponding to
        :meth:`requests.Session.post`.
        """
        url = self._format_endpoint_url(endpoint)
        return self._session.post(url=url, verify=self._verify_ssl, **kwargs)

    def _patch(self, endpoint: str, **kwargs: Any) -> requests.Response:
        r"""Send a PATCH request to the specified endpoint, with keyword
        arguments, returned objects, and exceptions raised corresponding to
        :meth:`requests.Session.patch`.
        """
        url = self._format_endpoint_url(endpoint)
        return self._session.patch(url=url, verify=self._verify_ssl, **kwargs)

    def _delete(self, endpoint: str, **kwargs: Any) -> requests.Response:
        r"""Send a DELETE request to the specified endpoint, with keyword
        arguments, returned objects, and exceptions raised corresponding to
        :meth:`requests.Session.delete`.
        """
        url = self._format_endpoint_url(endpoint)
        return self._session.delete(url=url, verify=self._verify_ssl, **kwargs)

    def _format_endpoint_url(self, endpoint: str) -> str:
        if endpoint[0] == "/":
            endpoint = endpoint[1:]
        return f"{self._api_url}/{endpoint}"
