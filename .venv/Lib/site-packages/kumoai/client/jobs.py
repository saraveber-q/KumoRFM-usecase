from datetime import datetime
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from kumoapi.common import ValidationResponse
from kumoapi.jobs import (
    ArtifactExportRequest,
    ArtifactExportResponse,
    AutoTrainerProgress,
    BaselineJobRequest,
    BaselineJobResource,
    BatchPredictionJobResource,
    BatchPredictionRequest,
    CancelBatchPredictionJobResponse,
    CancelTrainingJobResponse,
    ErrorDetails,
    GeneratePredictionTableJobResource,
    GeneratePredictionTableRequest,
    GenerateTrainTableJobResource,
    GenerateTrainTableRequest,
    GetEmbeddingsDfUrlResponse,
    GetPredictionsDfUrlResponse,
    JobRequestBase,
    JobResourceBase,
    JobStatus,
    PredictionProgress,
    TrainingJobRequest,
    TrainingJobResource,
    TrainingTableSpec,
)
from kumoapi.json_serde import from_json, to_json_dict
from kumoapi.source_table import LLMRequest, LLMResponse, SourceTableType
from kumoapi.train import TrainingStage
from typing_extensions import override

from kumoai.client import KumoClient
from kumoai.client.utils import (
    Returns,
    parse_patch_response,
    parse_response,
    raise_on_error,
)

TrainingJobID = str
BatchPredictionJobID = str
GenerateTrainTableJobID = str
GeneratePredictionTableJobID = str
LLMJobId = str
BaselineJobID = str

JobRequestType = TypeVar('JobRequestType', bound=JobRequestBase)
JobResourceType = TypeVar('JobResourceType', bound=JobResourceBase)


class CommonJobAPI(Generic[JobRequestType, JobResourceType]):
    def __init__(self, client: KumoClient, base_endpoint: str,
                 res_type: Type[JobResourceType]) -> None:
        self._client = client
        self._base_endpoint = base_endpoint
        self._res_type = res_type

    def create(self, request: JobRequestType) -> str:
        response = self._client._post(self._base_endpoint,
                                      json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(Dict[str, str], response)['id']

    def get(self, id: str) -> JobResourceType:
        response = self._client._get(f'{self._base_endpoint}/{id}')
        raise_on_error(response)
        return parse_response(self._res_type, response)

    def list(
        self,
        *,
        pquery_name: Optional[str] = None,
        pquery_id: Optional[str] = None,
        job_status: Optional[JobStatus] = None,
        limit: Optional[int] = None,
        additional_tags: Mapping[str, str] = {},
    ) -> List[JobResourceType]:
        params: Dict[str, Any] = {
            'pquery_name': pquery_name,
            'pquery_id': pquery_id,
            'job_status': job_status,
            'limit': limit
        }
        params.update(additional_tags)
        response = self._client._get(self._base_endpoint, params=params)
        raise_on_error(response)
        resource_elements = response.json()
        assert isinstance(resource_elements, list)
        return [from_json(e, self._res_type) for e in resource_elements]

    def delete_tags(self, job_id: str, tags: List[str]) -> bool:
        r"""Removes the tags from the job.

        Args:
            job_id (str): The ID of the job.
            tags (List[str]): The tags to remove.
        """
        return self.update_tags(job_id, {t: 'none' for t in tags})

    def update_tags(self, job_id: str,
                    custom_job_tags: Mapping[str, Optional[str]]) -> bool:
        r"""Updates the tags of the job.

        Args:
            job_id (str): The ID of the job.
            custom_job_tags (Mapping[str, Optional[str]]): The tags to update.
                Note that the value 'none' will remove the tag. If the tag is
                not present, it will be added.
        """
        response = self._client._patch(
            f'{self._base_endpoint}/{job_id}/tags',
            data=None,
            params={
                k: str(v)
                for k, v in custom_job_tags.items()
            },
        )
        raise_on_error(response)
        return parse_patch_response(response)


class BaselineJobAPI(CommonJobAPI[BaselineJobRequest, BaselineJobResource]):
    r"""Typed API definition for the baseline job resource."""
    def __init__(self, client: KumoClient) -> None:
        super().__init__(client, '/baseline_jobs', BaselineJobResource)

    def get_config(self, job_id: str) -> BaselineJobRequest:
        """Load the configuration for a baseline job by ID."""
        resource = self.get(job_id)
        return resource.config


class TrainingJobAPI(CommonJobAPI[TrainingJobRequest, TrainingJobResource]):
    r"""Typed API definition for the training job resource."""
    def __init__(self, client: KumoClient) -> None:
        super().__init__(client, '/training_jobs', TrainingJobResource)

    def get_progress(self, id: TrainingJobID) -> AutoTrainerProgress:
        response = self._client._get(f'{self._base_endpoint}/{id}/progress')
        raise_on_error(response)
        return parse_response(AutoTrainerProgress, response)

    def holdout_data_url(self, id: TrainingJobID,
                         presigned: bool = True) -> str:
        response = self._client._get(f'{self._base_endpoint}/{id}/holdout',
                                     params={'presigned': presigned})
        raise_on_error(response)
        return response.text

    def cancel(self, id: str) -> CancelTrainingJobResponse:
        response = self._client._post(f'{self._base_endpoint}/{id}/cancel')
        raise_on_error(response)
        return parse_response(CancelTrainingJobResponse, response)

    def get_config(self, job_id: str) -> TrainingJobRequest:
        """Load the configuration for a training job by ID."""
        resource = self.get(job_id)
        return resource.config


class BatchPredictionJobAPI(CommonJobAPI[BatchPredictionRequest,
                                         BatchPredictionJobResource]):
    r"""Typed API definition for the prediction job resource."""
    def __init__(self, client: KumoClient) -> None:
        super().__init__(client, '/prediction_jobs',
                         BatchPredictionJobResource)

    @override
    def create(self, request: BatchPredictionRequest) -> str:
        # TODO(manan): eventually, all `create` methods should
        # return a validation response:
        raise NotImplementedError

    def maybe_create(
        self, request: BatchPredictionRequest
    ) -> Tuple[Optional[str], ValidationResponse]:
        response = self._client._post(self._base_endpoint,
                                      json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(
            Returns[Tuple[Optional[str], ValidationResponse]], response)

    def list(
        self,
        *,
        model_id: Optional[TrainingJobID] = None,
        pquery_name: Optional[str] = None,
        pquery_id: Optional[str] = None,
        job_status: Optional[JobStatus] = None,
        limit: Optional[int] = None,
        additional_tags: Mapping[str, str] = {},
    ) -> List[BatchPredictionJobResource]:
        if model_id:
            additional_tags = {**additional_tags, 'model_id': model_id}
        return super().list(pquery_name=pquery_name, pquery_id=pquery_id,
                            job_status=job_status, limit=limit,
                            additional_tags=additional_tags)

    def get_progress(self, id: str) -> PredictionProgress:
        response = self._client._get(f'{self._base_endpoint}/{id}/progress')
        raise_on_error(response)
        return parse_response(PredictionProgress, response)

    def cancel(self, id: str) -> CancelBatchPredictionJobResponse:
        response = self._client._post(f'{self._base_endpoint}/{id}/cancel')
        raise_on_error(response)
        return parse_response(CancelBatchPredictionJobResponse, response)

    def get_batch_predictions_url(self, id: str) -> List[str]:
        """Returns presigned URLs pointing to the locations where the
        predictions are stored. Depending on the environment where this is run,
        they could be AWS S3 paths, Snowflake stage paths, or Databricks UC
        volume paths.

        Args:
            id (str): ID of the batch prediction job for which predictions are
                requested
        """
        response = self._client._get(
            f'{self._base_endpoint}/{id}/get_prediction_df_urls')
        raise_on_error(response)
        return parse_response(
            GetPredictionsDfUrlResponse,
            response,
        ).prediction_partitions

    def get_batch_embeddings_url(self, id: str) -> List[str]:
        """Returns presigned URLs pointing to the locations where the
        embeddings are stored. Depending on the environment where this is run,
        they could be AWS S3 paths, Snowflake stage paths, or Databricks UC
        volume paths.

        Args:
            id (str): ID of the batch prediction job for which embeddings are
                requested
        """
        response = self._client._get(
            f'{self._base_endpoint}/{id}/get_embedding_df_urls')
        raise_on_error(response)
        return parse_response(
            GetEmbeddingsDfUrlResponse,
            response,
        ).embedding_partitions

    def get_config(self, job_id: str) -> BatchPredictionRequest:
        """Load the configuration for a batch prediction job by ID."""
        resource = self.get(job_id)
        return resource.config


class GenerateTrainTableJobAPI(CommonJobAPI[GenerateTrainTableRequest,
                                            GenerateTrainTableJobResource]):
    r"""Typed API definition for training table generation job resource."""
    def __init__(self, client: KumoClient) -> None:
        super().__init__(client, '/gentraintable_jobs',
                         GenerateTrainTableJobResource)

    def get_table_data(self, id: GenerateTrainTableJobID,
                       presigned: bool = True) -> List[str]:
        """Return a list of URLs to access train table parquet data.
        There might be multiple URLs if the table data is partitioned into
        multiple files.
        """
        return self._get_table_data(id, presigned)

    def _get_table_data(self, id: GenerateTrainTableJobID,
                        presigned: bool = True,
                        raw_path: bool = False) -> List[str]:
        """Helper function to get train table data."""
        # Raw path to get local file path instead of SPCS stage path
        params: Dict[str, Any] = {'presigned': presigned, 'raw_path': raw_path}
        resp = self._client._get(f'{self._base_endpoint}/{id}/table_data',
                                 params=params)
        raise_on_error(resp)
        return parse_response(List[str], resp)

    def get_split_masks(
            self, id: GenerateTrainTableJobID) -> Dict[TrainingStage, str]:
        """Return a dictionary of presigned URLs keyed by training stage.
        Each URL points to a torch-serialized (default pickle protocol) file of
        the mask tensor for that training stage.

        Example:
            >>> # code to load a mask tensor:
            >>> import io
            >>> import torch
            >>> import requests
            >>> masks = get_split_masks('some-gen-traintable-job-id')
            >>> data_bytes = requests.get(masks[TrainingStage.TEST]).content
            >>> test_mask_tensor = torch.load(io.BytesIO(data))
        """
        resp = self._client._get(f'{self._base_endpoint}/{id}/split_masks')
        raise_on_error(resp)
        return parse_response(Dict[TrainingStage, str], resp)

    def get_progress(self, id: str) -> Dict[str, int]:
        response = self._client._get(f'{self._base_endpoint}/{id}/progress')
        raise_on_error(response)
        return parse_response(Dict[str, int], response)

    def cancel(self, id: str) -> None:
        response = self._client._post(f'{self._base_endpoint}/{id}/cancel')
        raise_on_error(response)

    def validate_custom_train_table(
        self,
        id: str,
        source_table_type: SourceTableType,
        train_table_mod: TrainingTableSpec,
    ) -> ValidationResponse:
        response = self._client._post(
            f'{self._base_endpoint}/{id}/validate_custom_train_table',
            json=to_json_dict({
                'custom_table': source_table_type,
                'train_table_mod': train_table_mod,
            }),
        )
        return parse_response(ValidationResponse, response)

    def get_job_error(self, id: str) -> ErrorDetails:
        """Thin API wrapper for fetching errors from the jobs.

        Arguments:
        id (str): Id of the job whose related errors are expected to be
            queried.
        """
        response = self._client._get(f'{self._base_endpoint}/{id}/get_errors')
        raise_on_error(response)
        return parse_response(ErrorDetails, response)

    def get_config(self, job_id: str) -> GenerateTrainTableRequest:
        """Load the configuration for a training table generation job by ID."""
        resource = self.get(job_id)
        return resource.config


class GeneratePredictionTableJobAPI(
        CommonJobAPI[GeneratePredictionTableRequest,
                     GeneratePredictionTableJobResource]):
    r"""Typed API definition for prediction table generation job resource."""
    def __init__(self, client: KumoClient) -> None:
        super().__init__(client, '/genpredtable_jobs',
                         GeneratePredictionTableJobResource)

    def get_anchor_time(self, id: BatchPredictionJobID) -> Optional[datetime]:
        response = self._client._get(
            f'{self._base_endpoint}/{id}/get_anchor_time')
        raise_on_error(response)
        return parse_response(Returns[Optional[datetime]], response)

    def get_table_data(self, id: GeneratePredictionTableJobID,
                       presigned: bool = True) -> List[str]:
        """Return a list of URLs to access prediction table parquet data.
        There might be multiple URLs if the table data is partitioned into
        multiple files.
        """
        params: Dict[str, Any] = {'presigned': presigned}
        resp = self._client._get(f'{self._base_endpoint}/{id}/table_data',
                                 params=params)
        raise_on_error(resp)
        return parse_response(List[str], resp)

    def cancel(self, id: str) -> None:
        response = self._client._post(f'{self._base_endpoint}/{id}/cancel')
        raise_on_error(response)

    def get_job_error(self, id: str) -> ErrorDetails:
        """Thin API wrapper for fetching errors from the jobs.

        Arguments:
        id (str): Id of the job whose related errors are expected to be
            queried.
        """
        response = self._client._get(f'{self._base_endpoint}/{id}/get_errors')
        raise_on_error(response)
        return parse_response(ErrorDetails, response)

    def get_config(self, job_id: str) -> GeneratePredictionTableRequest:
        """Load the configuration for a
        prediction table generation job by ID.
        """
        resource = self.get(job_id)
        return resource.config


class LLMJobAPI:
    r"""Typed API definition for LLM job resource."""
    def __init__(self, client: KumoClient) -> None:
        self._client = client
        self._base_endpoint = '/llm_embedding_job'

    def create(self, request: LLMRequest) -> LLMJobId:
        response = self._client._post(
            self._base_endpoint,
            json=to_json_dict(request),
        )
        raise_on_error(response)
        return parse_response(LLMResponse, response).job_id

    def get(self, id: LLMJobId) -> JobStatus:
        response = self._client._get(f'{self._base_endpoint}/status/{id}')
        raise_on_error(response)
        return parse_response(JobStatus, response)

    def cancel(self, id: LLMJobId) -> JobStatus:
        response = self._client._delete(f'{self._base_endpoint}/cancel/{id}')
        return response


class ArtifactExportJobAPI:
    r"""Typed API definition for artifact export job resource."""
    def __init__(self, client: KumoClient) -> None:
        self._client = client
        self._base_endpoint = '/artifact'

    def create(self, request: ArtifactExportRequest) -> str:
        response = self._client._post(
            self._base_endpoint,
            json=to_json_dict(request),
        )
        raise_on_error(response)
        return parse_response(ArtifactExportResponse, response).job_id

    # TODO Add an API in artifact export to get
    # JobStatusReport and not just JobStatus
    def get(self, id: str) -> JobStatus:
        response = self._client._get(f'{self._base_endpoint}/{id}')
        raise_on_error(response)
        return parse_response(JobStatus, response)

    def cancel(self, id: str) -> JobStatus:
        response = self._client._post(f'{self._base_endpoint}/{id}/cancel')
        raise_on_error(response)
        return parse_response(JobStatus, response)
