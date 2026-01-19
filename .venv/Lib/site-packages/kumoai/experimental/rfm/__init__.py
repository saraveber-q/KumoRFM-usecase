import ipaddress
import logging
import os
import re
import socket
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import kumoai
from kumoai.client.client import KumoClient

from .authenticate import authenticate
from .sagemaker import (
    KumoClient_SageMakerAdapter,
    KumoClient_SageMakerProxy_Local,
)
from .base import Table
from .backend.local import LocalTable
from .graph import Graph
from .rfm import ExplainConfig, Explanation, KumoRFM

logger = logging.getLogger('kumoai_rfm')


def _is_local_address(host: str | None) -> bool:
    """Return True if the hostname/IP refers to the local machine."""
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
        for _, _, _, _, sockaddr in infos:
            ip = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_loopback or ip_obj.is_unspecified:
                return True
        return False
    except Exception:
        return False


class InferenceBackend(str, Enum):
    REST = "REST"
    LOCAL_SAGEMAKER = "LOCAL_SAGEMAKER"
    AWS_SAGEMAKER = "AWS_SAGEMAKER"
    UNKNOWN = "UNKNOWN"


def _detect_backend(
        url: str) -> Tuple[InferenceBackend, Optional[str], Optional[str]]:
    parsed = urlparse(url)

    # Remote SageMaker
    if ("runtime.sagemaker" in parsed.netloc
            and parsed.path.endswith("/invocations")):
        # Example: https://runtime.sagemaker.us-west-2.amazonaws.com/
        # endpoints/Name/invocations
        match = re.search(r"runtime\.sagemaker\.([a-z0-9-]+)\.amazonaws\.com",
                          parsed.netloc)
        region = match.group(1) if match else None
        m = re.search(r"/endpoints/([^/]+)/invocations", parsed.path)
        endpoint_name = m.group(1) if m else None
        return InferenceBackend.AWS_SAGEMAKER, region, endpoint_name

    # Local SageMaker
    if parsed.port == 8080 and parsed.path.endswith(
            "/invocations") and _is_local_address(parsed.hostname):
        return InferenceBackend.LOCAL_SAGEMAKER, None, None

    # Default: regular REST
    return InferenceBackend.REST, None, None


@dataclass
class RfmGlobalState:
    _url: str = '__url_not_provided__'
    _backend: InferenceBackend = InferenceBackend.UNKNOWN
    _region: Optional[str] = None
    _endpoint_name: Optional[str] = None
    _thread_local = threading.local()

    # Thread-safe init-once.
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    @property
    def client(self) -> KumoClient:
        if self._backend == InferenceBackend.REST:
            return kumoai.global_state.client

        if hasattr(self._thread_local, '_sagemaker'):
            # Set the spcs token in the client to ensure it has the latest.
            return self._thread_local._sagemaker

        sagemaker_client: KumoClient
        if self._backend == InferenceBackend.LOCAL_SAGEMAKER:
            sagemaker_client = KumoClient_SageMakerProxy_Local(self._url)
        else:
            assert self._backend == InferenceBackend.AWS_SAGEMAKER
            assert self._region
            assert self._endpoint_name
            sagemaker_client = KumoClient_SageMakerAdapter(
                self._region, self._endpoint_name)

        self._thread_local._sagemaker = sagemaker_client
        return sagemaker_client

    def reset(self) -> None:  # For testing only.
        with self._lock:
            self._initialized = False
            self._url = '__url_not_provided__'
            self._backend = InferenceBackend.UNKNOWN
            self._region = None
            self._endpoint_name = None
            self._thread_local = threading.local()


global_state = RfmGlobalState()


def init(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    snowflake_credentials: Optional[Dict[str, str]] = None,
    snowflake_application: Optional[str] = None,
    log_level: str = "INFO",
) -> None:
    with global_state._lock:
        if global_state._initialized:
            if url != global_state._url:
                raise ValueError(
                    "Kumo RFM has already been initialized with a different "
                    "URL. Re-initialization with a different URL is not "
                    "supported.")
            return

        if url is None:
            url = os.getenv("RFM_API_URL", "https://kumorfm.ai/api")

        backend, region, endpoint_name = _detect_backend(url)
        if backend == InferenceBackend.REST:
            # Initialize kumoai.global_state
            if (kumoai.global_state.initialized
                    and kumoai.global_state._url != url):
                raise ValueError(
                    "Kumo AI SDK has already been initialized with different "
                    "API URL. Please restart Python interpreter and "
                    "initialize via kumoai.rfm.init()")
            kumoai.init(url=url, api_key=api_key,
                        snowflake_credentials=snowflake_credentials,
                        snowflake_application=snowflake_application,
                        log_level=log_level)
        elif backend == InferenceBackend.AWS_SAGEMAKER:
            assert region
            assert endpoint_name
            KumoClient_SageMakerAdapter(region, endpoint_name).authenticate()
        else:
            assert backend == InferenceBackend.LOCAL_SAGEMAKER
            KumoClient_SageMakerProxy_Local(url).authenticate()

        global_state._url = url
        global_state._backend = backend
        global_state._region = region
        global_state._endpoint_name = endpoint_name
        global_state._initialized = True
        logger.info("Kumo RFM initialized with backend: %s, url: %s", backend,
                    url)


LocalGraph = Graph  # NOTE Backward compatibility - do not use anymore.

__all__ = [
    'authenticate',
    'init',
    'Table',
    'LocalTable',
    'Graph',
    'KumoRFM',
    'ExplainConfig',
    'Explanation',
]
