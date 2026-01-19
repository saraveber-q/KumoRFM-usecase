import asyncio
import concurrent
import concurrent.futures
import time

from kumoapi.common import JobStatus
from typing_extensions import override

from kumoai import global_state
from kumoai.futures import KumoProgressFuture, create_future


class ArtifactExportResult:
    r"""Represents a completed artifact export job."""
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def tracking_url(self) -> str:
        r"""Returns a tracking URL pointing to the UI display of
        this prediction export job.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(job_id={self.job_id})"


class ArtifactExportJob(KumoProgressFuture[ArtifactExportResult]):
    """Represents an in-progress artifact export job."""
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._fut: concurrent.futures.Future = create_future(
            _poll_export(job_id))

    @property
    def id(self) -> str:
        """The unique ID of this export job."""
        return self.job_id

    @override
    def result(self) -> ArtifactExportResult:
        return self._fut.result()

    @override
    def future(self) -> 'concurrent.futures.Future[ArtifactExportResult]':
        return self._fut

    @override
    def _attach_internal(
        self,
        interval_s: float = 20.0,
    ) -> ArtifactExportResult:
        """Allows a user to attach to a running export job and view
        its progress.

        Args:
            interval_s (float): Time interval (seconds) between polls, minimum
                value allowed is 4 seconds.
        """
        assert interval_s >= 4.0
        print(f"Attaching to export job {self.job_id}. To detach from "
              f"this job, please enter Ctrl+C (the job will continue to run, "
              f"and you can re-attach anytime).")

        # TODO improve print statements.
        # Will require changes to status to return
        # JobStatusReport instead of JobStatus.
        while not self.done():
            status = self.status()
            print(f"Export job {self.job_id} status: {status}")
            time.sleep(interval_s)

        return self.result()

    def status(self) -> JobStatus:
        """Returns the status of a running export job."""
        return get_export_status(self.job_id)

    def cancel(self) -> bool:
        """Cancels a running export job.

        Returns:
            bool: True if the job is in a terminal state.
        """
        api = global_state.client.artifact_export_api
        status = api.cancel(self.job_id)
        if status == JobStatus.CANCELLED:
            return True
        return False


def get_export_status(job_id: str) -> JobStatus:
    api = global_state.client.artifact_export_api
    resource = api.get(job_id)
    return resource


async def _poll_export(job_id: str) -> ArtifactExportResult:
    status = get_export_status(job_id)
    while not status.is_terminal:
        await asyncio.sleep(10)
        status = get_export_status(job_id)

    if status != JobStatus.DONE:
        raise RuntimeError(f"Export job {job_id} failed "
                           f"with job status {status}.")

    return ArtifactExportResult(job_id=job_id)
