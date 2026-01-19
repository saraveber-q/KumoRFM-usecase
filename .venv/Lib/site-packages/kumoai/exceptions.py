import http
from typing import Dict, Optional


class HTTPException(Exception):
    r"""An HTTP exception, with detailed information and headers."""
    def __init__(
        self,
        status_code: int,
        detail: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        # Derived from starlette/blob/master/starlette/exceptions.py
        if detail is None:
            detail = http.HTTPStatus(status_code).phrase
        self.status_code = status_code
        self.detail = detail
        self.headers = headers

    def __str__(self) -> str:
        return f"{self.status_code}: {self.detail}"

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (f"{class_name}(status_code={self.status_code!r}, "
                f"detail={self.detail!r})")
