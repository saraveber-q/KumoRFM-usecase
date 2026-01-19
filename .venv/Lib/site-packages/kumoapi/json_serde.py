import json
from typing import Any, Dict, Type, TypeVar

from pydantic import SecretStr

from kumoapi.typing import WITH_PYDANTIC_V2

if WITH_PYDANTIC_V2:
    from pydantic_core import to_jsonable_python as lib_encoder
else:
    from pydantic.json import pydantic_encoder as lib_encoder

T = TypeVar('T')


def trusted_encoder(obj: Any) -> Any:
    if isinstance(obj, SecretStr):
        return obj.get_secret_value()

    if WITH_PYDANTIC_V2:
        return lib_encoder(obj, context={'insecure': True})

    return lib_encoder(obj)


def to_json(pydantic_obj: Any, insecure: bool = False) -> str:
    r"""Encodes a pydantic object into JSON.

    The `insecure` flag should only be used by trusted internal code where the
    output of the JSON is not accessible to any users and `SecretStr`s are
    hidden in some other fashion."""
    encoder = trusted_encoder if insecure else lib_encoder

    return json.dumps(
        pydantic_obj,
        default=encoder,
        allow_nan=True,
        indent=2,
    )


def to_json_dict(pydantic_obj: Any, insecure: bool = False) -> Dict[str, Any]:
    return json.loads(to_json(pydantic_obj, insecure=insecure))


def from_json(obj: Any, cls: Type[T]) -> T:
    if isinstance(obj, str):
        obj = json.loads(obj)
    if WITH_PYDANTIC_V2:
        from pydantic import TypeAdapter
        adapter = TypeAdapter(cls)
        return adapter.validate_python(obj)
    else:
        from pydantic import parse_obj_as
        return parse_obj_as(cls, obj)
