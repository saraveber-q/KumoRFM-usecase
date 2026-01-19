from dataclasses import dataclass
from typing import Any

from kumoapi.typing import Dtype, Stype


@dataclass(init=False, repr=False, eq=False)
class Column:
    stype: Stype

    def __init__(
        self,
        name: str,
        dtype: Dtype,
        stype: Stype,
        is_primary_key: bool = False,
        is_time_column: bool = False,
        is_end_time_column: bool = False,
    ) -> None:
        self._name = name
        self._dtype = Dtype(dtype)
        self._is_primary_key = is_primary_key
        self._is_time_column = is_time_column
        self._is_end_time_column = is_end_time_column
        self.stype = Stype(stype)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> Dtype:
        return self._dtype

    def __setattr__(self, key: str, val: Any) -> None:
        if key == 'stype':
            if isinstance(val, str):
                val = Stype(val)
            assert isinstance(val, Stype)
            if not val.supports_dtype(self.dtype):
                raise ValueError(f"Column '{self.name}' received an "
                                 f"incompatible semantic type (got "
                                 f"dtype='{self.dtype}' and stype='{val}')")
            if self._is_primary_key and val != Stype.ID:
                raise ValueError(f"Primary key '{self.name}' must have 'ID' "
                                 f"semantic type (got '{val}')")
            if self._is_time_column and val != Stype.timestamp:
                raise ValueError(f"Time column '{self.name}' must have "
                                 f"'timestamp' semantic type (got '{val}')")
            if self._is_end_time_column and val != Stype.timestamp:
                raise ValueError(f"End time column '{self.name}' must have "
                                 f"'timestamp' semantic type (got '{val}')")

        super().__setattr__(key, val)

    def __hash__(self) -> int:
        return hash((self.name, self.stype, self.dtype))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Column):
            return False
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(name={self.name}, '
                f'stype={self.stype}, dtype={self.dtype})')
