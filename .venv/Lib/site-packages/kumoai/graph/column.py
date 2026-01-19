from dataclasses import dataclass
from typing import Any, Optional, Union

from kumoapi.table import TimestampUnit
from kumoapi.typing import Dtype, Stype

from kumoai.mixin import CastMixin


@dataclass(init=False)
class Column(CastMixin):
    r"""A column represents metadata information for a column in a Kumo
    :class:`~kumoai.graph.Table`. Columns can be created independent of
    a table, or can be fetched from a table with the
    :meth:`~kumoai.graph.Table.column` method.

    .. code-block:: python

        import kumoai

        # Fetch a column from a `kumoai.Table`:
        table = kumoai.Table(...)

        column = table.column("col_name")
        column = table["col_name"]  # equivalent to the above.

        # Edit a column's data type:
        print("Existing dtype: ", column.dtype)
        column.dtype = "int"

        # Edit a column's semantic type:
        print("Existing stype: ", column.stype)
        column.stype = "ID"

    Args:
        name: The name of this column.
        stype: The semantic type of this column. Semantic types can be
            specified as strings: the list of possible semantic types
            is located at :class:`~kumoai.Stype`.
        dtype: The data type of this column. Data types can be specified
            as strings: the list of possible data types is located at
            :class:`~kumoai.Dtype`.
        timestamp_format: If this column represents a timestamp, the format
            that the timestamp should be parsed in. The format can either be
            a :class:`~kumoapi.table.TimestampUnit` for integer columns or a
            string with a format identifier described
            `here <https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html>`__
            for a SaaS Kumo deployment and
            `here <https://docs.snowflake.com/en/sql-reference/date-time-input-output#about-the-elements-used-in-input-and-output-formats>`__
            for a Snowpark Container Services Kumo deployment. If left empty,
            will be intelligently inferred by Kumo.
    """  # noqa: E501
    name: str
    stype: Optional[Stype] = None
    dtype: Optional[Dtype] = None
    timestamp_format: Optional[Union[str, TimestampUnit]] = None

    def __init__(
        self,
        name: str,
        stype: Optional[Union[Stype, str]] = None,
        dtype: Optional[Union[Dtype, str]] = None,
        timestamp_format: Optional[Union[str, TimestampUnit]] = None,
    ) -> None:
        self.name = name
        self.stype = Stype(stype) if stype is not None else None
        self.dtype = Dtype(dtype) if dtype is not None else None
        try:
            self.timestamp_format = TimestampUnit(timestamp_format)
        except ValueError:
            self.timestamp_format = timestamp_format

    def __hash__(self) -> int:
        return hash((self.name, self.stype, self.dtype, self.timestamp_format))

    def __setattr__(self, key: Any, value: Any) -> None:
        if key == 'name' and value != getattr(self, key, value):
            raise AttributeError("Attribute 'name' is read-only")
        elif key == 'stype' and isinstance(value, str):
            value = Stype(value)
        elif key == 'dtype' and isinstance(value, str):
            value = Dtype(value)
        elif key == 'timestamp_format' and isinstance(value, str):
            try:
                value = TimestampUnit(value)
            except ValueError:
                pass
        super().__setattr__(key, value)

    def update(self, obj: 'Column', override: bool = True) -> 'Column':
        for key in self.__dict__:
            if key[0] == '_':  # Skip private attributes:
                continue
            value = getattr(obj, key, None)
            if value is not None:
                if override or getattr(self, key, None) is None:
                    setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        out = (f"Column(name=\"{self.name}\", stype=\"{self.stype}\", "
               f"dtype=\"{self.dtype}\"")
        if self.timestamp_format is not None:
            out += f", timestamp_format=\"{self.timestamp_format}\""
        out += ")"
        return out
