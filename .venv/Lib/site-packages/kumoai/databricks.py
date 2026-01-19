from typing import Optional, Tuple, Union

DB_SEP = '__kumo__'


def to_db_table_name(
        table_name: Optional[Union[str, Tuple]] = None) -> Optional[str]:
    r"""For Databricks connectors, return the table name whichs is
    a Tuple as a string with the format `f"{schema}__kumo__{table}"`.
    """
    if table_name and isinstance(table_name, tuple):
        return (f"{table_name[0]}{DB_SEP}"
                f"{table_name[1]}")
    return table_name  # type: ignore
