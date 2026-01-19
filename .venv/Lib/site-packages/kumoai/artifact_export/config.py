import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from kumoapi.jobs import MetadataField, WriteMode
from kumoapi.typing import WITH_PYDANTIC_V2
from pydantic.dataclasses import dataclass as pydantic_dataclass

if WITH_PYDANTIC_V2:
    from pydantic import field_validator, model_validator  # type: ignore
else:
    from pydantic import root_validator, validator

from kumoai.connector.base import Connector


def compatible_field_validator(field_name: str):  # type: ignore
    """Decorator factory that creates a field validator compatible with both
    Pydantic v1 and v2.

    Usage:
        @compatible_field_validator('field_name')
        def validate_field(cls, v, values_or_info):
            # Your validation logic here
            return v
    """
    def decorator(func):  # type: ignore
        if WITH_PYDANTIC_V2:

            @field_validator(field_name)
            @classmethod
            @functools.wraps(func)
            def wrapper(cls, v, info):  # type: ignore
                # Convert info to values dict for compatibility
                values = info.data if hasattr(info, 'data') else {}
                return func(cls, v, values)

            return wrapper
        else:

            @validator(field_name)
            @functools.wraps(func)
            def wrapper(cls, v, values):  # type: ignore
                return func(cls, v, values)

            return wrapper

    return decorator


# TODO: probably will need to be removed b/c using __post_init__ instead
def compatible_model_validator(mode='before'):  # type: ignore
    """Decorator factory that creates a model validator compatible with both
    Pydantic v1 and v2.

    Usage:
        @compatible_model_validator()
        def validate_model(cls, values):
            # Your validation logic here
            return values
    """
    def decorator(func):  # type: ignore
        if WITH_PYDANTIC_V2:

            @model_validator(mode=mode)
            @classmethod
            @functools.wraps(func)
            def wrapper(cls, values):  # type: ignore
                return func(cls, values)

            return wrapper
        else:

            @root_validator
            @functools.wraps(func)
            def wrapper(cls, values):  # type: ignore
                return func(cls, values)

            return wrapper

    return decorator


@dataclass(frozen=True)
class QueryConnectorConfig:
    # If using OVERWRITE, big query connector will first write to a staging
    # table followed by overwriting to the destination table.
    # When using APPEND, it is strongly recommended to use
    # MetadataField.JOB_TIMESTAMP to indicate the timestamp of the job.
    write_mode: WriteMode = WriteMode.APPEND


@dataclass(frozen=True)
class BigQueryOutputConfig(QueryConnectorConfig):
    pass


@dataclass(frozen=True)
class SnowflakeConnectorConfig(QueryConnectorConfig):
    pass


CONNECTOR_CONFIG_MAPPING = {
    'BigQueryConnector': BigQueryOutputConfig,
    'SnowflakeConnector': SnowflakeConnectorConfig,
    # 'DatabricksConnector': DatabricksOutputConfig,
    # 'S3Connector': S3OutputConfig,
}


@pydantic_dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class OutputConfig:
    """Output configuration associated with a Batch Prediction Job.
    Specifies the output types and optionally output data source
    configuration.

    Args:
        output_types(`Set[str]`): The types of outputs that should be produced
            by the prediction job. Can include either ``'predictions'``,
            ``'embeddings'``, or both.
        output_connector(`Connector` or None): The output data source that Kumo
                should write batch predictions to, if it is None,
                produce local download output only.
        output_table_name(`str` or `Tuple[str, str]` or None): The name of the
            table in the output data source
            that Kumo should write batch predictions to. In the case of
            a Databricks connector, this should be a tuple of two strings:
            the schema name and the output prediction table name.
        output_metadata_fields(`List[MetadataField]` or None): Any additional
            metadata fields to include as new columns in the produced
            ``'predictions'`` output. Currently, allowed options are
            ``JOB_TIMESTAMP`` and ``ANCHOR_TIMESTAMP``.
        connector_specific_config(`QueryConnectorConfig` or None): The custom
            connector specific output config for predictions, for
            example whether to append or overwrite existing table.
    """
    output_types: Set[str]
    output_connector: Optional[Connector] = None
    output_table_name: Optional[Union[str, Tuple]] = None
    output_metadata_fields: Optional[List[MetadataField]] = None
    connector_specific_config: Optional[Union[
        BigQueryOutputConfig,
        SnowflakeConnectorConfig,
    ]] = None

    @compatible_field_validator('connector_specific_config')
    def validate_connector_config(cls, v: Any, values: Dict) -> Any:
        """Validate the connector specific output config. Raises ValueError if
        there is a mismatch between the connector type and the config type.
        """
        # Skip validation if no connector or no specific config
        if values.get('output_connector') is None or v is None:
            return v

        connector_type = type(values['output_connector']).__name__
        expected_config_type = CONNECTOR_CONFIG_MAPPING.get(connector_type)

        # If we don't have a mapping for this connector type, it doesn't
        # support specific configs yet
        if expected_config_type is None:
            raise ValueError(
                f"Connector type '{connector_type}' does not support "
                f"specific output configurations")

        # Check if the provided config is of the correct type
        if not isinstance(v, expected_config_type):
            raise ValueError(
                f"Connector type '{connector_type}' requires output "
                f"config of type '{expected_config_type.__name__}', but "
                f"got '{type(v).__name__}'")

        return v


@pydantic_dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class TrainingTableExportConfig(OutputConfig):
    """Export configuration associated with a Training Table.

    Args:
        output_types(`Set[str]`): The artifact to export from the training
            table job. Currently only `'training_table'` is supported.
            Which exports the full training table to the output connector.
        output_connector(`Connector`): The output data source that Kumo should
            write training table artifacts to.
        output_table_name(str): The name of the table in the output data source
            that Kumo should write batch predictions to. In the case of
            a Databricks connector, this should be a tuple of two strings:
            the schema name and the output prediction table name.
        connector_specific_config(QueryConnectorConfig or None):
            Defines custom connector specific output
            for example whether to append or overwrite
            existing table. This is currently only supported for BigQuery and
            Snowflake.
    """
    output_connector: Connector
    output_table_name: str

    def __post_init__(self) -> None:
        if self.output_types != {'training_table'}:
            raise ValueError("output_type must be set(['training_table'])"
                             f" (got {self.output_types})")
        if self.output_connector is None:
            raise ValueError("output_connector is required")
        if self.output_table_name is None:
            raise ValueError("output_table_name is required")
        if self.output_metadata_fields is not None:
            raise ValueError(
                "output_metadata_fields is not supported for training"
                "table export")
