import os
from typing import List, Optional, Set, Tuple, Union

from kumoapi.jobs import (
    BigQueryPredictionOutput,
    DatabricksPredictionOutput,
    MetadataField,
    PredictionArtifactType,
    PredictionOutputConfig,
    PredictionStorageType,
    S3PredictionOutput,
    SnowflakePredictionOutput,
    WriteMode,
)

from kumoai.artifact_export.config import OutputConfig
from kumoai.connector import (
    BigQueryConnector,
    Connector,
    DatabricksConnector,
    S3Connector,
    SnowflakeConnector,
)
from kumoai.databricks import DB_SEP


def validate_output_arguments(
    output_types: Set[str],
    output_connector: Optional[Connector] = None,
    output_table_name: Optional[str] = None,
) -> None:
    r"""Validate the output arguments for a prediction job or an export job."""
    output_types = {x.lower() for x in output_types}
    assert output_types.issubset({'predictions', 'embeddings'})
    if output_connector is not None:
        assert output_table_name is not None
        if not isinstance(output_connector,
                          (S3Connector, SnowflakeConnector,
                           DatabricksConnector, BigQueryConnector)):
            raise ValueError(
                f"Connector type {type(output_connector)} is not supported for"
                f" outputs. Supported output connector types are S3, "
                f"Snowflake, Databricks, and BigQuery.")
        if not isinstance(output_table_name, str):
            raise ValueError(
                f"The output table name must be a string for all "
                f"non-Databricks connectors. Got '{output_table_name}'.")

        if isinstance(output_connector, S3Connector):
            assert output_connector.root_dir is not None
        if isinstance(output_connector, DatabricksConnector):
            assert DB_SEP in output_table_name


def build_prediction_output_config(
    output_type: str,
    output_connector: Optional[Connector] = None,
    output_table_name: Optional[Union[str, Tuple]] = None,
    output_metadata_fields: Optional[List[MetadataField]] = None,
    output_config: Optional[OutputConfig] = None,
) -> PredictionOutputConfig:
    r"""Build the prediction output config."""
    assert output_config is not None
    artifact_type = PredictionArtifactType(output_type.upper())
    output_name = f"{output_table_name}_{output_type}"
    output_metadata_fields = output_metadata_fields or []
    if isinstance(output_connector, S3Connector):
        assert output_connector.root_dir is not None
        return S3PredictionOutput(
            artifact_type=artifact_type,
            file_path=os.path.join(output_connector.root_dir, output_name),
            extra_fields=output_metadata_fields,
        )
    elif isinstance(output_connector, SnowflakeConnector):
        return SnowflakePredictionOutput(
            artifact_type=artifact_type,
            connector_id=output_connector.name,
            table_name=output_name,
            extra_fields=output_metadata_fields,
            write_mode=output_config.connector_specific_config.write_mode
            if output_config.connector_specific_config is not None else
            WriteMode.OVERWRITE,
        )
    elif isinstance(output_connector, DatabricksConnector):
        return DatabricksPredictionOutput(
            artifact_type=artifact_type,
            connector_id=output_connector.name,
            table_name=output_name,
            extra_fields=output_metadata_fields,
        )
    elif isinstance(output_connector, BigQueryConnector):
        return BigQueryPredictionOutput(
            storage_type=PredictionStorageType.BIGQUERY,
            artifact_type=artifact_type,
            connector_id=output_connector.name,
            table_name=output_name,
            extra_fields=output_metadata_fields,
            write_mode=output_config.connector_specific_config.write_mode
            if output_config.connector_specific_config is not None else
            WriteMode.OVERWRITE,
        )
    else:
        raise NotImplementedError()
