from __future__ import annotations

from typing import Dict, Type

from kumoai.codegen.context import CodegenContext
from kumoai.codegen.handlers.utils import _get_canonical_import_path
from kumoai.codegen.registry import Handler
from kumoai.connector import (
    BigQueryConnector,
    DatabricksConnector,
    FileUploadConnector,
    S3Connector,
    SnowflakeConnector,
)


def _get_by_name_handler_factory(cls: type) -> Handler:
    """Factory for creating a Handler for any connector that uses the
    get_by_name pattern.
    """
    def get_imports(obj: object, bound_cls: type = cls) -> list[str]:
        canonical_module = _get_canonical_import_path(bound_cls)
        return [f"from {canonical_module} import {bound_cls.__name__}"]

    def get_lines(
        obj: object,
        var_name: str,
        context: dict,
        codegen_ctx: CodegenContext,
    ) -> list[str]:
        assert isinstance(obj, cls)
        assert hasattr(obj, "name")
        obj_name = getattr(obj, "name")
        return [f"{var_name} = {cls.__name__}.get_by_name('{obj_name}')"]

    return Handler(
        parents=lambda e, ctx: [],
        required_imports=get_imports,
        emit_lines=get_lines,
        detect_edits=None,
    )


def get_handlers() -> Dict[Type, Handler]:
    """Returns a dictionary of handlers for all connector types."""
    handlers: Dict[Type, Handler] = {}

    # S3Connector gets special handling to support both patterns

    def _s3_connector_parents(obj: object,
                              codegen_ctx: CodegenContext) -> list[object]:
        return []

    def _s3_connector_imports(obj: object) -> list[str]:
        canonical_module = _get_canonical_import_path(S3Connector)
        return [f"from {canonical_module} import S3Connector"]

    def _s3_connector_emit_lines(
        obj: object,
        var_name: str,
        context: dict,
        codegen_ctx: CodegenContext,
    ) -> list[str]:
        assert isinstance(obj, S3Connector)
        assert hasattr(obj, "name")

        # Check if connector has root_dir attribute and it's not None
        if hasattr(obj, "root_dir") and obj.root_dir is not None:
            root_dir = getattr(obj, "root_dir")
            return [f"{var_name} = S3Connector(root_dir='{root_dir}')"]
        else:
            obj_name = getattr(obj, "name")
            return [f"{var_name} = S3Connector.get_by_name('{obj_name}')"]

    handlers[S3Connector] = Handler(
        parents=_s3_connector_parents,
        required_imports=_s3_connector_imports,
        emit_lines=_s3_connector_emit_lines,
        detect_edits=None,
    )

    # Other persistent connectors use the standard get_by_name pattern
    other_persistent_connectors = [
        BigQueryConnector,
        SnowflakeConnector,
        DatabricksConnector,
    ]
    for connector_cls in other_persistent_connectors:
        handlers[connector_cls] = _get_by_name_handler_factory(connector_cls)

    def _file_upload_parents(obj: object,
                             codegen_ctx: CodegenContext) -> list[object]:
        return []

    def _file_upload_imports(obj: object) -> list[str]:
        canonical_module = _get_canonical_import_path(type(obj))
        return [f"from {canonical_module} import {type(obj).__name__}"]

    def _file_upload_emit_lines(
        obj: object,
        var_name: str,
        context: dict,
        codegen_ctx: CodegenContext,
    ) -> list[str]:
        assert isinstance(obj, FileUploadConnector)
        return [
            f"{var_name} = {type(obj).__name__}"
            f"(file_type='{obj.file_type}')"
        ]

    handlers[FileUploadConnector] = Handler(
        parents=_file_upload_parents,
        required_imports=_file_upload_imports,
        emit_lines=_file_upload_emit_lines,
        detect_edits=None,
    )

    return handlers
