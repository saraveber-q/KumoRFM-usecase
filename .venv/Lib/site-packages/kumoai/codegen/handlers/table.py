import logging
from typing import Dict, Sequence, Type

from kumoai.codegen.context import CodegenContext
from kumoai.codegen.edits import (
    UniversalReplacementEdit,
    detect_edits_recursive,
)
from kumoai.codegen.handlers.utils import _get_canonical_import_path
from kumoai.codegen.naming import NameManager
from kumoai.codegen.registry import Handler, get_object_var
from kumoai.graph.table import Table

logger = logging.getLogger(__name__)


def get_handlers() -> Dict[Type, Handler]:
    """Returns a dictionary of handlers for Table types."""
    handlers: Dict[Type, Handler] = {}

    def _table_parents(obj: object,
                       codegen_ctx: CodegenContext) -> list[object]:
        """Table depends on its source_table's connector."""
        assert isinstance(obj, Table)
        return [obj.source_table.connector]

    def _table_imports(obj: object) -> list[str]:
        """Get import statements needed for Table."""
        canonical_module = _get_canonical_import_path(Table)
        return [f"from {canonical_module} import Table"]

    def _build_args(obj: Table, connector_var: str,
                    table_name: str) -> list[str]:
        """Build arguments for Table.from_source_table() call."""
        args = [f"source_table={connector_var}['{table_name}']"]

        if obj.primary_key is not None:
            args.append(f"primary_key='{obj.primary_key.name}'")

        if obj.time_column is not None:
            args.append(f"time_column='{obj.time_column.name}'")

        if obj.end_time_column is not None:
            args.append(f"end_time_column='{obj.end_time_column.name}'")

        source_table_cols = {col.name: col for col in obj.source_table.columns}
        table_cols = {col.name: col for col in obj.columns}

        if len(source_table_cols) != len(table_cols):
            assert len(source_table_cols) > len(table_cols)
            table_col_names = [col_name for col_name in table_cols.keys()]
            args.append(f"column_names={table_col_names}")

        return args

    def _table_emit_lines(
        obj: object,
        var_name: str,
        context: dict,
        codegen_ctx: CodegenContext,
    ) -> list[str]:
        """Generate code lines to recreate a Table using from_source_table."""
        assert isinstance(obj, Table)

        connector_var = get_object_var(codegen_ctx, obj.source_table.connector)
        table_name = obj.source_table.name

        args = _build_args(obj, connector_var, table_name)

        lines = []

        if context.get("input_method") == "id" and context.get("target_id"):
            note = (f"# Note: This could also be done with Table.load("
                    f"'{context['target_id']}') for simpler code")
            lines.append(note)

        if len(args) == 1:
            lines.append(f"{var_name} = Table.from_source_table({args[0]})")
        else:
            args_str = ",\n    ".join(args)
            lines.append(
                f"{var_name} = Table.from_source_table(\n    {args_str},\n)")

        return lines

    def _table_detect_edits(
            target: object, baseline: object,
            name_manager: NameManager) -> Sequence[UniversalReplacementEdit]:
        """Detect edits needed to make baseline match target table."""
        assert isinstance(target, Table)
        assert isinstance(baseline, Table)

        try:
            result = detect_edits_recursive(target, baseline, "", name_manager)
            logger.debug(f"Found for table {len(result.edits)} edits with "
                         f"{len(result.imports)} imports")
            return result.edits
        except Exception as e:
            logger.error(f"Error during table edit detection: {e}")
            return []

    handlers[Table] = Handler(
        parents=_table_parents,
        required_imports=_table_imports,
        emit_lines=_table_emit_lines,
        detect_edits=_table_detect_edits,
    )

    return handlers
