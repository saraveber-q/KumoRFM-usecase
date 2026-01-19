from kumoai.connector import FileUploadConnector
from kumoai.connector.utils import replace_table
from kumoai.graph import Edge, Graph, Table


def from_relbench(dataset_name: str) -> Graph:
    r"""Creates a Kumo graph from a RelBench dataset. This function processes
    the specified RelBench dataset, uploads its tables to the Kumo data plane,
    and registers them as part of a Kumo graph, including inferred table
    metadata and edges.

    .. note::

        Please note that this method is subject to the limitations for file
        upload in :class:`~kumoai.connector.FileUploadConnector`.

    .. code-block:: python

        import kumoai
        from kumoai.utils.datasets import from_relbench

        # Assume dataset `example_dataset` in the RelBench repository:
        graph = from_relbench(dataset_name="example_dataset")

    Args:
        dataset_name: The name of the RelBench dataset to be processed.

    Returns:
        A :class:`~kumoai.Graph` object containing the tables and edges
        derived from the RelBench dataset.

    Raises:
        ValueError: If the dataset cannot be retrieved or processed.
    """
    try:
        import relbench
    except ImportError:
        raise RuntimeError(
            "Creating a Kumo Graph from a RelBench dataset requires the "
            "'relbench' package to be installed. Please install the package "
            "before proceeding.")

    connector = FileUploadConnector(file_type="parquet")
    dataset = relbench.datasets.get_dataset(dataset_name, download=True)
    db = dataset.get_db(upto_test_timestamp=False)

    # Store table metadata and edges:
    table_metadata = {}

    # Process each table in the database
    for table_key in db.table_dict.keys():
        # Save the table locally as a parquet file:
        table = db.table_dict[table_key]
        parquet_path = f"tmp_{table_key}.parquet"
        table.df.to_parquet(parquet_path, index=False)

        # Replace the table on the Kumo data plane:
        replace_table(name=table_key, path=parquet_path, file_type="parquet")

        # Register the table with inferred metadata and collect edge
        # information:
        table_metadata[table_key] = dict(
            table=Table.from_source_table(
                source_table=connector[table_key],
                primary_key=table.pkey_col,
                time_column=table.time_col,
            ).infer_metadata(), edges=table.fkey_col_to_pkey_table)

    tables = {
        table_key: table_metadata[table_key]['table']
        for table_key in table_metadata.keys()
    }
    edges = []
    for table_key, table_data in table_metadata.items():
        for edge_key, dst_table in table_data['edges'].items():
            edges.append(
                Edge(src_table=table_key, fkey=edge_key, dst_table=dst_table))

    # Create and return the graph
    return Graph(
        tables=tables,
        edges=edges,
    )
