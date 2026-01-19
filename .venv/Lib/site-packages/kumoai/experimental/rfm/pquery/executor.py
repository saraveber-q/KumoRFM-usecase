from abc import ABC, abstractmethod
from typing import Dict, Generic, Tuple, TypeVar

from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    Column,
    Condition,
    Filter,
    Join,
    LogicalOperation,
)

TableData = TypeVar('TableData')
ColumnData = TypeVar('ColumnData')
IndexData = TypeVar('IndexData')


class PQueryExecutor(Generic[TableData, ColumnData, IndexData], ABC):
    @abstractmethod
    def execute_column(
        self,
        column: Column,
        feat_dict: Dict[str, TableData],
        filter_na: bool = True,
    ) -> Tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute_aggregation(
        self,
        aggr: Aggregation,
        feat_dict: Dict[str, TableData],
        time_dict: Dict[str, ColumnData],
        batch_dict: Dict[str, IndexData],
        anchor_time: ColumnData,
        filter_na: bool = True,
        num_forecasts: int = 1,
    ) -> Tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute_condition(
        self,
        condition: Condition,
        feat_dict: Dict[str, TableData],
        time_dict: Dict[str, ColumnData],
        batch_dict: Dict[str, IndexData],
        anchor_time: ColumnData,
        filter_na: bool = True,
        num_forecasts: int = 1,
    ) -> Tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute_logical_operation(
        self,
        logical_operation: LogicalOperation,
        feat_dict: Dict[str, TableData],
        time_dict: Dict[str, ColumnData],
        batch_dict: Dict[str, IndexData],
        anchor_time: ColumnData,
        filter_na: bool = True,
        num_forecasts: int = 1,
    ) -> Tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute_join(
        self,
        join: Join,
        feat_dict: Dict[str, TableData],
        time_dict: Dict[str, ColumnData],
        batch_dict: Dict[str, IndexData],
        anchor_time: ColumnData,
        filter_na: bool = True,
        num_forecasts: int = 1,
    ) -> Tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute_filter(
        self,
        filter: Filter,
        feat_dict: Dict[str, TableData],
        time_dict: Dict[str, ColumnData],
        batch_dict: Dict[str, IndexData],
        anchor_time: ColumnData,
    ) -> Tuple[ColumnData, IndexData]:
        pass

    @abstractmethod
    def execute(
        self,
        query: ValidatedPredictiveQuery,
        feat_dict: Dict[str, TableData],
        time_dict: Dict[str, ColumnData],
        batch_dict: Dict[str, IndexData],
        anchor_time: ColumnData,
        num_forecasts: int = 1,
    ) -> Tuple[ColumnData, IndexData]:
        pass
