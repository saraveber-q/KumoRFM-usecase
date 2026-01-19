import sys
import time
from typing import Any, List, Optional, Union

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.padding import Padding
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    Task,
    TextColumn,
    TimeRemainingColumn,
)
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from typing_extensions import Self


class ProgressLogger:
    def __init__(self, msg: str) -> None:
        self.msg = msg
        self.logs: List[str] = []

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @property
    def duration(self) -> float:
        assert self.start_time is not None
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.perf_counter() - self.start_time

    def log(self, msg: str) -> None:
        self.logs.append(msg)

    def __enter__(self) -> Self:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.msg})'


class ColoredMofNCompleteColumn(MofNCompleteColumn):
    def __init__(self, style: str = 'green') -> None:
        super().__init__()
        self.style = style

    def render(self, task: Task) -> Text:
        return Text(str(super().render(task)), style=self.style)


class ColoredTimeRemainingColumn(TimeRemainingColumn):
    def __init__(self, style: str = 'cyan') -> None:
        super().__init__()
        self.style = style

    def render(self, task: Task) -> Text:
        return Text(str(super().render(task)), style=self.style)


class InteractiveProgressLogger(ProgressLogger):
    def __init__(
        self,
        msg: str,
        verbose: bool = True,
        refresh_per_second: int = 10,
    ) -> None:
        super().__init__(msg=msg)

        self.verbose = verbose
        self.refresh_per_second = refresh_per_second

        self._progress: Optional[Progress] = None
        self._task: Optional[int] = None

        self._live: Optional[Live] = None
        self._exception: bool = False

    def init_progress(self, total: int, description: str) -> None:
        assert self._progress is None
        if self.verbose:
            self._progress = Progress(
                TextColumn(f'   ↳ {description}', style='dim'),
                BarColumn(bar_width=None),
                ColoredMofNCompleteColumn(style='dim'),
                TextColumn('•', style='dim'),
                ColoredTimeRemainingColumn(style='dim'),
            )
            self._task = self._progress.add_task("Progress", total=total)

    def step(self) -> None:
        if self.verbose:
            assert self._progress is not None
            assert self._task is not None
            self._progress.update(self._task, advance=1)  # type: ignore

    def __enter__(self) -> Self:
        from kumoai import in_notebook

        super().__enter__()

        if not in_notebook():  # Render progress bar in TUI.
            sys.stdout.write("\x1b]9;4;3\x07")
            sys.stdout.flush()

        if self.verbose:
            self._live = Live(
                self,
                refresh_per_second=self.refresh_per_second,
                vertical_overflow='visible',
            )
            self._live.start()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        from kumoai import in_notebook

        super().__exit__(exc_type, exc_val, exc_tb)

        if exc_type is not None:
            self._exception = True

        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task = None

        if self._live is not None:
            self._live.update(self, refresh=True)
            self._live.stop()
            self._live = None

        if not in_notebook():
            sys.stdout.write("\x1b]9;4;0\x07")
            sys.stdout.flush()

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:

        table = Table.grid(padding=(0, 1))

        icon: Union[Text, Padding]
        if self._exception:
            style = 'red'
            icon = Text('❌', style=style)
        elif self.end_time is not None:
            style = 'green'
            icon = Text('✅', style=style)
        else:
            style = 'cyan'
            icon = Padding(Spinner('dots', style=style), (0, 1, 0, 0))

        title = Text.from_markup(
            f'{self.msg} ({self.duration:.2f}s)',
            style=style,
        )
        table.add_row(icon, title)

        for log in self.logs:
            table.add_row('', Text(f'↳ {log}', style='dim'))

        yield table

        if self.verbose and self._progress is not None:
            yield self._progress.get_renderable()
