from codeguard.analyzers.python_ast import PythonASTAnalyzer
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.symbols import FileSymbolIndex


class FileSymbolIndexer:
    """Build a symbol index for one discovered Python file."""

    def __init__(
        self,
        analyzer: PythonASTAnalyzer | None = None,
    ) -> None:
        self._analyzer = analyzer if analyzer is not None else PythonASTAnalyzer()

    def index(
        self,
        discovered_file: DiscoveredPythonFile,
        source: str,
    ) -> FileSymbolIndex:
        """Analyze source code and associate the result with its file."""

        analysis = self._analyzer.analyze(
            source,
            file_path=discovered_file.path,
        )

        return FileSymbolIndex(
            discovered_file=discovered_file,
            analysis=analysis,
        )
