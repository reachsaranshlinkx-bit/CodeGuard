from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FunctionInfo:
    """Information about a Python function or method."""

    name: str
    qualified_name: str
    lineno: int
    end_lineno: int
    is_async: bool
    is_method: bool


@dataclass(frozen=True, slots=True)
class ClassInfo:
    """Information about a Python class."""

    name: str
    qualified_name: str
    lineno: int
    end_lineno: int
    methods: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImportInfo:
    """Information about a Python import."""

    module: str | None
    name: str
    alias: str | None
    level: int


@dataclass(frozen=True, slots=True)
class CallInfo:
    """Information about a recognizable function or method call."""

    callee: str
    lineno: int


@dataclass(frozen=True, slots=True)
class PythonAnalysisResult:
    """Complete deterministic analysis result for a Python file."""

    file_path: str
    functions: tuple[FunctionInfo, ...]
    classes: tuple[ClassInfo, ...]
    imports: tuple[ImportInfo, ...]
    calls: tuple[CallInfo, ...]


@dataclass(slots=True)
class _ClassRecord:
    """Mutable internal representation used during AST traversal."""

    name: str
    qualified_name: str
    lineno: int
    end_lineno: int
    methods: list[str] = field(default_factory=list)


def _get_end_lineno(node: ast.AST) -> int:
    """Return the ending source line of an AST node."""

    end_lineno = getattr(node, "end_lineno", None)

    if isinstance(end_lineno, int):
        return end_lineno

    return getattr(node, "lineno", 1)


def _get_callee_name(node: ast.expr) -> str | None:
    """Extract a recognizable dotted name from a call expression."""

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = _get_callee_name(node.value)

        if parent is not None:
            return f"{parent}.{node.attr}"

    return None


class _PythonASTVisitor(ast.NodeVisitor):
    """Collect structural information from a Python AST."""

    def __init__(self) -> None:
        self.functions: list[FunctionInfo] = []
        self.imports: list[ImportInfo] = []
        self.calls: list[CallInfo] = []

        self._scope: list[tuple[str, str]] = []
        self._class_stack: list[_ClassRecord] = []
        self._classes: list[_ClassRecord] = []

    def _qualified_name(self, name: str) -> str:
        """Build a qualified name from the current scope."""

        parts = [scope_name for _, scope_name in self._scope]
        parts.append(name)

        return ".".join(parts)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract a class and recursively visit its contents."""

        record = _ClassRecord(
            name=node.name,
            qualified_name=self._qualified_name(node.name),
            lineno=node.lineno,
            end_lineno=_get_end_lineno(node),
        )

        self._classes.append(record)

        for decorator in node.decorator_list:
            self.visit(decorator)

        for base in node.bases:
            self.visit(base)

        for keyword in node.keywords:
            self.visit(keyword.value)

        self._scope.append(("class", node.name))
        self._class_stack.append(record)

        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._class_stack.pop()
            self._scope.pop()

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool,
    ) -> None:
        """Extract a function or method and visit its contents."""

        is_method = bool(self._scope) and self._scope[-1][0] == "class"

        self.functions.append(
            FunctionInfo(
                name=node.name,
                qualified_name=self._qualified_name(node.name),
                lineno=node.lineno,
                end_lineno=_get_end_lineno(node),
                is_async=is_async,
                is_method=is_method,
            )
        )

        if is_method and self._class_stack:
            self._class_stack[-1].methods.append(node.name)

        self._scope.append(("function", node.name))

        try:
            self.generic_visit(node)
        finally:
            self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a normal Python function."""

        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        """Visit an asynchronous Python function."""

        self._visit_function(node, is_async=True)

    def visit_Import(self, node: ast.Import) -> None:
        """Extract direct imports."""

        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    module=None,
                    name=alias.name,
                    alias=alias.asname,
                    level=0,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract imports from a module."""

        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    module=node.module,
                    name=alias.name,
                    alias=alias.asname,
                    level=node.level,
                )
            )

    def visit_Call(self, node: ast.Call) -> None:
        """Extract recognizable function and method calls."""

        callee = _get_callee_name(node.func)

        if callee is not None:
            self.calls.append(
                CallInfo(
                    callee=callee,
                    lineno=node.lineno,
                )
            )

        self.generic_visit(node)

    def get_classes(self) -> tuple[ClassInfo, ...]:
        """Convert internal class records into immutable results."""

        return tuple(
            ClassInfo(
                name=record.name,
                qualified_name=record.qualified_name,
                lineno=record.lineno,
                end_lineno=record.end_lineno,
                methods=tuple(record.methods),
            )
            for record in self._classes
        )


class PythonASTAnalyzer:
    """Analyze Python source code without executing it."""

    def analyze(
        self,
        source: str,
        file_path: str,
    ) -> PythonAnalysisResult:
        """Parse Python source and return structural information."""

        if not isinstance(source, str):
            raise TypeError("Python source must be a string.")

        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("File path must be a non-empty string.")

        tree = ast.parse(
            source,
            filename=file_path,
        )

        visitor = _PythonASTVisitor()
        visitor.visit(tree)

        return PythonAnalysisResult(
            file_path=file_path,
            functions=tuple(visitor.functions),
            classes=visitor.get_classes(),
            imports=tuple(visitor.imports),
            calls=tuple(visitor.calls),
        )
