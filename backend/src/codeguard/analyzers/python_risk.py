from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RiskFinding:
    """A structured static security-risk finding."""

    rule_id: str
    severity: str
    message: str
    file_path: str
    lineno: int
    cwe: str


@dataclass(slots=True)
class _Scope:
    """A lexical scope containing imported aliases and local bindings."""

    kind: str
    bindings: dict[str, str | None] = field(default_factory=dict)
    global_names: set[str] = field(default_factory=set)
    nonlocal_names: set[str] = field(default_factory=set)


class _LocalBindingCollector(ast.NodeVisitor):
    """Collect names bound in a function's local scope."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            self.names.add(local_name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.names.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        """Collect names captured by match-as patterns."""

        if node.name is not None:
            self.names.add(node.name)

        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        """Collect names captured by starred match patterns."""

        if node.name is not None:
            self.names.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        """Collect names captured by mapping rest patterns."""

        if node.rest is not None:
            self.names.add(node.rest)

        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # Lambda bodies have their own local scope.
        return

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.names.add(node.name)

        if node.type is not None:
            self.visit(node.type)

        for statement in node.body:
            self.visit(statement)

    def _visit_comprehension_expressions(
        self,
        generators: list[ast.comprehension],
        expressions: tuple[ast.expr, ...],
    ) -> None:
        # Comprehension targets belong to their own scope.
        # Still inspect iterables, conditions and expressions for bindings.
        for generator in generators:
            self.visit(generator.iter)

            for condition in generator.ifs:
                self.visit(condition)

        for expression in expressions:
            self.visit(expression)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension_expressions(
            node.generators,
            (node.elt,),
        )

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension_expressions(
            node.generators,
            (node.elt,),
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension_expressions(
            node.generators,
            (node.elt,),
        )

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension_expressions(
            node.generators,
            (node.key, node.value),
        )

    def get_local_names(self) -> set[str]:
        """Return locally bound names excluding global and nonlocal names."""

        return self.names - self.global_names - self.nonlocal_names


class _PythonRiskVisitor(ast.NodeVisitor):
    """Detect selected security-sensitive Python operations."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.findings: list[RiskFinding] = []
        self._scopes: list[_Scope] = [_Scope(kind="module")]

    def _bind_name(
        self,
        name: str,
        canonical_name: str | None,
    ) -> None:
        """Bind or shadow a name in the appropriate scope."""

        current = self._scopes[-1]

        if name in current.global_names:
            self._scopes[0].bindings[name] = canonical_name
            return

        if name in current.nonlocal_names:
            for scope in reversed(self._scopes[:-1]):
                if scope.kind == "function":
                    scope.bindings[name] = canonical_name
                    return

        current.bindings[name] = canonical_name

    def _bind_target(self, target: ast.expr) -> None:
        """Invalidate imported aliases when variables are reassigned."""

        if isinstance(target, ast.Name):
            self._bind_name(target.id, None)

        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_target(element)

        elif isinstance(target, ast.Starred):
            self._bind_target(target.value)

    def _resolve_name(self, name: str) -> str | None:
        """Resolve an imported name through enclosing lexical scopes."""

        for index in range(len(self._scopes) - 1, -1, -1):
            scope = self._scopes[index]

            # Class namespaces are not lexical closures for methods.
            if scope.kind == "class" and any(
                inner.kind == "function" for inner in self._scopes[index + 1 :]
            ):
                continue

            if name in scope.bindings:
                return scope.bindings[name]

        # Preserve direct built-in and fully qualified call recognition.
        return name

    def _resolve_expression(self, node: ast.expr) -> str | None:
        """Resolve names and dotted attributes to canonical names."""

        if isinstance(node, ast.Name):
            return self._resolve_name(node.id)

        if isinstance(node, ast.Attribute):
            parent = self._resolve_expression(node.value)

            if parent is not None:
                return f"{parent}.{node.attr}"

        return None

    def _add_finding(
        self,
        node: ast.Call,
        rule_id: str,
        message: str,
        cwe: str,
    ) -> None:
        """Add a structured risk finding."""

        self.findings.append(
            RiskFinding(
                rule_id=rule_id,
                severity="high",
                message=message,
                file_path=self.file_path,
                lineno=node.lineno,
                cwe=cwe,
            )
        )

    @staticmethod
    def _has_shell_true(node: ast.Call) -> bool:
        """Check whether shell=True is explicitly specified."""

        return any(
            keyword.arg == "shell"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )

    def _has_safe_yaml_loader(self, node: ast.Call) -> bool:
        """Check whether yaml.load uses a recognized safe loader."""

        loader: ast.expr | None = None

        for keyword in node.keywords:
            if keyword.arg == "Loader":
                loader = keyword.value
                break

        if loader is None and len(node.args) >= 2:
            loader = node.args[1]

        if loader is None:
            return False

        loader_name = self._resolve_expression(loader)

        return loader_name in {
            "yaml.SafeLoader",
            "yaml.CSafeLoader",
            "yaml.BaseLoader",
        }

    def visit_Import(self, node: ast.Import) -> None:
        """Resolve module imports and their aliases."""

        for alias in node.names:
            if alias.asname is not None:
                local_name = alias.asname
                canonical_name = alias.name
            else:
                local_name = alias.name.split(".", maxsplit=1)[0]
                canonical_name = local_name

            self._bind_name(local_name, canonical_name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Resolve direct imports and their aliases."""

        for alias in node.names:
            if alias.name == "*":
                continue

            local_name = alias.asname or alias.name

            if node.level != 0 or node.module is None:
                canonical_name = None
            else:
                canonical_name = f"{node.module}.{alias.name}"

            self._bind_name(local_name, canonical_name)

    def visit_Global(self, node: ast.Global) -> None:
        self._scopes[-1].global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._scopes[-1].nonlocal_names.update(node.names)

    def _visit_function_header(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Visit expressions evaluated in the enclosing scope."""

        for decorator in node.decorator_list:
            self.visit(decorator)

        for default_expr in node.args.defaults:
            self.visit(default_expr)

        for keyword_default in node.args.kw_defaults:
            if keyword_default is not None:
                self.visit(keyword_default)

        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )

        for argument in arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)

        if node.args.vararg is not None:
            if node.args.vararg.annotation is not None:
                self.visit(node.args.vararg.annotation)

        if node.args.kwarg is not None:
            if node.args.kwarg.annotation is not None:
                self.visit(node.args.kwarg.annotation)

        if node.returns is not None:
            self.visit(node.returns)

    def _visit_function_body(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Visit a function using an isolated lexical scope."""

        self._visit_function_header(node)
        self._bind_name(node.name, None)

        collector = _LocalBindingCollector()

        for statement in node.body:
            collector.visit(statement)

        local_names = collector.get_local_names()

        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )

        local_names.update(argument.arg for argument in arguments)

        if node.args.vararg is not None:
            local_names.add(node.args.vararg.arg)

        if node.args.kwarg is not None:
            local_names.add(node.args.kwarg.arg)

        local_scope: dict[str, str | None] = {name: None for name in local_names}

        self._scopes.append(
            _Scope(
                kind="function",
                bindings=local_scope,
                global_names=collector.global_names.copy(),
                nonlocal_names=collector.nonlocal_names.copy(),
            )
        )

        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_body(node)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        self._visit_function_body(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Handle lambda parameters as local bindings."""

        for default_expr in node.args.defaults:
            self.visit(default_expr)

        for keyword_default in node.args.kw_defaults:
            if keyword_default is not None:
                self.visit(keyword_default)

        names = {
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }

        if node.args.vararg is not None:
            names.add(node.args.vararg.arg)

        if node.args.kwarg is not None:
            names.add(node.args.kwarg.arg)

        self._scopes.append(
            _Scope(
                kind="function",
                bindings={name: None for name in names},
            )
        )

        try:
            self.visit(node.body)
        finally:
            self._scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class using a separate class namespace."""

        for decorator in node.decorator_list:
            self.visit(decorator)

        for base in node.bases:
            self.visit(base)

        for keyword in node.keywords:
            self.visit(keyword.value)

        self._bind_name(node.name, None)
        self._scopes.append(_Scope(kind="class"))

        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._scopes.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)

        for target in node.targets:
            self.visit(target)
            self._bind_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)

        if node.value is not None:
            self.visit(node.value)

        self._bind_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.target)
        self.visit(node.value)
        self._bind_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_target(node.target)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._bind_target(target)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._bind_target(node.target)

        for statement in node.body:
            self.visit(statement)

        for statement in node.orelse:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._bind_target(node.target)

        for statement in node.body:
            self.visit(statement)

        for statement in node.orelse:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with_items(node.items, node.body)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with_items(node.items, node.body)

    def _visit_with_items(
        self,
        items: list[ast.withitem],
        body: list[ast.stmt],
    ) -> None:
        for item in items:
            self.visit(item.context_expr)

            if item.optional_vars is not None:
                self._bind_target(item.optional_vars)

        for statement in body:
            self.visit(statement)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)

        previous_binding: str | None = None
        had_previous_binding = False

        if node.name is not None:
            bindings = self._scopes[-1].bindings
            had_previous_binding = node.name in bindings
            previous_binding = bindings.get(node.name)
            self._bind_name(node.name, None)

        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            if node.name is not None:
                bindings = self._scopes[-1].bindings

                if had_previous_binding:
                    bindings[node.name] = previous_binding
                else:
                    bindings.pop(node.name, None)

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        expressions: tuple[ast.expr, ...],
    ) -> None:
        self._scopes.append(_Scope(kind="comprehension"))

        try:
            for generator in generators:
                self.visit(generator.iter)
                self._bind_target(generator.target)

                for condition in generator.ifs:
                    self.visit(condition)

            for expression in expressions:
                self.visit(expression)
        finally:
            self._scopes.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(
            node.generators,
            (node.key, node.value),
        )

    def visit_Call(self, node: ast.Call) -> None:
        """Inspect calls after resolving imported aliases."""

        callee = self._resolve_expression(node.func)

        # PY001: Dynamic code execution.
        if callee in {
            "eval",
            "exec",
            "builtins.eval",
            "builtins.exec",
        }:
            function_name = callee.rsplit(".", maxsplit=1)[-1]

            self._add_finding(
                node,
                rule_id="PY001",
                message=(
                    f"{function_name}() executes dynamically supplied code. "
                    "Avoid executing untrusted input."
                ),
                cwe="CWE-95",
            )

        # PY002: Potentially unsafe command execution.
        elif callee == "os.system":
            self._add_finding(
                node,
                rule_id="PY002",
                message=(
                    "os.system() executes a shell command. "
                    "Prefer safer subprocess APIs."
                ),
                cwe="CWE-78",
            )

        elif callee in {
            "subprocess.run",
            "subprocess.call",
            "subprocess.Popen",
            "subprocess.check_call",
            "subprocess.check_output",
        } and self._has_shell_true(node):
            self._add_finding(
                node,
                rule_id="PY002",
                message=(
                    "Subprocess execution with shell=True can "
                    "introduce command-injection risks."
                ),
                cwe="CWE-78",
            )

        # PY003: Unsafe pickle deserialization.
        elif callee in {
            "pickle.load",
            "pickle.loads",
        }:
            self._add_finding(
                node,
                rule_id="PY003",
                message=(
                    "Pickle deserialization can execute malicious "
                    "code when processing untrusted data."
                ),
                cwe="CWE-502",
            )

        # PY004: Potentially unsafe YAML deserialization.
        elif callee == "yaml.load" and not self._has_safe_yaml_loader(node):
            self._add_finding(
                node,
                rule_id="PY004",
                message=(
                    "yaml.load() without a recognized safe loader "
                    "can deserialize unsafe Python objects. "
                    "Prefer yaml.safe_load()."
                ),
                cwe="CWE-502",
            )

        self.generic_visit(node)


class PythonRiskAnalyzer:
    """Perform deterministic AST-based security-risk detection."""

    def analyze(
        self,
        source: str,
        file_path: str,
    ) -> tuple[RiskFinding, ...]:
        """Analyze Python source without executing repository code."""

        if not isinstance(source, str):
            raise TypeError("Python source must be a string.")

        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("File path must be a non-empty string.")

        tree = ast.parse(
            source,
            filename=file_path,
        )

        visitor = _PythonRiskVisitor(file_path)
        visitor.visit(tree)

        return tuple(visitor.findings)
