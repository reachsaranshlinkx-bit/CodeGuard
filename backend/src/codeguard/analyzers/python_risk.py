from __future__ import annotations

import ast
from dataclasses import dataclass, field

_ResolvedBinding = str | frozenset[str] | None


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
    bindings: dict[str, _ResolvedBinding] = field(default_factory=dict)
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
        canonical_name: _ResolvedBinding,
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

    def _resolve_name(self, name: str) -> _ResolvedBinding:
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

    def _resolve_expression(self, node: ast.expr) -> _ResolvedBinding:
        """Resolve names and dotted attributes to canonical names."""

        if isinstance(node, ast.Name):
            return self._resolve_name(node.id)

        if isinstance(node, ast.Attribute):
            parent = self._resolve_expression(node.value)

            if isinstance(parent, str):
                return f"{parent}.{node.attr}"

            if isinstance(parent, frozenset):
                return frozenset(f"{candidate}.{node.attr}" for candidate in parent)

        return None

    @staticmethod
    def _name_candidates(value: _ResolvedBinding) -> frozenset[str]:
        """Return all possible canonical names for a resolved binding."""

        if isinstance(value, str):
            return frozenset({value})

        if isinstance(value, frozenset):
            return value

        return frozenset()

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

        loader_names = self._name_candidates(self._resolve_expression(loader))

        safe_loaders = {
            "yaml.SafeLoader",
            "yaml.CSafeLoader",
            "yaml.BaseLoader",
        }

        return bool(loader_names) and loader_names.issubset(safe_loaders)

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

    @staticmethod
    def _merge_match_scope_bindings(
        states: list[dict[str, _ResolvedBinding]],
        fallback_bindings: dict[str, _ResolvedBinding],
    ) -> dict[str, _ResolvedBinding]:
        """Conservatively merge bindings from possible match outcomes."""

        missing = object()

        names = set().union(*(state.keys() for state in states))
        merged: dict[str, _ResolvedBinding] = {}

        for name in names:
            values = [state.get(name, missing) for state in states]

            canonical_names: set[str] = set()
            has_unresolved_path = False

            for value in values:
                if value is missing:
                    value = fallback_bindings.get(name, name)

                if isinstance(value, str):
                    canonical_names.add(value)

                elif isinstance(value, frozenset):
                    canonical_names.update(value)

                elif value is None:
                    has_unresolved_path = True

            if len(canonical_names) == 1:
                merged[name] = next(iter(canonical_names))

            elif len(canonical_names) > 1:
                merged[name] = frozenset(canonical_names)

            elif has_unresolved_path:
                merged[name] = None

        return merged

    def visit_Match(self, node: ast.Match) -> None:
        """Analyze match cases independently and merge their scope effects."""

        self.visit(node.subject)

        initial_bindings = [scope.bindings.copy() for scope in self._scopes]

        branch_states: list[list[dict[str, _ResolvedBinding]]] = []
        has_guaranteed_match = False

        for case in node.cases:
            # Reset every active scope before analyzing this case.
            for scope, bindings in zip(
                self._scopes,
                initial_bindings,
                strict=True,
            ):
                scope.bindings.clear()
                scope.bindings.update(bindings)

            self.visit(case.pattern)

            # Collect names captured by this pattern.
            collector = _LocalBindingCollector()
            collector.visit(case.pattern)

            for name in collector.get_local_names():
                self._bind_name(name, None)

            if case.guard is not None:
                self.visit(case.guard)

            for statement in case.body:
                self.visit(statement)

            # Preserve all binding changes made by this case.
            branch_states.append([scope.bindings.copy() for scope in self._scopes])

            # An unguarded irrefutable case guarantees that no
            # unmatched execution path remains.
            if case.guard is None and self._is_irrefutable_pattern(case.pattern):
                has_guaranteed_match = True
                break

        # When no case is guaranteed to match, preserve the possibility
        # that all patterns fail and the original bindings remain.
        if not has_guaranteed_match:
            branch_states.append(initial_bindings)

        # Merge the possible states independently for every active scope.
        for scope_index, scope in enumerate(self._scopes):
            scope_states = [state[scope_index] for state in branch_states]

            names = set().union(*(state.keys() for state in scope_states))
            fallback_bindings: dict[str, _ResolvedBinding] = {}

            for name in names:
                fallback: _ResolvedBinding = name

                # Resolve the name through enclosing scopes.
                for outer_index in range(scope_index - 1, -1, -1):
                    outer_scope = self._scopes[outer_index]

                    # Class namespaces are not closures for methods.
                    if outer_scope.kind == "class" and any(
                        inner.kind == "function"
                        for inner in self._scopes[outer_index + 1 : scope_index + 1]
                    ):
                        continue

                    if name in initial_bindings[outer_index]:
                        fallback = initial_bindings[outer_index][name]
                        break

                fallback_bindings[name] = fallback

            merged_bindings = self._merge_match_scope_bindings(
                scope_states,
                fallback_bindings,
            )

            scope.bindings.clear()
            scope.bindings.update(merged_bindings)

    @staticmethod
    def _is_irrefutable_pattern(pattern: ast.pattern) -> bool:
        """Return whether a pattern is guaranteed to match."""

        if isinstance(pattern, ast.MatchAs):
            return pattern.pattern is None or (
                _PythonRiskVisitor._is_irrefutable_pattern(pattern.pattern)
            )

        if isinstance(pattern, ast.MatchOr):
            return any(
                _PythonRiskVisitor._is_irrefutable_pattern(item)
                for item in pattern.patterns
            )

        return False

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

        local_scope: dict[str, _ResolvedBinding] = {name: None for name in local_names}

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

        previous_binding: _ResolvedBinding = None
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

        callees = self._name_candidates(self._resolve_expression(node.func))

        # PY001: Dynamic code execution.
        dynamic_callees = callees & {
            "eval",
            "exec",
            "builtins.eval",
            "builtins.exec",
        }

        if dynamic_callees:
            function_name = sorted(dynamic_callees)[0].rsplit(".", maxsplit=1)[-1]

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
        subprocess_callees = callees & {
            "subprocess.run",
            "subprocess.call",
            "subprocess.Popen",
            "subprocess.check_call",
            "subprocess.check_output",
        }

        if "os.system" in callees or (
            subprocess_callees and self._has_shell_true(node)
        ):
            self._add_finding(
                node,
                rule_id="PY002",
                message=(
                    "Command execution can introduce command-injection risks. "
                    "Prefer safer subprocess APIs."
                ),
                cwe="CWE-78",
            )

        # PY003: Unsafe pickle deserialization.
        pickle_callees = callees & {
            "pickle.load",
            "pickle.loads",
        }

        if pickle_callees:
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
        if "yaml.load" in callees and not self._has_safe_yaml_loader(node):
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
