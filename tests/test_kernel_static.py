"""Kernel notebook static checks: every dotted name must resolve.
Catches module-prefix leaks (F./L.) without a Kaggle round trip."""
import ast
import json
import re

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
NB = f"{PROJECT}/kernels/baseline-v0/notebook.ipynb"

ALLOWED_MODS = {"np", "pd", "torch", "os", "re", "json", "glob",
                "pickle", "math", "mp", "tqdm", "nn", "F",
                "CFG", "FuncF"}


def _cells():
    nb = json.load(open(NB))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_no_module_prefixes():
    for i, src in enumerate(_cells()):
        for m in re.finditer(r"^(\s*)([A-Z][A-Za-z0-9_]*)\.(\w+)", src, flags=re.M):
            mod = m.group(2)
            assert mod in ALLOWED_MODS, \
                f"cell {i}: unresolved module prefix {mod}.{m.group(3)}"


def test_run_globals_resolve():
    """Bare globals used in run cell must be defined in earlier cells or run itself."""
    import builtins
    cells = _cells()
    assert len(cells) >= 2
    defined = set(dir(builtins)) | {"np", "pd", "torch", "os", "re", "json",
                                    "glob", "pickle", "math", "gc", "nn", "F",
                                    "FuncF", "CFG", "_pk", "FRAG_TOL"}
    for src in cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Import):
                defined.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                defined.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
                    elif isinstance(t, ast.Tuple):
                        for e in t.elts:
                            if isinstance(e, ast.Name):
                                defined.add(e.id)
    run = cells[-1]
    tree = ast.parse(run)
    # names assigned anywhere in run cell are local; exclude them
    run_local = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            run_local.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    run_local.add(t.id)
    missing = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined and node.id not in run_local \
                    and not node.id.startswith("_") and node.id != "__import__":
                # allow function-local names: args and assignments inside main()
                missing.add(node.id)
    # filter: names bound inside any function in run cell (args + stores)
    main_bound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for a in list(node.args.args) + list(node.args.kwonlyargs):
                main_bound.add(a.arg)
            if node.args.vararg:
                main_bound.add(node.args.vararg.arg)
            if node.args.kwarg:
                main_bound.add(node.args.kwarg.arg)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    main_bound.add(sub.id)
                elif isinstance(sub, ast.Import):
                    main_bound.update(a.asname or a.name.split(".")[0] for a in sub.names)
                elif isinstance(sub, ast.ImportFrom):
                    main_bound.update(a.asname or a.name for a in sub.names)
    missing -= main_bound
    # remaining must be defined globals; report
    assert not missing, f"unresolved names in run cell: {sorted(missing)[:15]}"
