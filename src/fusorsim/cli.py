"""Console entry points. The component modules are still run-as-script style, so each
entry point runs its module as __main__ with the (possibly case-expanded) argv."""
import runpy
import sys


def _expand_case(argv):
    """Replace `--case cases/X.yaml` by the flags it contains (explicit CLI flags win,
    because argparse keeps the LAST occurrence and they are appended after the case)."""
    if "--case" not in argv:
        return argv
    i = argv.index("--case")
    path = argv[i + 1]
    rest = argv[:i] + argv[i + 2:]
    with open(path) as f:
        text = f.read()
    try:
        import yaml
        case = yaml.safe_load(text) or {}
    except ImportError:   # flat `key: value` files only (all our case files are)
        case = {}
        for ln in text.splitlines():
            ln = ln.split("#", 1)[0].strip()
            if ":" in ln:
                k, v = ln.split(":", 1)
                v = v.strip().strip('"').strip("'")
                case[k.strip()] = {"true": True, "false": False}.get(v.lower(), v) if v else None
    flags = []
    for k, v in case.items():
        if k in ("name", "description"):
            continue
        flag = "--" + str(k).replace("_", "-")
        if v is True:
            flags.append(flag)
        elif v is not False and v is not None:
            flags += [flag, str(v)]
    return flags + rest


def _run(module, argv=None):
    sys.argv = [module] + _expand_case(list(sys.argv[1:] if argv is None else argv))
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def warpx():
    _run("fusorsim.warpx.driver")


def post():
    _run("fusorsim.analysis.postprocess")


def viz():
    _run("fusorsim.viz.render")


def openmc():
    _run("fusorsim.openmc.dose")
