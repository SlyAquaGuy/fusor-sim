from fusorsim.cli import _expand_case


def test_case_expands_to_flags_and_cli_wins(tmp_path):
    c = tmp_path / "c.yaml"
    c.write_text("name: x\ndescription: y\nvoltage: -40000\nreflection: \"off\"\npreflight: true\n")
    argv = _expand_case(["--case", str(c), "--voltage", "-50000"])
    assert argv[:2] == ["--voltage", "-40000"] and argv.index("--voltage", 2) > 2   # explicit flag comes later => wins
    assert "--reflection" in argv and "off" in argv and "--preflight" in argv
    assert "--name" not in argv and "--description" not in argv
