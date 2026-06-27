from __future__ import annotations

from omnifold_publication.cli import main
from omnifold_publication import write_package


def test_cli_validate_and_summary(tmp_path, source_hdf, capsys):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )

    assert main(["validate", str(package_dir)]) == 0
    validate_output = capsys.readouterr().out
    assert "Validation passed" in validate_output

    assert main(["summary", str(package_dir)]) == 0
    summary_output = capsys.readouterr().out
    assert "Format version" in summary_output
    assert "Systematics: replica" in summary_output
