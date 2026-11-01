# Repository validation

Validation performed on November 1, 2026 after assembling the publication
snapshot:

| Check | Result |
| --- | --- |
| Python syntax compilation | PASS, 104 files |
| JSON parse | PASS, 20 files |
| Shell syntax (`bash -n`) | PASS, 15 files |
| Forbidden binary/evidence extensions in publish set | none found |
| Obvious credential/private-key pattern scan | no hits |
| Symlinks | none |
| Largest file | approximately 140 kB |
| Candidate repository size before Git metadata | approximately 1.8 MB |
| Local Git repository | initialized on branch `main` |

The four policy unit-test modules require project dependencies such as NumPy
and OpenCV, which are not installed in the packaging workstation's Python 3.14
environment. They were therefore not re-executed during the final packaging
step. The source integration report from the dependency-complete DGX
environment records 52/52 main tests and 70/70 post-drop SAFE_HOME regression
tests passing before the final CSI runs.

This packaging validation is not a robot or perception rerun. No physical
motion, training, or video generation was performed.

