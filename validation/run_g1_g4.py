#!/usr/bin/env python3
"""G1–G10 Automated Validation Entry Point.

Runs the ten mandatory acceptance gates before full dataset generation.
Exit code 0 = all gates passed; non-zero = at least one gate failed.

Usage
-----
    python validation/run_g1_g4.py              # run all gates
    python validation/run_g1_g4.py G1 G3        # run specific gates only
    python validation/run_g1_g4.py --verbose    # pass -v to pytest
    python validation/run_g1_g4.py --tb=long    # verbose tracebacks
"""

import argparse
import subprocess
import sys
import pathlib
import time


GATES = {
    'G1':  ('NR Resource Configurations',   'tests/test_g1_nr_resources.py'),
    'G2':  ('8×8 UPA Beamforming',          'tests/test_g2_beamforming.py'),
    'G3':  ('Radar Link Budget',             'tests/test_g3_link_budget.py'),
    'G4':  ('CRLB / Measurement Covariance','tests/test_g4_crlb.py'),
    'G5':  ('Fixed-Threshold Detector',     'tests/test_g5_detector.py'),
    'G6':  ('Coverage: P_D & σ vs Range',  'tests/test_g6_coverage.py'),
    'G7':  ('Scan Timing & Target Motion',  'tests/test_g7_scan_motion.py'),
    'G8':  ('Multi-Target & Best-Beam',     'tests/test_g8_multitarget.py'),
    'G9':  ('Clutter Statistics',           'tests/test_g9_clutter.py'),
    'G10': ('HDF5 Dataset Integrity',       'tests/test_g10_dataset.py'),
}

ROOT = pathlib.Path(__file__).parent.parent  # rcatt_test_run/


def run_gate(gate_id: str, label: str, test_file: str,
             pytest_extra: list[str]) -> bool:
    """Run one gate; return True if it passes."""
    path = ROOT / test_file
    if not path.exists():
        print(f"  [MISSING] {path}")
        return False

    cmd = [sys.executable, '-m', 'pytest', str(path), '--tb=short',
           '-q'] + pytest_extra
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    passed = result.returncode == 0
    status = 'PASS' if passed else 'FAIL'
    # Extract the last summary line from pytest -q output
    lines = (result.stdout + result.stderr).strip().splitlines()
    summary = lines[-1] if lines else ''
    print(f"  [{status}] {gate_id}: {label}  ({elapsed:.2f}s)  {summary}")
    if not passed:
        # Show truncated failure details
        for line in lines:
            if line.startswith('FAILED') or line.startswith('E '):
                print(f"         {line}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description='Run G1–G10 validation suite.')
    parser.add_argument('gates', nargs='*', default=list(GATES),
                        help='Gates to run, e.g. G1 G2 (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Pass -v to pytest for per-test output')
    parser.add_argument('--tb', default='short',
                        choices=['short', 'long', 'no', 'line'],
                        help='Traceback style (default: short)')
    args = parser.parse_args()

    gates_to_run = [g.upper() for g in args.gates]
    unknown = [g for g in gates_to_run if g not in GATES]
    if unknown:
        parser.error(f"Unknown gates: {unknown}.  Valid: {list(GATES)}")

    pytest_extra = [f'--tb={args.tb}']
    if args.verbose:
        pytest_extra.append('-v')

    print("=" * 60)
    print("RCATT G1–G10 Validation Suite")
    print("=" * 60)

    results = {}
    for gate_id in gates_to_run:
        label, test_file = GATES[gate_id]
        results[gate_id] = run_gate(gate_id, label, test_file, pytest_extra)

    print("=" * 60)
    n_pass = sum(results.values())
    n_total = len(results)
    overall = 'ALL GATES PASSED' if n_pass == n_total else f'{n_total - n_pass} GATE(S) FAILED'
    print(f"Result: {overall}  ({n_pass}/{n_total} gates)")
    print("=" * 60)

    return 0 if n_pass == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
