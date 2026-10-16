"""Drives predicate_cli to seed all 24 threshold mutations and check the
regression gate catches each one, then runs the unmutated predicates 50
times to confirm 0 false alarms.

Usage: python3 run_seeded_regressions.py <corpus.tsv> <predicate_cli_path>
"""
import subprocess
import sys

NUM_MUTATIONS = 24


def run(cli, args):
    result = subprocess.run([cli] + args, capture_output=True, text=True)
    return result.returncode, result.stdout


def main():
    corpus_path, cli = sys.argv[1], sys.argv[2]
    baseline_path = "data/regression_baseline.tsv"

    rc, out = run(cli, ["run", corpus_path, baseline_path])
    assert rc == 0, out
    print("baseline run complete")

    caught = 0
    for mutation_id in range(NUM_MUTATIONS):
        mutated_path = f"data/regression_mutated_{mutation_id}.tsv"
        rc, out = run(cli, ["run", corpus_path, mutated_path, str(mutation_id)])
        assert rc == 0, out
        rc, out = run(cli, ["diff", corpus_path, baseline_path, mutated_path])
        detected = rc == 1
        caught += detected
        failing_class = ""
        for line in out.splitlines():
            if line.startswith("REGRESSION_DETECTED=true"):
                failing_class = line.split("FAILING_EVENT_CLASS=")[1]
        print(f"mutation {mutation_id}: detected={detected} failing_class={failing_class or '<none>'}")

    print(f"SEEDED_REGRESSIONS_CAUGHT={caught}/{NUM_MUTATIONS}")

    false_alarms = 0
    num_unchanged_runs = 50
    for i in range(num_unchanged_runs):
        rerun_path = f"data/regression_rerun_{i}.tsv"
        rc, out = run(cli, ["run", corpus_path, rerun_path])
        assert rc == 0, out
        rc, out = run(cli, ["diff", corpus_path, baseline_path, rerun_path])
        false_alarms += rc == 1

    print(f"FALSE_ALARMS_OVER_UNCHANGED_RUNS={false_alarms}/{num_unchanged_runs}")


if __name__ == "__main__":
    main()
