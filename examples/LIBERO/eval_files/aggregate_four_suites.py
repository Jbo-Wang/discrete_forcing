#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


SUITES = (
    ("libero_spatial", "Spatial"),
    ("libero_object", "Object"),
    ("libero_goal", "Goal"),
    ("libero_10", "Long"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate four LIBERO evaluation summaries.")
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main():
    output_dir = parse_args().output_dir.resolve()
    suite_results = []
    total_episodes = 0
    total_successes = 0

    for suite_name, display_name in SUITES:
        summary_path = output_dir / suite_name / "evaluation_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing suite summary: {summary_path}")

        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)

        episodes = int(summary["total_episodes"])
        successes = int(summary["total_successes"])
        success_rate = float(summary["total_success_rate"])
        suite_results.append(
            {
                "suite": suite_name,
                "display_name": display_name,
                "episodes": episodes,
                "successes": successes,
                "success_rate": success_rate,
                "summary_path": str(summary_path),
            }
        )
        total_episodes += episodes
        total_successes += successes

    macro_average = sum(item["success_rate"] for item in suite_results) / len(suite_results)
    micro_average = total_successes / total_episodes
    combined = {
        "checkpoint": json.loads(
            (output_dir / "run_metadata.json").read_text(encoding="utf-8")
        )["checkpoint"],
        "suite_results": suite_results,
        "macro_average": macro_average,
        "micro_average": micro_average,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
    }

    json_path = output_dir / "four_suite_summary.json"
    json_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "four_suite_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("suite", "display_name", "episodes", "successes", "success_rate")
        )
        writer.writeheader()
        for result in suite_results:
            writer.writerow(
                {
                    key: result[key]
                    for key in ("suite", "display_name", "episodes", "successes", "success_rate")
                }
            )

    lines = ["LIBERO four-suite evaluation"]
    for item in suite_results:
        lines.append(
            f"{item['display_name']:<7} {item['successes']:>3}/{item['episodes']:<3} "
            f"{100.0 * item['success_rate']:.2f}%"
        )
    lines.extend(
        (
            f"Macro average: {100.0 * macro_average:.2f}%",
            f"Micro average: {total_successes}/{total_episodes} ({100.0 * micro_average:.2f}%)",
        )
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "four_suite_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
