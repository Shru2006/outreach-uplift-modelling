"""Run the whole project end to end with one command.

Run:  python -m src.pipeline
"""
from src import generate_data, run

STEPS = [
    ("1/2 Generate the synthetic outreach experiment", generate_data.main),
    ("2/2 Train learners, evaluate, write reports", run.main),
]


def main() -> None:
    for title, step in STEPS:
        print(f"\n=== {title} ===")
        step()


if __name__ == "__main__":
    main()
