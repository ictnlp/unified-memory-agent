import copy
import json
import random
from pathlib import Path


DEFAULT_BENCHES = [
    "synth-ss2",
    "synth-ss3",
    "synth-ss4",
    "synth-ss5",
    "synth-ss10",
    "synth-ss20",
    "synth-ss30",
    "synth-ss40",
    "synth-ss50",
]
DEFAULT_EXPECTED_NUM_Q = 10
DEFAULT_SEED = 114514
DATA_DIR = Path(__file__).resolve().parent


def build_small_valset(
    data: list[dict],
    *,
    expected_num_q: int = DEFAULT_EXPECTED_NUM_Q,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Sample a small validation split capped by question count."""
    rng = random.Random(seed)
    shuffled = copy.deepcopy(data)
    rng.shuffle(shuffled)

    small_valset: list[dict] = []
    num_q_sofar = 0

    for sample in shuffled:
        questions = list(sample.get("questions", []))
        remaining = expected_num_q - num_q_sofar
        if remaining <= 0:
            break

        sample_copy = copy.deepcopy(sample)
        if len(questions) > remaining:
            sample_copy["questions"] = rng.sample(questions, remaining)
            small_valset.append(sample_copy)
            break

        small_valset.append(sample_copy)
        num_q_sofar += len(questions)

    return small_valset


def get_small_valset_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    return input_path.parent / "small_valsets" / input_path.name


def generate_small_valset(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    expected_num_q: int = DEFAULT_EXPECTED_NUM_Q,
    seed: int = DEFAULT_SEED,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else get_small_valset_path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    small_valset = build_small_valset(data, expected_num_q=expected_num_q, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(small_valset, f, indent=4, ensure_ascii=False)

    return output_path


def ensure_small_valset(
    input_path: str | Path,
    *,
    expected_num_q: int = DEFAULT_EXPECTED_NUM_Q,
    seed: int = DEFAULT_SEED,
) -> Path:
    input_path = Path(input_path)
    output_path = get_small_valset_path(input_path)
    if output_path.exists():
        return output_path

    print(f"[SmallValset] {output_path} not found. Generating from {input_path}...")
    return generate_small_valset(
        input_path,
        output_path,
        expected_num_q=expected_num_q,
        seed=seed,
    )


def main() -> None:
    for bench in DEFAULT_BENCHES:
        input_path = DATA_DIR / f"processed_{bench}.json"
        output_path = generate_small_valset(input_path)
        print(f"[SmallValset] Wrote {output_path}")


if __name__ == "__main__":
    main()
