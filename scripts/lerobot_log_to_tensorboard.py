#!/usr/bin/env python3

import argparse
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

METRICS = {
    "train/loss": re.compile(rf"\bloss:({NUMBER})"),
    "train/gradient_norm": re.compile(rf"\bgrdn:({NUMBER})"),
    "train/learning_rate": re.compile(rf"\blr:({NUMBER})"),
    "performance/update_seconds": re.compile(rf"\bupdt_s:({NUMBER})"),
    "performance/data_seconds": re.compile(rf"\bdata_s:({NUMBER})"),
    "train/epoch": re.compile(rf"\bepch:({NUMBER})"),
}


def parse_step(text: str) -> int | None:
    match = re.search(r"\bstep:(\d+)([KMB]?)\b", text)
    if not match:
        return None

    value = int(match.group(1))
    suffix = match.group(2)

    multiplier = {
        "": 1,
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
    }[suffix]

    return value * multiplier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()

    log_file = Path(args.log_file)
    log_dir = Path(args.log_dir)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"[TensorBoard writer] 读取：{log_file}")
    print(f"[TensorBoard writer] 写入：{log_dir}")

    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as stream:
            while True:
                line = stream.readline()

                if not line:
                    time.sleep(0.2)
                    continue

                step = parse_step(line)
                if step is not None:
                    wrote_metric = False

                    for tag, pattern in METRICS.items():
                        match = pattern.search(line)
                        if match:
                            writer.add_scalar(tag, float(match.group(1)), step)
                            wrote_metric = True

                    if wrote_metric:
                        writer.flush()

                if "End of training" in line:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
