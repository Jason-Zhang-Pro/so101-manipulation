#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


METRIC_RE = re.compile(
    r"\bstep:(?P<step>\d+).*?"
    r"\bloss:(?P<loss>[-+0-9.eE]+).*?"
    r"\bgrdn:(?P<grad_norm>[-+0-9.eE]+).*?"
    r"\blr:(?P<lr>[-+0-9.eE]+)"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    args = parser.parse_args()

    logdir = Path(args.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(logdir))
    try:
        for line in sys.stdin:
            sys.stdout.write(line)
            sys.stdout.flush()

            match = METRIC_RE.search(line)
            if not match:
                continue

            step = int(match.group("step"))
            writer.add_scalar("train/loss", float(match.group("loss")), step)
            writer.add_scalar("train/grad_norm", float(match.group("grad_norm")), step)
            writer.add_scalar("train/lr", float(match.group("lr")), step)
            writer.flush()
    finally:
        writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
