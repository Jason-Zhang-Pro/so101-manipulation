#!/usr/bin/env python3
"""Offline regression checks for command generation and outcome accounting."""
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("act_workflow",ROOT/"learning/act/workflow.py")
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);cfg=json.loads((ROOT/"configs/act_workflow.json").read_text())
        cfg["dataset"]["root"]=str(root/"dataset")
        path=root/"workflow.json";path.write_text(json.dumps(cfg))
        output=io.StringIO()
        with contextlib.redirect_stdout(output):w.main(["collect","--config",str(path)])
        assert "--resume=false" in output.getvalue() and "Preview only" in output.getvalue()
        assert not (root/"dataset").exists()
        meta=root/"dataset/meta";meta.mkdir(parents=True)
        features={"action":{},"observation.state":{},"observation.images.fixed":{},"observation.images.handeye":{}}
        (meta/"info.json").write_text(json.dumps({"fps":15,"total_episodes":1,"features":features}))
        output=io.StringIO()
        with contextlib.redirect_stdout(output):w.main(["train","--config",str(path),"--run","offline_test_fixture"])
        command=output.getvalue()
        assert "--policy.optimizer_lr=1e-05" in command and "--policy.n_action_steps=1" in command
        assert not (ROOT/"outputs/train/offline_test_fixture").exists()
        trials=root/"trials.csv"
        with trials.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=w.TRIAL_COLUMNS);writer.writeheader()
            for i,(g,p) in enumerate([("1","1"),("1","0"),("0","")]):
                writer.writerow({"trial_id":str(i),"cell":"P1","position_set":"seen","lighting":"fixed","grasp_success":g,"place_success":p})
        summary=w.summarize_trials(trials)[0]
        assert summary["grasp_rate"]==2/3 and summary["full_task_rate"]==.5 and summary["full_task_trials"]==2
        assert summary["wilson95"][0]<2/3<summary["wilson95"][1]
    print("PASS: collection preview, training preset, no side effects, grasp/place accounting")


if __name__=="__main__":main()
