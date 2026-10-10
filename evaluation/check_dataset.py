#!/usr/bin/env python3
"""Read-only integrity audit for a LeRobot v2.1 episode dataset."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--skip-video",action="store_true",help="Audit tabular data only")
    args=parser.parse_args();root=args.root.expanduser().resolve()
    info=json.loads((root/"meta/info.json").read_text())
    metadata={r["episode_index"]:r for r in map(json.loads,(root/"meta/episodes.jsonl").read_text().splitlines())}
    rows=[]
    for ep in range(info["total_episodes"]):
        chunk=ep//info.get("chunks_size",1000)
        path=root/info["data_path"].format(episode_chunk=chunk,episode_index=ep)
        problems=[]
        try:
            data=pq.read_table(path).to_pydict();n=len(data["timestamp"])
            if n!=metadata[ep]["length"]:problems.append("metadata length mismatch")
            ts=np.asarray(data["timestamp"])
            if not np.isfinite(ts).all():problems.append("nonfinite timestamp")
            if len(ts)>1 and not np.allclose(np.diff(ts),1/info["fps"],atol=1e-4):problems.append("timestamp step mismatch")
            if n and abs(float(ts[0]))>1e-4:problems.append("timestamp does not start at zero")
            if not np.array_equal(np.asarray(data["frame_index"]),np.arange(n)):problems.append("frame index mismatch")
            if not np.all(np.asarray(data["episode_index"])==ep):problems.append("episode index mismatch")
            for key in ("action","observation.state"):
                x=np.asarray(data[key])
                if x.shape!=(n,6) or not np.isfinite(x).all():problems.append(f"invalid {key}")
            if not args.skip_video:
                for key,value in info["features"].items():
                    if value["dtype"]!="video":continue
                    video=root/info["video_path"].format(episode_chunk=chunk,episode_index=ep,video_key=key)
                    count=0
                    with av.open(str(video)) as container:
                        for f in container.decode(video=0):
                            count+=1
                            if [f.height,f.width,3]!=value["shape"]:problems.append(f"resolution mismatch: {key}");break
                    if count!=n:problems.append(f"video count mismatch: {key} ({count}/{n})")
        except Exception as error:
            problems.append(f"{type(error).__name__}: {error}")
        rows.append({"episode":ep,"issues":problems})
    result={"episodes":len(rows),"video_checked":not args.skip_video,
            "failed_episodes":[r for r in rows if r["issues"]]}
    print(json.dumps(result,indent=2))
    if result["failed_episodes"]:raise SystemExit(1)


if __name__=="__main__":main()
