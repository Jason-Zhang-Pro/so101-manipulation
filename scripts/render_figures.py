#!/usr/bin/env python3
"""Render portfolio figures from explicitly reported project results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "configs/reported_results.json").read_text())
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(["Seen centres\n128/150", "Unseen positions\ntrial count unavailable"],
                [100 * data["seen"]["successes"] / data["seen"]["trials"], 100 * data["unseen"]["rate"]],
                color=["#087f8c", "#bd4b58"], width=.55)
    axes[0].set_title("Spatial generalisation")
    axes[1].bar([x["condition"]+f"\n{x['successes']}/{x['trials']}" for x in data["lighting"]],
                [100*x["successes"]/x["trials"] for x in data["lighting"]],
                color=["#087f8c", "#d09b31", "#bd4b58"], width=.6)
    axes[1].set_title("Lighting sensitivity")
    for ax in axes:
        ax.set_ylim(0, 105); ax.set_ylabel("Reported grasp success (%)")
        for bar in ax.patches:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{bar.get_height():.1f}%", ha="center")
    fig.suptitle("SO-101 ACT: Real-Robot Evaluation", fontsize=17)
    fig.text(.5,.02,"Project-log results; grasp success, not full pick-and-place completion",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94)); fig.savefig(out/"experiment_results.png",dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4.5))
    sim=data["simulation"]
    x=[0,1]
    ax.bar([i-.18 for i in x],[sim["gui_rtf"]["original"],sim["headless_rtf"]["original"]],width=.35,color="#9b9b9b",label="Original")
    ax.bar([i+.18 for i in x],[sim["gui_rtf"]["optimized"],sim["headless_rtf"]["optimized"]],width=.35,color="#087f8c",label="Optimized")
    ax.set_xticks(x,["GUI enabled","Headless"]);ax.set_ylim(0,1.12);ax.set_ylabel("Real-time factor");ax.legend()
    ax.set_title("Four-world simulation: reported free-run benchmark")
    for b in ax.patches:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.02,f"{b.get_height():.3f}",ha="center")
    fig.text(.5,.02,"ACT/PPO excluded; optimized parallel implementation not included in this snapshot",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(out/"simulation_results.png",dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,3.8));ax.set_xlim(0,12);ax.set_ylim(0,3.8);ax.axis("off")
    labels=[("Teleoperation", "Leader + follower\nDual RGB + qpos"),("ACT learning","Demonstrations\nTraining + deployment"),("Evaluation","Seen / unseen\nLighting sensitivity"),("Simulation","ROS 2 / Gazebo\nIK + PPO reach"),("Refinement","ACT-to-RL\nOngoing")]
    for i,(title,sub) in enumerate(labels):
        x=.15+i*2.4;ax.add_patch(Rectangle((x,1.1),2.1,1.5,facecolor="#edf5f4" if i<4 else "#fff4df",edgecolor="#087f8c" if i<4 else "#b7872c"))
        ax.text(x+1.05,2.15,title,ha="center",fontsize=12,weight="bold");ax.text(x+1.05,1.65,sub,ha="center",fontsize=10)
        if i<4:ax.add_patch(FancyArrowPatch((x+2.13,1.85),(x+2.36,1.85),arrowstyle="->",mutation_scale=15,color="#555555"))
    ax.text(6,3.25,"SO-101 Project Pipeline",ha="center",fontsize=18)
    ax.text(6,.4,"Measured ACT baseline and simulation groundwork; transfer validation remains ongoing",ha="center",fontsize=10)
    fig.tight_layout();fig.savefig(out/"system_pipeline.png",dpi=160);plt.close(fig)
    print("Rendered 3 figures")


if __name__ == "__main__":main()
