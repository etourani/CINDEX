# usage: python boo.py ./traj_files/dump2
import sys, os, csv
from pyscal3 import Trajectory

if len(sys.argv) < 2:
    print("Usage: python boo.py <trajectory_path>")
    sys.exit(1)

trajectory_path = sys.argv[1]

# make output dir & file immediately
outdir = "./boo_results"
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, "boo_q6_all.csv")

f = open(outfile, "w", newline="")
writer = csv.writer(f)
writer.writerow(["timestep", "atom_id", "q6"])
f.flush()
print(f"Writing to: {outfile}")

traj = Trajectory(trajectory_path)

for i, frame in enumerate(traj):              # stream frames
    # get timestep if available, else fallback to frame index
    ts = None
    for attr in ("timestep", "step", "time"):
        if hasattr(frame, attr):
            ts = int(getattr(frame, attr))
            break
    if ts is None:
        ts = i

    # convert this frame to a System (handle list return)
    sys_obj = frame.to_system()
    sys_at = sys_obj[0] if isinstance(sys_obj, (list, tuple)) else sys_obj

    # neighbors & q6
    sys_at.find.neighbors(method="cutoff", cutoff=1.45)
    q6 = sys_at.calculate.steinhardt_parameter(6, averaged=True)[0]

    # write rows for this timestep
    for aid, val in zip(sys_at.atoms.ids, q6):
        writer.writerow([ts+1, aid, f"{val:.8f}"])
        # ts+1 because I needed here to starts from 1 not zero
        # feel free to just consider ts

    f.flush()
    if (i + 1) % 50 == 0:
        print(f"processed {i+1} frames; last timestep {ts}")

f.close()
print("Done.")
