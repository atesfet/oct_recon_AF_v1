"""Sample system resources of a running reconstruction into a CSV.

  python scripts/monitor_resources.py --pid <PID> --out benchmarks/results/run_resources.csv
Stops when the PID exits. Columns: time, CPU % (total, busiest core, #cores>50%),
RAM used, process RSS (incl. children threads), GPU util, VRAM, GPU power/temp,
disk read/write MB/s for the data drive.
"""
import argparse
import csv
import subprocess
import time

import psutil


def gpu():
    q = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    return [float(v) for v in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--disk", default="sda")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--children", action="store_true",
                    help="sum CPU/RSS/threads over the whole process tree (e.g. MATLAB parpool workers)")
    a = ap.parse_args()
    proc = psutil.Process(a.pid)
    psutil.cpu_percent(percpu=True)
    d0 = psutil.disk_io_counters(perdisk=True)[a.disk]
    t0 = time.time()
    tp = t0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "cpu_total_pct", "cpu_max_core_pct", "cores_over_50pct", "proc_cpu_pct",
                    "proc_threads", "ram_used_GB", "ram_avail_GB", "proc_rss_GB", "gpu_util_pct",
                    "vram_used_MB", "vram_total_MB", "gpu_power_W", "gpu_temp_C", "disk_read_MBps", "disk_write_MBps"])
        proc.cpu_percent()
        seen = {proc.pid: proc}
        while proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
            time.sleep(a.interval)
            try:
                per = psutil.cpu_percent(percpu=True)
                vm = psutil.virtual_memory()
                procs = [proc] + (proc.children(recursive=True) if a.children else [])
                pc = rss = nth = 0
                for q in procs:
                    try:
                        if q.pid not in seen:
                            seen[q.pid] = q
                            q.cpu_percent()
                        q = seen[q.pid]
                        pc += q.cpu_percent()
                        rss += q.memory_info().rss
                        nth += q.num_threads()
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                break
            now = time.time()
            d1 = psutil.disk_io_counters(perdisk=True)[a.disk]
            dt = now - tp
            rd = (d1.read_bytes - d0.read_bytes) / dt / 1e6
            wr = (d1.write_bytes - d0.write_bytes) / dt / 1e6
            d0, tp = d1, now
            g = gpu()
            w.writerow([round(now - t0, 1), round(sum(per) / len(per), 1), max(per), sum(p > 50 for p in per),
                        pc, nth, round(vm.used / 1e9, 2), round(vm.available / 1e9, 2), round(rss / 1e9, 2),
                        *g, round(rd, 1), round(wr, 1)])
            f.flush()


if __name__ == "__main__":
    main()
