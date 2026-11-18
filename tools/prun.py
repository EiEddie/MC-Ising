from concurrent.futures import ProcessPoolExecutor
from subprocess import run, TimeoutExpired
from itertools import product
from pathlib import Path
import sys, os


CPU = 40
TIMEOUT = 10 #min


WORKDIR = Path(__file__).parent.parent
EXE = "mc_ising" if sys.platform != "win32" else "mc_ising.exe"

def main(x):
    L, K = x
    K = round(K, 6)
    try:
        # run([WORKDIR/"tools"/"run", str(L), str(K)], check=True, timeout=60*TIMEOUT)
        run([WORKDIR/"build/release"/EXE, f"L={L},K={K}", "eps_m=0.2,eps_e=0.2"],
            check=True, timeout=60*TIMEOUT,
            stdout=open(WORKDIR/"data"/f"L{L}-K{K}.csv", "w", buffering=1))
        print(f"[info] calc completed: L={L},K={K}", file=sys.stdout)
    except TimeoutExpired as e:
        print(f"[error] calc timeout {e.timeout:.1f}s: L={L},K={K}", file=sys.stderr)


if __name__ == "__main__":
    L = [8,16,32,64,128,256]
    K = [0.1,0.2,0.3,0.4,0.5]

    with ProcessPoolExecutor(max_workers=min(CPU, os.cpu_count()-1)) as executor:
        list(executor.map(main, product(L, K)))
