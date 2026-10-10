"""Keep CPU/memory-sharing README inference suites sequential on this host."""
import fcntl
from pathlib import Path


def acquire():
    path = Path(__file__).resolve().parents[1] / 'runtime/showcase-benchmark.lock'
    path.parent.mkdir(exist_ok=True)
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Another README inference suite is active. Run throughput and agent suites sequentially, even on different GPUs.') from None
    return handle
