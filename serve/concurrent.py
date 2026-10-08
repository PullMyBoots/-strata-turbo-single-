"""Bounded independent sessions sharing immutable CUDA weights through IPC and MPS.

The weight owner outlives its readers. Each engine has separate pipes, graph buffers,
KV, recurrence, RNG and prefix cache. GPU vision encoding takes an exclusive turn;
prefill shares a leased workspace inside the engine, while decode can overlap.
"""
from __future__ import annotations

import atexit
import contextlib
import copy
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time

from serve import server


class MediaGate:
    """Writer preference prevents vision starvation during a stream of text requests."""
    def __init__(self):
        self.cv = threading.Condition()
        self.readers = self.waiting_writers = 0
        self.writer = False

    @contextlib.contextmanager
    def read(self):
        with self.cv:
            self.cv.wait_for(lambda: not self.writer and not self.waiting_writers)
            self.readers += 1
        try:
            yield
        finally:
            with self.cv:
                self.readers -= 1
                self.cv.notify_all()

    @contextlib.contextmanager
    def write(self):
        with self.cv:
            self.waiting_writers += 1
            try:
                self.cv.wait_for(lambda: not self.writer and not self.readers)
                self.writer = True
            finally:
                self.waiting_writers -= 1
        try:
            yield
        finally:
            with self.cv:
                self.writer = False
                self.cv.notify_all()


class SingleService(server.Service):
    """Limit the entire inference request, including vision preparation, to one."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_cv = threading.Condition()
        self._request_local = threading.local()
        self._request_active = self._request_waiting = 0

    @contextlib.contextmanager
    def request_slot(self):
        if getattr(self._request_local, "held", False):
            yield
            return
        with self._request_cv:
            self._request_waiting += 1
            try:
                self._request_cv.wait_for(lambda: self._request_active == 0)
                self._request_active = 1
            finally:
                self._request_waiting -= 1
        self._request_local.held = True
        try:
            yield
        finally:
            self._request_local.held = False
            with self._request_cv:
                self._request_active = 0
                self._request_cv.notify_all()

    def _request_counts(self):
        with self._request_cv:
            return self._request_active, self._request_waiting

    def _snapshot(self):
        active, waiting = self._request_counts()
        with self.status_lock:
            state = dict(self.status, id=0, n_ctx=self.engine.max_context,
                         is_processing=bool(active), loaded=self.loaded())
        return active, waiting, [state]

    def v1_status(self):
        result = super().v1_status()
        active, waiting = self._request_counts()
        result["concurrency"] = dict(serving=1, requested=1, active=active, queued=waiting,
                                     mode="single_session", prefill="single_workspace")
        result["activity"]["in_flight"] = active + waiting
        return result

    def metrics(self, all_requests=False):
        result = super().metrics(all_requests=all_requests)
        active, waiting = self._request_counts()
        result["live"].update(active=active, queued=waiting)
        return result


class PoolRuntime:
    """A private MPS daemon and IPC directory, never the machine's default MPS pipe."""
    def __init__(self, cfg, env):
        n = cfg.get("parallel_workers", 1)
        if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 6:
            raise ValueError("parallel_workers must be an integer from 2 to 6")
        if os.name != "posix" or cfg.get("backend") == "hip":
            raise ValueError("parallel workers require Linux NVIDIA CUDA IPC and MPS")
        if cfg.get("lazy_load") or cfg.get("idle_unload_s"):
            raise ValueError("parallel workers require a resident model (no lazy/idle unload)")
        if not shutil.which("nvidia-cuda-mps-control"):
            raise ValueError("nvidia-cuda-mps-control is missing")
        pct = cfg.get("parallel_worker_active_thread_pct", 50)
        if isinstance(pct, bool) or not isinstance(pct, int) or not 1 <= pct <= 100:
            raise ValueError("parallel_worker_active_thread_pct must be an integer from 1 to 100")
        self.reader_pct = pct
        connections = cfg.get("parallel_device_max_connections", 2)
        if isinstance(connections, bool) or not isinstance(connections, int) or not 1 <= connections <= 32:
            raise ValueError("parallel_device_max_connections must be an integer from 1 to 32")
        self.count, self.cfg = n, copy.deepcopy(cfg)
        # Fully GPU-resident experts have no CPU misses. Map their original bytes for validation
        # and prefill metadata instead of pinning a second 33-40 GiB CPU copy per worker.
        if "--mmap-experts" not in self.cfg["args"]:
            self.cfg["args"].append("--mmap-experts")
        # Limit host helpers: the static resident GPU path does not need hundreds of CPU workers.
        args = self.cfg["args"]
        if "--pool-workers" not in args:
            args.extend(["--pool-workers", "8"])
        self.gate = MediaGate()
        self.engines, self.vision = [], None
        self.closed = False
        self.lock = threading.RLock()
        self.directory = Path(tempfile.mkdtemp(prefix="strata-pool-", dir="/tmp"))
        for name in ("weights", "pipe", "logs"):
            (self.directory / name).mkdir(mode=0o700)
        self.env = dict(env, CUDA_MPS_PIPE_DIRECTORY=str(self.directory / "pipe"),
                        CUDA_MPS_LOG_DIRECTORY=str(self.directory / "logs"),
                        STRATA_SHARED_WEIGHTS_DIR=str(self.directory / "weights"),
                        STRATA_SHARED_WEIGHTS_MODE="owner", CUDA_DEVICE_MAX_CONNECTIONS=str(connections),
                        CUDA_MODULE_LOADING="LAZY", CUBLAS_WORKSPACE_CONFIG=":16:8",
                        STRATA_STAGER_THREADS="2", CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="100")
        try:
            subprocess.run(["nvidia-cuda-mps-control", "-d"], env=self.env,
                           stdin=subprocess.DEVNULL, check=True, timeout=10)
        except BaseException:
            shutil.rmtree(self.directory)
            raise
        atexit.register(self.close)

    def attach(self, engine):
        self.engines.append(engine)
        if self.env.get("STRATA_EPHEMERAL_PREFILL") == "1":
            # Graph initialization touches managed pages. Return idle pages to
            # RAM before the next worker allocates its transient prompt scratch.
            engine.park()
        engine.session_lock = threading.Lock()
        original = engine.generate
        def generate(*args, **kwargs):
            with self.gate.read():
                with engine.session_lock:
                    try:
                        yield from original(*args, **kwargs)
                    finally:
                        if self.env.get("STRATA_EPHEMERAL_PREFILL") == "1" and engine.alive():
                            # Keep completed long contexts in RAM while their
                            # session is idle; bytes and prefix caches survive.
                            engine.park()
        engine.generate = generate
        # A reader cannot restart independently after the owner frees its IPC allocations.
        def no_restart():
            raise server.EngineDied("parallel engine pool requires a whole-service restart")
        engine.restart = no_restart

    def healthy(self):
        return not self.closed and len(self.engines) == self.count and all(e.alive() for e in self.engines)

    def park_for_vision(self):
        # The exclusive media gate is held by the caller. Parking changes residency, not KV precision or content.
        for engine in self.engines:
            engine.park()

    def park_idle(self, exclude):
        # An idle session's managed pages must not crowd a 128K/256K active context out of VRAM.
        # Never wait for an active engine or interrupt its decode. The media gate also excludes vision encoding.
        with self.gate.read():
            for engine in self.engines:
                if engine is exclude or not engine.session_lock.acquire(blocking=False):
                    continue
                try:
                    engine.park()
                finally:
                    engine.session_lock.release()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            # Send STOP to every reader before waiting; keep the owner alive until all readers exit.
            for e in reversed(self.engines):
                if e.alive():
                    try:
                        e.proc.stdin.write("STOP\n")
                        e.proc.stdin.flush()
                    except (OSError, ValueError):
                        pass
            failures = []
            for e in reversed(self.engines):
                try:
                    e.close()
                except Exception as exc:
                    failures.append(str(exc))
                    # Do not free the owner's allocation while a reader is still running.
                    if e is not self.engines[0] and e.alive():
                        print("[strata] pool shutdown: reader is still alive; retaining owner", flush=True)
                        return
            if self.vision:
                self.vision.close()
            try:
                subprocess.run(["nvidia-cuda-mps-control"], env=self.env, input="quit\n",
                               text=True, capture_output=True, timeout=30, check=True)
            except (OSError, subprocess.SubprocessError) as exc:
                failures.append(str(exc))
            if failures:
                print("[strata] pool shutdown: " + "; ".join(failures), flush=True)
            else:
                shutil.rmtree(self.directory, ignore_errors=True)


class WorkerService(server.Service):
    def ensure_loaded(self):
        if not self.pool_runtime.healthy() or self._vision_down():
            raise server.EngineDied("parallel engine pool is unavailable; restart the service as a unit")

    def prepare(self, messages, tools, kwargs, max_new=None):
        # Includes native video PRE, which talks directly to the vision pipe.
        if server.images_of(messages):
            with self.pool_runtime.gate.write():
                self.pool_runtime.park_for_vision()
                return super().prepare(messages, tools, kwargs, max_new)
        prepared = super().prepare(messages, tools, kwargs, max_new)
        if len(prepared[0]) >= 65536 or self.pool_runtime.env.get("STRATA_EPHEMERAL_PREFILL") == "1":
            self.pool_runtime.park_idle(self.engine)
        return prepared

    def _say_died(self, exc):
        print(f"[strata] parallel worker failed: {exc}; restart the whole service", flush=True)


class ConcurrentService:
    """Select a worker once per HTTP request, including template, tool loops and stream drain."""
    def __init__(self, engine, tokenizer, template, *, runtime, **kwargs):
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_local", threading.local())
        object.__setattr__(self, "_slots", queue.PriorityQueue(maxsize=runtime.count))
        object.__setattr__(self, "_counts", threading.Lock())
        object.__setattr__(self, "_active", 0)
        object.__setattr__(self, "_waiting", 0)
        object.__setattr__(self, "_maintenance", threading.Lock())
        object.__setattr__(self, "_workers", [])
        if engine not in runtime.engines:
            runtime.attach(engine)
        for i in range(runtime.count):
            if i:
                log = runtime.cfg.get("log")
                if log:
                    p = Path(log)
                    log = str(p.with_name(p.stem + f".worker-{i}" + p.suffix))
                print(f"[strata] loading shared-weight worker {i + 1}/{runtime.count}", flush=True)
                engine = server.StrataEngine(runtime.cfg["exe"], server.engine_args(runtime.cfg),
                    cwd=runtime.cfg.get("cwd"), log=log,
                    env=dict(runtime.env, STRATA_SHARED_WEIGHTS_MODE="reader",
                             CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(getattr(runtime,"reader_pct",50))))
                engine.silence_s = server.engine_silence_s(runtime.cfg)
                runtime.attach(engine)
            worker = WorkerService(engine, tokenizer, template, **kwargs)
            worker.pool_runtime = runtime
            self._workers.append(worker)
            worker.pool_index = i
            self._slots.put((i, worker))
        print(f"[strata] {runtime.count} independent sessions ready; native context {engine.max_context}", flush=True)
        if isinstance(runtime, PoolRuntime):
            threading.Thread(target=self._watch_health, daemon=True, name="strata-pool-health").start()

    def _watch_health(self):
        while not self._runtime.closed:
            time.sleep(1)
            if not self._runtime.closed and not self._runtime.healthy():
                print("[strata] a shared-weight worker stopped; closing the pool before supervisor restart", flush=True)
                self._runtime.close()
                os._exit(1)

    def __getattr__(self, name):
        worker = getattr(self._local, "worker", None) or self._workers[0]
        return getattr(worker, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            for worker in self._workers:
                setattr(worker, name, value)

    @contextlib.contextmanager
    def request_slot(self):
        if getattr(self._local, "worker", None):
            yield
            return
        with self._counts:
            self._waiting += 1
        worker = None
        try:
            while worker is None:
                if not self._runtime.healthy():
                    raise server.EngineDied("parallel engine pool is unavailable; restart the service")
                try:
                    _, worker = self._slots.get(timeout=1)
                except queue.Empty:
                    continue
            with self._counts:
                self._waiting -= 1
                self._active += 1
            self._local.worker = worker
            try:
                yield
            finally:
                self._local.worker = None
                with self._counts:
                    self._active -= 1
                self._slots.put((worker.pool_index, worker))
        finally:
            if worker is None:
                with self._counts:
                    self._waiting -= 1

    def loaded(self):
        return self._runtime.healthy()

    def load(self):
        if not self.loaded():
            raise server.EngineDied("parallel engine pool is unavailable; restart the service")

    def ensure_loaded(self):
        self.load()

    def unload(self, idle_for=None):
        # MPS and IPC need a coordinated close; never allow a single worker to free shared weights.
        with self._counts:
            if self._active or self._waiting:
                return "busy"
        return "unsupported"

    def start_idle_unload(self):
        if self.idle_unload_s:
            raise ValueError("idle unload is not available with shared-weight workers")

    def set_aliases(self, aliases):
        for worker in self._workers:
            worker.set_aliases(aliases)

    def set_shared(self, defaults):
        clean = server.clean_shared_defaults(defaults)
        for worker in self._workers:
            worker.shared = dict(clean)
        return clean

    def _snapshot(self):
        with self._counts:
            active, waiting = self._active, self._waiting
        states = []
        for i, worker in enumerate(self._workers):
            with worker.status_lock:
                states.append(dict(worker.status, id=i, n_ctx=worker.engine.max_context,
                                   is_processing=bool(worker.status.get("busy")), loaded=worker.loaded()))
        return active, waiting, states

    @property
    def status(self):
        active, waiting, states = self._snapshot()
        return dict(busy=bool(active), queued=waiting, active=active, slots=states)

    def v1_status(self):
        result = self._workers[0].v1_status()
        active, waiting, states = self._snapshot()
        result["loaded"] = self.loaded()
        result["concurrency"] = dict(serving=self._runtime.count, requested=self._runtime.count,
            active=active, queued=waiting, mode="shared_weights_independent_sessions", prefill="serialized_workspace",
            primary_active_thread_pct=100, worker_active_thread_pct=getattr(self._runtime,"reader_pct",50))
        result["activity"]["in_flight"] = active + waiting
        result["activity"]["requests"] = sum(w.totals["requests"] for w in self._workers) + active
        result["slots"] = states
        return result

    def metrics(self, all_requests=False):
        parts = [worker.metrics(all_requests=all_requests) for worker in self._workers]
        result = parts[0]
        active, waiting, states = self._snapshot()
        result["engine"]["parallel_workers"] = self._runtime.count
        result["live"].update(active=active, queued=waiting, state="generating" if active else "idle", slots=states)
        result["requests"] = sorted([r for part in parts for r in part["requests"]],
                                     key=lambda r: r.get("at", r.get("started", 0)), reverse=True)
        if not all_requests:
            result["requests"] = result["requests"][:12]
        result["requests_kept"] = sum(part["requests_kept"] for part in parts)
        result["totals"] = {key: min(p["totals"][key] for p in parts) if key == "since"
                            else sum(p["totals"][key] for p in parts) for key in result["totals"]}
        return result

    def request_records(self, request_id=None):
        if request_id:
            return next((r for worker in self._workers if (r := worker.request_records(request_id))), None)
        return sorted([r for worker in self._workers for r in worker.request_records()],
                      key=lambda r: r["started_at"], reverse=True)[:100]

    def close(self):
        self._runtime.close()


def handler_with_slots(base, svc):
    class ConcurrentHandler(base):
        _strata_slots = True

        def _json(self, code, obj):
            if code == 200 and self.path.split("?")[0].rstrip("/") == "/props":
                obj = dict(obj, total_slots=svc._runtime.count if hasattr(svc, "_runtime") else 1)
            return super()._json(code, obj)

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            if path not in ("/v1/chat/completions", "/v1/messages", "/chat/completions", "/messages"):
                return super().do_POST()
            # Authenticate before waiting for a costly inference slot.
            if not self._authorized() or self._foreign_page():
                return
            try:
                with svc.request_slot():
                    return super().do_POST()
            except server.EngineDied as exc:
                self._json(503, {"error": {"type": "server_error", "message": str(exc)}})

        def do_GET(self):
            if self.path.split("?")[0].rstrip("/") in ("/slots", "/v1/slots"):
                if self._authorized() and not self._foreign_page():
                    self._json(200, svc._snapshot()[2])
                return
            return super().do_GET()
    return ConcurrentHandler
