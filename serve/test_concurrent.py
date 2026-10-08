"""Exercise bounded session ownership, cancellation cleanup and the media/decode gate."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import threading
import time
import types
import unittest
from unittest.mock import patch

from serve import concurrent, server


class FakeEngine:
    max_context = 262144
    info = {}
    alive = lambda self: True
    generate = lambda self, *a, **kw: iter(())


class Sessions(unittest.TestCase):
    def test_single_capacity_queues_whole_request_and_releases_after_failure(self):
        svc=concurrent.SingleService(FakeEngine(),server.ByteTokenizer(),None)
        entered=threading.Event();release=threading.Event();extra_entered=threading.Event()
        def first():
            with svc.request_slot():
                with svc.request_slot():
                    self.assertEqual(svc._request_counts(),(1,0))
                entered.set();release.wait(timeout=3)
                raise RuntimeError('client disconnected')
        def extra():
            with svc.request_slot():extra_entered.set()
        with ThreadPoolExecutor(max_workers=2) as executor:
            a=executor.submit(first);self.assertTrue(entered.wait(3));b=executor.submit(extra)
            self.assertFalse(extra_entered.wait(.05))
            self.assertEqual(svc._request_counts(),(1,1))
            active,queued,slots=svc._snapshot()
            self.assertEqual((active,queued),(1,1))
            self.assertEqual(slots[0]['n_ctx'],262144)
            self.assertTrue(slots[0]['is_processing'])
            release.set()
            with self.assertRaises(RuntimeError):a.result(timeout=3)
            b.result(timeout=3)
        self.assertEqual(svc._request_counts(),(0,0))
        self.assertFalse(svc._snapshot()[2][0]['is_processing'])

    def test_ephemeral_session_parks_after_cancel_cleanup_before_unlock(self):
        runtime=concurrent.PoolRuntime.__new__(concurrent.PoolRuntime)
        runtime.engines=[];runtime.env={'STRATA_EPHEMERAL_PREFILL':'1'};runtime.gate=concurrent.MediaGate()
        events=[]
        def generate():
            try:
                yield 7
                yield 8
            finally:
                events.append('drained')
        engine=types.SimpleNamespace(alive=lambda:True,park=lambda:events.append('parked'),generate=generate)
        runtime.attach(engine)
        events.clear()
        stream=engine.generate();self.assertEqual(next(stream),7)
        self.assertTrue(engine.session_lock.locked())
        stream.close()
        self.assertEqual(events,['drained','parked'])
        self.assertFalse(engine.session_lock.locked())
        self.assertEqual(runtime.gate.readers,0)

    def test_cache_control_reaches_engine_without_changing_default(self):
        self.assertNotIn('cache_prompt', server.StrataEngine.sampling_keys({}))
        self.assertIn('cache_prompt=0', server.StrataEngine.sampling_keys({'cache_prompt':False}))
        self.assertIn('cache_prompt=1', server.StrataEngine.sampling_keys({'cache_prompt':True}))

    def make_pool(self,count=6):
        runtime = types.SimpleNamespace(count=count, engines=[], cfg={"exe":"unused","args":[]},
                                        env={}, gate=concurrent.MediaGate(), healthy=lambda:True)
        runtime.attach = runtime.engines.append
        with patch.object(server, "StrataEngine", side_effect=lambda *a, **kw:FakeEngine()):
            return concurrent.ConcurrentService(FakeEngine(), server.ByteTokenizer(), None, runtime=runtime)

    def test_six_distinct_sessions_and_seventh_waits(self):
        pool=self.make_pool();entered=threading.Barrier(7);release=threading.Event();seventh=threading.Event()
        def occupy():
            with pool.request_slot():
                worker=pool.engine
                entered.wait(timeout=3);release.wait(timeout=3)
                self.assertIs(worker,pool.engine)
                return id(worker)
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures=[executor.submit(occupy) for _ in range(6)]
            entered.wait(timeout=3)
            def extra():
                with pool.request_slot():seventh.set()
            seventh_job=executor.submit(extra)
            self.assertFalse(seventh.wait(.05));self.assertEqual(pool.status['active'],6)
            release.set();owners=[f.result(timeout=3) for f in futures];seventh_job.result(timeout=3)
        self.assertEqual(len(set(owners)),6);self.assertEqual(pool.status['active'],0)

    def test_smaller_limits_queue_extra_request_and_recover_after_failure(self):
        for count in range(2,6):
            with self.subTest(count=count):
                pool=self.make_pool(count);ready=threading.Barrier(count+1)
                release=threading.Event();extra_entered=threading.Event()
                def occupy():
                    with pool.request_slot():
                        ready.wait(timeout=3);release.wait(timeout=3)
                        raise RuntimeError('cancelled request')
                def extra():
                    with pool.request_slot():extra_entered.set()
                with ThreadPoolExecutor(max_workers=count+1) as executor:
                    jobs=[executor.submit(occupy) for _ in range(count)]
                    ready.wait(timeout=3);job=executor.submit(extra)
                    self.assertFalse(extra_entered.wait(.05))
                    self.assertEqual(pool.status['active'],count)
                    release.set()
                    for f in jobs:
                        with self.assertRaises(RuntimeError):f.result(timeout=3)
                    job.result(timeout=3)
                self.assertEqual(pool._slots.qsize(),count)
                self.assertEqual(pool.status['active'],0)

    def test_failed_and_nested_request_return_one_slot(self):
        pool=self.make_pool()
        with self.assertRaises(RuntimeError):
            with pool.request_slot():
                worker=pool.engine
                with pool.request_slot():self.assertIs(pool.engine,worker)
                raise RuntimeError('client disconnected')
        self.assertEqual(pool._slots.qsize(),6);self.assertEqual(pool.status['active'],0)

    def test_media_waits_for_decode_and_blocks_new_decode(self):
        gate=concurrent.MediaGate();events=[];first=threading.Event();release=threading.Event();writer_waiting=threading.Event()
        def decode():
            with gate.read():first.set();release.wait(timeout=3)
        def media():
            writer_waiting.set()
            with gate.write():events.append('media');time.sleep(.02)
        def next_decode():
            with gate.read():events.append('decode')
        with ThreadPoolExecutor(max_workers=3) as executor:
            a=executor.submit(decode);self.assertTrue(first.wait(3))
            b=executor.submit(media);self.assertTrue(writer_waiting.wait(3))
            until=time.monotonic()+3
            while not gate.waiting_writers and time.monotonic()<until:time.sleep(.001)
            c=executor.submit(next_decode);release.set()
            for f in (a,b,c):f.result(timeout=3)
        self.assertEqual(events,['media','decode'])

    def test_long_request_parks_idle_sessions_and_skips_busy_one(self):
        runtime=concurrent.PoolRuntime.__new__(concurrent.PoolRuntime)
        runtime.gate=concurrent.MediaGate();parked=[]
        runtime.engines=[types.SimpleNamespace(session_lock=threading.Lock(),park=lambda i=i:parked.append(i))
                         for i in range(6)]
        runtime.engines[1].session_lock.acquire()
        runtime.park_idle(runtime.engines[0])
        self.assertEqual(parked,[2,3,4,5])
        self.assertTrue(runtime.engines[1].session_lock.locked())
        runtime.engines[1].session_lock.release()
        self.assertTrue(all(not e.session_lock.locked() for e in runtime.engines))


if __name__ == '__main__':unittest.main()
