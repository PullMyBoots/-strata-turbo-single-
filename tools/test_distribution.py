"""Release-policy regression tests; no GPU, downloads or model loading."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import distribution as dist
from qualify_distribution import decide


class ResourcePolicy(unittest.TestCase):
    def hardware(self, vram=48, ram=128, cores=32):
        return {'gpu': {'free_vram_mib': vram * 1024}, 'ram_available_bytes': ram * 10**9,
                'local_physical_cores': cores, 'wsl': False}

    def test_resources_keep_capabilities_at_different_sizes(self):
        for vram in (12, 16, 24, 32, 48, 80):
            for weights in dist.MODELS:
                for count in (1, 2, 4, 6):
                    plan = dist.resource_plan(self.hardware(vram), weights, count)
                    self.assertEqual(plan['context'], 262144)
                    self.assertEqual(plan['kv'], 'int8')
                    self.assertTrue(plan['full_experts'])
                    self.assertEqual(plan['input_modalities'], ['text', 'image', 'video'])
                    self.assertFalse(plan['audio_input'])
                    self.assertEqual(plan['concurrency'], count)
                    self.assertEqual(plan['vision_gpu'], vram >= 20)

    def test_cpu_count_and_cgroup_memory_are_respected(self):
        self.assertEqual(dist.resource_plan(self.hardware(cores=4), 'swift')['workers'], 4)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'memory.max').write_text(str(32 * 10**9))
            (root / 'memory.current').write_text(str(8 * 10**9))
            total, free = dist.effective_memory(128 * 10**9, 100 * 10**9, root)
            self.assertEqual((total, free), (32 * 10**9, 24 * 10**9))

    def test_low_ram_does_not_shorten_context_or_quantize_kv(self):
        hw = self.hardware(vram=24, ram=32)
        plan = dist.resource_plan(hw, 'iq3_s', 6)
        self.assertTrue(plan['low_ram'])
        self.assertFalse(plan['kv_streaming'])
        self.assertEqual(plan['kv_resident'], 0)
        self.assertEqual(plan['context'], 262144)
        hw['wsl'] = True
        self.assertFalse(dist.resource_plan(hw, 'iq2_xs')['kv_streaming'])

    def test_invalid_resources_are_rejected(self):
        for count in (0, 7, True, 1.0):
            with self.assertRaises(ValueError):
                dist.resource_plan(self.hardware(), 'iq2_xs', count)
        with self.assertRaises(ValueError):
            dist.resource_plan(self.hardware(ram=8), 'iq2_xs')


class SpeedGate(unittest.TestCase):
    def rates(self, rate):
        return {case: [rate * .99, rate, rate * 1.01] for case in ('english', 'chinese', 'code')}

    def test_fast_valid_candidate_is_promoted(self):
        result = decide(self.rates(100), self.rates(110), True)
        self.assertEqual(result['selection'], 'turbo')
        self.assertAlmostEqual(result['gain_percent'], 10)

    def test_noise_and_slow_candidate_use_upstream(self):
        for rate in (95, 100, 102):
            self.assertEqual(decide(self.rates(100), self.rates(rate), True)['selection'], 'official')

    def test_failed_quality_and_incomplete_tasks_cannot_be_certified(self):
        self.assertEqual(decide(self.rates(100), self.rates(150), False)['state'], 'failed')
        self.assertEqual(decide(self.rates(100), {'english': [150]}, True)['state'], 'failed')

    def test_large_code_gain_cannot_hide_material_regression(self):
        turbo = self.rates(120)
        turbo['english'] = [90, 90, 90]
        self.assertEqual(decide(self.rates(100), turbo, True)['selection'], 'official')

    def test_output_gain_cannot_hide_slower_prefill(self):
        result = decide(self.rates(100), self.rates(120), True,
                        official_prefill={'recall': [4000, 4100]}, turbo_prefill={'recall': [2000, 2100]})
        self.assertEqual(result['selection'], 'official')
        self.assertLess(result['prefill_ratios']['recall'], .97)
        result = decide(self.rates(100), self.rates(120), True,
                        official_prefill={'recall': [4000]}, turbo_prefill={'recall': [4200]})
        self.assertEqual(result['selection'], 'turbo')

    def test_native_prefill_regression_and_invalid_results_are_rejected(self):
        for turbo in ({'recall': [4200]}, {'recall': [4200], 'native_long': [2000]},
                      {'recall': [float('nan')], 'native_long': [4000]}):
            result = decide(self.rates(100), self.rates(120), True,
                            official_prefill={'recall': [4000], 'native_long': [4000]}, turbo_prefill=turbo)
            self.assertEqual(result['selection'], 'official')

    def test_binary_and_resource_changes_invalidate_identity(self):
        base = {'binary': 'a', 'gpu': 'one', 'concurrency': 1, 'context': 262144}
        for field, value in (('binary', 'b'), ('gpu', 'two'), ('concurrency', 6)):
            changed = dict(base, **{field: value})
            self.assertNotEqual(dist.key(base), dist.key(changed))


class ConfigPolicy(unittest.TestCase):
    def test_frontend_and_template_changes_invalidate_certificate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in ('serve', 'experimental/native-batch/serve'):
                (root / relative).mkdir(parents=True)
                (root / relative / 'local_api.py').write_text('entrypoint')
                (root / relative / 'frontend.py').write_text('old frontend')
                (root / relative / 'chat_template.jinja').write_text('old template')
            original = dist.key(dist.api_source_hashes(root))
            for relative in ('serve', 'experimental/native-batch/serve'):
                for name in ('frontend.py', 'chat_template.jinja'):
                    path = root / relative / name
                    old = path.read_text()
                    path.write_text('changed input rendering')
                    self.assertNotEqual(original, dist.key(dist.api_source_hashes(root)))
                    path.write_text(old)
            (root / 'serve' / 'test_example.py').write_text('test changes')
            self.assertEqual(original, dist.key(dist.api_source_hashes(root)))

    def test_actual_prefill_and_cache_restore_options_invalidate_qualification(self):
        cfg = {'args': ['--pack', '/assets/model', '--prefill', 'auto', '--spec', '8'],
               'env': {'CUDA_VISIBLE_DEVICES': 'GPU-one', 'STRATA_VERIFY_PREFILL_RESTORE': '1'},
               'vision': {'gpu': True, 'native_video': {'fps': 8}}}
        original = dist.key(dist.performance_options(cfg))
        changed = copy.deepcopy(cfg)
        changed['args'] = dist.set_arg(changed['args'], '--prefill', 'auto:32768')
        self.assertNotEqual(original, dist.key(dist.performance_options(changed)))
        changed = copy.deepcopy(cfg)
        changed['env']['STRATA_VERIFY_PREFILL_RESTORE'] = '0'
        self.assertNotEqual(original, dist.key(dist.performance_options(changed)))
        changed = copy.deepcopy(cfg)
        changed['args'][1] = '/relocated/assets/model'
        self.assertEqual(original, dist.key(dist.performance_options(changed)))

    def test_consumer_configs_preserve_one_gpu_and_use_portable_engines(self):
        gpu = {'index': 3, 'uuid': 'GPU-unit-test', 'pci': '00000000:01:00.0', 'free_vram_mib': 48*1024}
        hw = {'gpu': gpu, 'ram_available_bytes': 128*10**9, 'local_physical_cores': 32, 'wsl': False}
        manifest = {'binaries': {name: {'path': 'engine/distribution/' + name} for name in ('turbo', 'official_media', 'vision')}}
        with tempfile.TemporaryDirectory() as temp, patch.object(dist, 'BUILD', Path(temp) / 'build.json'), \
                patch.object(dist.single_gpu, 'numa_node', return_value=1):
            dist.save(dist.BUILD, manifest)
            for weights in dist.MODELS:
                for count in (1, 6):
                    plan = dist.resource_plan(hw, weights, count)
                    for arm in ('turbo', 'official'):
                        cfg = dist.build_config(Path(temp), hw, weights, plan, arm)
                        self.assertEqual(cfg['gpu'], [3])
                        self.assertEqual(cfg['env']['CUDA_VISIBLE_DEVICES'], 'GPU-unit-test')
                        self.assertEqual(cfg['vision']['cuda_device'], 'GPU-unit-test')
                        self.assertIn('--vision', cfg['args'])
                        self.assertEqual(cfg['args'][cfg['args'].index('--max-context') + 1], '262144')
                        self.assertEqual(cfg['args'][cfg['args'].index('--kv') + 1], 'int8')
                        self.assertEqual(cfg['vision']['native_video']['fps'], 8)
                        self.assertEqual(cfg['vision'].get('min_tokens', 0), 128 if weights == 'swift' else 0)
                        self.assertEqual(cfg['deployment']['max_concurrent_requests'], count)
                        if count > 1:
                            self.assertEqual(cfg['parallel_workers'], 1)
                            self.assertEqual(cfg['parallel'], count)
                        if weights == 'iq2_xs' and count == 1 and arm == 'turbo':
                            self.assertNotIn('--no-prefill-borrow', cfg['args'])
                            self.assertEqual(cfg['args'][cfg['args'].index('--prefill') + 1], 'auto:16384')
                            self.assertEqual(cfg['env']['STRATA_VERIFY_PREFILL_RESTORE'], '1')
                            self.assertEqual(cfg['env']['STRATA_VERIFY_DEVICE_ONLY'], '1')
                            self.assertEqual(cfg['args'][cfg['args'].index('--adapt-swaps') + 1], '0')


if __name__ == '__main__':
    unittest.main()
