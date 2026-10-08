"""Check single-device isolation, full context, concurrency and inherited experiment cleanup."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import single_gpu as launch

class Policy(unittest.TestCase):
    def setUp(self):
        self.gpu={'index':7,'uuid':'GPU-test-only','pci':'00000000:01:00.0','name':'L40S','memory_mib':46068}

    def config(self,**kw):
        with patch.object(launch,'numa_node',return_value=1):
            return launch.build_config(Path('/portable/assets'),self.gpu,**kw)

    def arg(self,cfg,flag):return cfg['args'][cfg['args'].index(flag)+1]

    def test_one_gpu_full_context_and_same_gpu_vision(self):
        cfg=self.config()
        self.assertEqual(cfg['gpu'],[7]);self.assertEqual(cfg['env']['CUDA_VISIBLE_DEVICES'],'GPU-test-only')
        self.assertEqual(cfg['vision']['cuda_device'],'GPU-test-only')
        self.assertEqual(self.arg(cfg,'--max-context'),'262144')
        self.assertEqual(self.arg(cfg,'--rope-scaling'),'none')
        self.assertEqual(self.arg(cfg,'--kv'),'int8')
        self.assertEqual(cfg['vision']['native_video']['fps'],8)

    def test_residency_and_native_auxiliary_presets(self):
        cfg=self.config()
        self.assertEqual(self.arg(cfg,'--prefill'),'1024')
        self.assertEqual(self.arg(cfg,'--spec'),'8')
        self.assertEqual(self.arg(cfg,'--adapt-swaps'),'0')
        self.assertIn('--no-prefill-borrow',cfg['args'])
        self.assertTrue(self.arg(cfg,'--mtp').endswith('/native-q5'))
        self.assertEqual(cfg['env']['STRATA_VERIFY_DEVICE_ONLY'],'1')

    def test_relocatable_assets_and_binary(self):
        cfg=self.config(engine='built')
        self.assertEqual(Path(cfg['exe']).name,'strata-single-built')
        self.assertTrue(self.arg(cfg,'--native').startswith('/portable/assets/'))
        self.assertTrue(cfg['env']['STRATA_MTP_HEAD_ROUTER'].startswith('/portable/assets/'))
        self.assertNotIn('/home/student_su/qwen/Strata/',str(cfg))

    def test_pool_keeps_one_gpu_and_clears_single_request_flags(self):
        for count in range(2,7):
            with self.subTest(count=count):
                cfg=self.config(concurrency=count,engine='validated')
                self.assertEqual(cfg['gpu'],[7]);self.assertEqual(cfg['parallel_workers'],count)
                self.assertEqual(Path(cfg['exe']).name,'strata-concurrent-validated')
                self.assertTrue(self.arg(cfg,'--mtp').endswith('/rt'))
                self.assertEqual(self.arg(cfg,'--max-context'),'262144')
                self.assertNotIn('STRATA_MTP_HEAD_ROUTER',cfg['env'])
                self.assertNotIn('STRATA_TOPK_PARALLEL',cfg['env'])
                self.assertNotIn('STRATA_HEAD_PARALLEL_GRAPH',cfg['env'])

    def test_invalid_concurrency_and_unvalidated_pool_build_rejected(self):
        for count in (0,7,True,1.0):
            with self.assertRaises(ValueError):self.config(concurrency=count)
        with self.assertRaises(ValueError):self.config(concurrency=2,engine='built')

    def test_gpu_lists_and_missing_devices_rejected(self):
        for value in ('0,1','all','3','GPU-missing'):
            with self.assertRaises(ValueError):launch.select_gpu(value,[self.gpu])
        self.assertIs(launch.select_gpu('7',[self.gpu]),self.gpu)
        self.assertIs(launch.select_gpu('GPU-test-only',[self.gpu]),self.gpu)

    def test_experiment_env_does_not_leak_to_children(self):
        with patch.dict(os.environ,{'STRATA_MTP_LABEL_TRACE':'private','STRATA_HEAD_PARALLEL_GRAPH':'1',
                  'STRATA_API_KEY':'test-key','CUDA_VISIBLE_DEVICES':'0,1'},clear=True):
            launch.clean_environment()
            self.assertEqual(dict(os.environ),{'STRATA_API_KEY':'test-key'})

    def test_no_production_paths_or_service_control_in_single_wrapper(self):
        source=(launch.ROOT/'tools/single_gpu.py').read_text()
        self.assertNotIn('systemctl',source)
        self.assertNotIn('docker',source)
        self.assertNotIn('/home/student_su/qwen/Strata',source)

if __name__=='__main__':unittest.main()
