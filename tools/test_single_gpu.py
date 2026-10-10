"""Check single-device isolation, full context, concurrency and inherited experiment cleanup."""
import os
import json
import tempfile
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
        self.assertTrue(self.arg(cfg,'--mtp').endswith('/native-q5-full'))
        self.assertEqual(cfg['env']['STRATA_VERIFY_DEVICE_ONLY'],'1')

    def test_validated_fallback_retains_its_original_auxiliary_and_flags(self):
        cfg=self.config(engine='validated')
        self.assertTrue(self.arg(cfg,'--mtp').endswith('/native-q5'))
        self.assertNotIn('STRATA_CUDA_EXPERT_FUSED',cfg['env'])
        self.assertNotIn('STRATA_MMVQ_ROWS',cfg['env'])
        self.assertEqual(cfg['draft_vocab'],'cjk')

    def test_larger_weights_keep_prefill_loans_and_native_vision(self):
        for weights in ('swift','iq3_s'):
            cfg=self.config(weights=weights)
            self.assertNotIn('--no-prefill-borrow',cfg['args'])
            self.assertNotIn('--adapt-swaps',cfg['args'])
            if weights=='swift':self.assertEqual(cfg['env']['STRATA_MEDIA_FIXED_PROFILE'],'1')
            self.assertNotIn('STRATA_VERIFY_DEVICE_ONLY',cfg['env'])
            self.assertEqual(self.arg(cfg,'--max-context'),'262144')
            self.assertEqual(self.arg(cfg,'--pool-workers'),'32')
            self.assertEqual(cfg['vision']['model'],self.arg(cfg,'--native'))
            self.assertEqual(cfg['vision']['cuda_device'],'GPU-test-only')
            self.assertIn('head-router-'+weights,cfg['env']['STRATA_MTP_HEAD_ROUTER'])
            if weights=='swift':self.assertEqual(self.arg(cfg,'--ple-gguf'),self.arg(cfg,'--native'))
            else:self.assertTrue(self.arg(cfg,'--ple-gguf').endswith('-00002-of-00002.gguf'))

    def test_larger_weights_do_not_silently_use_static_ipc_pool(self):
        for weights in ('swift','iq3_s'):
            with self.assertRaises(ValueError):self.config(weights=weights,engine='validated',concurrency=2)
            with self.assertRaises(ValueError):self.config(weights=weights,engine='validated')

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

    def test_native_default_requires_qualification_and_respects_mode_selection(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(launch,'ROOT',Path(folder)):
            path=Path(folder)/'experimental/native-batch/manifest.json'
            self.assertFalse(launch.native_batch_is_default('swift',2))
            path.parent.mkdir(parents=True)
            q={'passed':False,'weights':['swift','iq2_xs'],'preferred_counts':{'iq2_xs':[2,3,4]}}
            path.write_text(json.dumps({'qualification':q}))
            self.assertFalse(launch.native_batch_is_default('swift',2))
            q['passed']=True;path.write_text(json.dumps({'qualification':q}))
            self.assertTrue(launch.native_batch_is_default('swift',6))
            self.assertTrue(launch.native_batch_is_default('iq2_xs',2))
            self.assertFalse(launch.native_batch_is_default('iq2_xs',6))
            self.assertFalse(launch.native_batch_is_default('swift',1))
            self.assertFalse(launch.native_batch_is_default('unqualified-weight',2))

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
