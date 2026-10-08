# 构建和资产

主引擎构建使用根目录build.sh，默认CUDA sm_89、Release、系统原生CPU优化。CUDA编译器可通过CUDACXX指定，STRATA_CUDA_ARCH指定架构，STRATA_BUILD_JOBS指定并行编译数。需要cmake>=3.24、C++20编译器和CUDA12.x工具链。这里的二进制仅在本机Linux x86_64和L40S上验证，不保证换机器后保持同样的速度。

Python运行依赖版本列在requirements.txt；本服务器已经准备独立.venv。本机移植已有资产时不需要再下载模型。新主机可把下列目录置于assets/，或用--assets指定包含models/、data/的目录：

```text
assets/
  models/IQ2_XS/
    Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
    Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
    mmproj-Qwen3.8-Flash-Next-BF16.gguf
  data/
    packs/iq2_xs/                 # native_experts.txt, index.txt, dense.bin, tokenizer/
    mtp/native-q5/                # 高精度辅助草稿运行资产
    mtp/head-router-iq2-r128-k8192/ # 辅助词表路由
    mtp/rt/                       # 并发模式原辅助运行资产
    expert-profile.bin
```

attach_assets.py只创建指向这些资产的独立链接，不修改源文件。它不会链接旧项目的代码或.venv。原生模型pack可用tools/iq_pack.py从已下载GGUF重新生成；辅助Q5运行资产和路由必须一同迁移，其尺寸和原始生成方式记录在对应资产目录的元数据中，不能只复制主GGUF便声称复现200/160。

主权重约68.0GB；native-q5辅助资产约1.85GB、路由约16MB，pack另占磁盘空间。导出源码/引擎时不包含这些权重数据、.venv、构建中间件、日志或本机资产链接。

原生图片/视频编码器可从项目内的源码重建：

```bash
cmake -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_DIR="$PWD/third_party/llama.cpp"
cmake --build build-vision --target strata-vision -j12
install -m755 build-vision/bin/strata-vision engine/strata-vision-native
python3 tools/release_manifest.py
```

llama.cpp以源码形式内置，用于ggml量化格式和mtmd编码，不启动llama-server。SOURCE_PROVENANCE记录上游Strata基线以及本地源树快照；dependency目录使用内容快照保存，不能把上层Strata的git提交误认为llama.cpp提交。源码修改/重新编译后，binary SHA及能力/速度需要重新核对。

导出本地发布包：

```bash
.venv/bin/python tools/package_release.py
```

源码包用于构建；Linux sm_89运行包包含四个引擎二进制和源码。两包均排除模型及辅助权重。版本和文件SHA记录在artifacts/SHA256SUMS以及RELEASE.json。本项目尚未绑定远端发布仓库。
