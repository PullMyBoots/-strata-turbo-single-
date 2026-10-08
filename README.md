# Strata Single-GPU

用于单张 NVIDIA L40S 48GB 的 Strata 定制推理项目。当前高性能预设是 **Qwen3.8 Flash Next GSQ-RCO IQ2_XS**，原生上下文 **262,144 token**。主模型权重不变，启动入口只向引擎和视觉编码器开放同一张 GPU。

本项目包含优化后的 C++/CUDA 源码、独立 Python API、单卡启动器、单请求和并发预设、构建脚本、复现请求及性能证据。新目录重新编译的版本已实测通过，默认使用它；原先已验证的单卡二进制作为回退保留。

## 实测速度

| 配置 | 短输入综合输出 token/s | 256,013 token 冷输入后的输出 token/s |
|---|---:|---:|
| 正常运行的旧单卡配置，GPU1 | 164.3 | 不使用其他长度数据混算 |
| 较早的原生长上下文单卡版本，GPU1 | 155.6 | 114.5 |
| 已验证单卡优化版本，两次复测，GPU1 | 203.1 / 200.9 | 162.3 / 161.6 |
| **本项目源码重新编译，GPU0** | **202.2** | **162.0** |

相对于正常单卡的 164.3，优化后的短输入提升约 **22%–24%**；对同一张 GPU1 的同一条长输入请求，输出由 114.5 提升到约 162，提升约 **41%–42%**。GPU0 的新编译成绩单独列出，不与 GPU1 的历史结果混算加速比。

短输入是中文、英文、代码各三次中位数的几何平均，每条输出512 token，温度0、关闭思考、缓存命中0。新编译版的三类中位数为186.8 / 179.6 / 246.5；不是每类任务都超过200。长输入输出阶段为162.0，输入处理本身约132秒，整个请求并非只需3秒。长输入复测为一次，不能作为所有任务的速度下限。

新编译版的9条短回复和1条完整长回复，与已验证单卡参考全文一致，长文五个检索点全部正确。主权重和int8 KV精度保持；此项回归不等同于完整的智力评测。详细对照及不同基线的限制见[性能说明](docs/performance.md)。

## 本服务器直接使用

当前目录已配置独立`.venv`并链接现有模型资产。默认GPU0、端口8090；启动器不控制其他项目的服务。

```bash
cd /home/student_su/qwen/release/strata-single-GPU
./start.sh
```

可以指定一张卡和端口；当前正式Swift占用GPU1、2，测试使用空闲GPU0。

```bash
./start.sh --gpu 0 --port 8090 --max-concurrency 1
./start.sh --gpu 0 --port 8090 --max-concurrency 6
./start.sh --gpu 0 --dry-run
```

单请求默认`engine/strata-single-built`，使用本轮单卡优化；并发2–6自动选用此前独立验证的`engine/strata-concurrent-validated`和对应预设。表中200/160是单请求成绩，不是六路并发下每路的速度。输入预处理串行，多个会话的输出可重叠，超出并发限额的请求排队。

API兼容OpenAI和Anthropic，兼容模型ID保留，额外接受`qwen3.8 flash`别名：

```text
http://127.0.0.1:8090/v1/chat/completions
http://127.0.0.1:8090/v1/messages
```

思考、工具调用、图片、原生视频8fps及请求排队已通过真实模型检查。视频沿用原生mtmd处理及时间信息；音频输入未启用。默认请求采样配置沿用当前服务，不替调用方增加思考额度。模型容量仍为262144，输入和输出共用该窗口。

## 安装与从源码构建

验证环境：Linux x86_64、NVIDIA L40S 48GB、CUDA12.5、驱动580.173.02、Python3.13.9及本目录requirements版本；构建默认sm_89。其他显卡或操作系统未取得本表成绩。

```bash
./install-python.sh
CUDACXX=/usr/local/cuda-12.5/bin/nvcc ./build.sh
```

llama.cpp/ggml及原生图像编码器源码已经放在`third_party/llama.cpp`，构建主引擎不需要从网上获取这个依赖。主引擎编译产物为`engine/strata-single-built`，校验值会更新到`engine/manifest.json`。旧版回退：

```bash
./start.sh --engine validated --gpu 0
```

模型及运行资产与源码分开存放。主模型约68GB，不包含在源码或引擎发布包中。已有Strata资产时，可使用独立的资产目录或只创建链接：

```bash
.venv/bin/python tools/attach_assets.py --source /path/to/existing/Strata
./start.sh --assets /path/to/asset-root --gpu 0
```

所需文件布局和编码器重建方式见[构建与资产说明](docs/build-and-assets.md)。普通启动不访问原项目的源码、Python环境或服务配置；本服务器仅共享已经下载的模型和数据资产。

## 复现与检查

```bash
.venv/bin/python tools/test_single_gpu.py
.venv/bin/python -m unittest serve.test_local_api serve.test_concurrent
ctest --test-dir build-single -R '^(conversation_cache_test|conversation_memory_test)$' --output-on-failure
.venv/bin/python tools/benchmark_release.py --url http://127.0.0.1:8090 --device-label '你的实际显卡/NUMA配置' --long
.venv/bin/python tools/smoke_api.py --url http://127.0.0.1:8090
```

源码、二进制和记录的对应关系在[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)、[引擎清单](engine/manifest.json)和[新构建复测](evidence/release-build-summary.json)。源码归档和Linux sm_89引擎归档位于`artifacts/`。重新构建后要分别报告新测量，不复用旧二进制的速度标签。

本项目基于Niko1221/Strata，保留原[MIT许可证](LICENSE)；llama.cpp/ggml保留其依赖目录中的许可证。模型权重单独获取。上游说明快照在`docs/upstream/`，本定制版使用`start.sh`作为启动入口。
