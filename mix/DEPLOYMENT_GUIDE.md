# Distributed-AirLLM 部署手冊 (Deployment Guide)

> **Language**: This guide is available in English and Chinese (中文)  
> **Version**: 1.0  
> **Last Updated**: 2026-01-25

## 目錄 (Table of Contents)

1. [快速開始 (Quick Start)](#快速開始-quick-start)
2. [系統需求 (System Requirements)](#系統需求-system-requirements)
3. [部署步驟 (Deployment Steps)](#部署步驟-deployment-steps)
4. [啟動方式 (How to Start)](#啟動方式-how-to-start)
5. [配置說明 (Configuration)](#配置說明-configuration)
6. [故障排除 (Troubleshooting)](#故障排除-troubleshooting)
7. [性能優化 (Performance Optimization)](#性能優化-performance-optimization)

---

## 快速開始 (Quick Start)

### 什麼是 Distributed-AirLLM？

Distributed-AirLLM 是結合了兩個專案的優勢：
- **Distributed-Llama**: 跨多個節點的張量並行處理
- **AirLLM**: 高效的分層推理與記憶體管理

**核心創新**: "共享存儲零數據移動" (Shared-Storage Zero-Data Movement)
- ✅ 每個節點在本地 SSD 上存儲完整模型
- ✅ 每個節點僅將分配的層加載到 RAM 中
- ✅ 節點之間僅傳輸激活值 (KBs)，而非權重 (GBs)
- ✅ 網絡流量從 GBs 減少到 KBs/token

### 30 秒部署 (For Experienced Users)

```bash
# 1. 準備模型文件 (所有節點需要相同的模型文件)
# Download or convert your model to distributed-llama format

# 2. 啟動 Root 節點 (C++ 主節點)
./dllama inference --model /path/to/model.m --workers 192.168.1.2:9999 192.168.1.3:9999

# 3. 在其他機器上啟動 Python Worker
cd mix/target/distributed-llama.python
python -m worker --host 0.0.0.0 --port 9999 --model /path/to/model.m
```

---

## 系統需求 (System Requirements)

### 硬件需求 (Hardware Requirements)

#### Root 節點 (C++ 主節點)
- **CPU**: x86_64 with AVX2 or ARM with NEON
- **RAM**: Depends on model size and assigned layers
  - 8B model: 8-16 GB
  - 70B model: 40-80 GB (distributed across nodes)
- **Storage**: SSD recommended for model file (same as model size)
- **Network**: Gigabit Ethernet minimum, 10GbE recommended

#### Worker 節點 (Python Workers)
- **CPU**: Any modern CPU (AVX2 recommended for optimal performance)
- **RAM**: 4-32 GB depending on assigned layers
  - Formula: `RAM_needed = (model_size / num_nodes) + 2GB overhead`
- **Storage**: SSD with full model copy (same file as root)
- **Network**: Same network as root node

### 軟件需求 (Software Requirements)

#### Root 節點
- **OS**: Linux, macOS, or Windows
- **Compiler**: GCC 9+, Clang 10+, or MSVC 2019+
- **Python**: 3.8+ (for launch.py script)
- **C++ Standard**: C++11 or later

#### Python Worker 節點
- **OS**: Linux (recommended), macOS, or Windows
- **Python**: 3.8 or later
- **Dependencies**:
  ```bash
  pip install numpy psutil
  # Optional for C++ acceleration:
  pip install pybind11
  ```

### 網絡需求 (Network Requirements)

- All nodes must be on the same local network
- Firewall rules should allow TCP connections on worker ports (default: 9999)
- Low latency is critical (< 1ms preferred)
- Bandwidth: 1 Gbps minimum, 10 Gbps for large models

---

## 部署步驟 (Deployment Steps)

### 步驟 1: 準備模型文件 (Prepare Model File)

#### 選項 A: 使用預轉換模型 (Use Pre-converted Models)

```bash
# 在 root 節點上，使用 launch.py 自動下載模型
cd /home/runner/work/distributed-llama/distributed-llama
python launch.py llama3_1_8b_instruct_q40  # 下載並啟動 8B 模型
```

#### 選項 B: 轉換自己的模型 (Convert Your Own Model)

參考主項目文檔:
```bash
# See: docs/HOW_TO_CONVERT_HF_MODEL.md
# 從 Hugging Face 模型轉換
```

#### 步驟 1.1: 確保所有節點都有相同的模型文件

**關鍵要求**: 每個節點必須有**相同的模型文件**在相同或不同的路徑上。

方法 1: 共享網絡存儲 (推薦)
```bash
# 使用 NFS 或 SMB 掛載共享存儲
sudo mount -t nfs 192.168.1.1:/models /mnt/models
# 所有節點指向: /mnt/models/llama-8b-q40.m
```

方法 2: 複製文件到每個節點
```bash
# 從 root 節點複製到每個 worker
scp /path/to/model.m user@worker1:/path/to/model.m
scp /path/to/model.m user@worker2:/path/to/model.m

# 驗證文件相同 (校驗和)
md5sum /path/to/model.m  # 在所有節點上運行，確保輸出相同
```

### 步驟 2: 構建 Root 節點 (Build Root Node)

Root 節點使用原始 distributed-llama 的 C++ 實現。

```bash
cd /home/runner/work/distributed-llama/distributed-llama

# 構建 C++ 主程序
make dllama

# 驗證構建
./dllama --help
```

### 步驟 3: 設置 Python Worker 環境 (Setup Python Worker)

在每個 worker 節點上:

```bash
# 1. 安裝 Python 依賴
cd /home/runner/work/distributed-llama/distributed-llama/mix/target/distributed-llama.python
pip install -r requirements.txt

# 2. (可選但推薦) 構建 C++ 擴展以獲得 5-15x 加速
cd ../airllm/cpp_ext
python setup.py build_ext --inplace

# 3. 驗證安裝
python -c "from worker import Worker; print('Worker module loaded successfully')"
```

### 步驟 4: 配置網絡 (Configure Network)

#### 確定 IP 地址

```bash
# 在每個節點上查找 IP
ip addr show | grep inet

# 示例輸出:
# Root: 192.168.1.10
# Worker1: 192.168.1.11
# Worker2: 192.168.1.12
```

#### 配置防火牆 (如果需要)

```bash
# Ubuntu/Debian
sudo ufw allow 9999/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=9999/tcp --permanent
sudo firewall-cmd --reload

# 驗證連接性
# 在 root 節點上測試 worker 連接
nc -zv 192.168.1.11 9999
```

---

## 啟動方式 (How to Start)

### 場景 1: 單機測試 (Single Machine Test)

適合開發和測試。

```bash
# 終端 1: 啟動 Root 節點
cd /home/runner/work/distributed-llama/distributed-llama
./dllama inference \
    --model /path/to/model.m \
    --tokenizer /path/to/tokenizer.t \
    --workers 127.0.0.1:9999 \
    --nthreads 4

# 終端 2: 啟動 Python Worker
cd mix/target/distributed-llama.python
python -m worker \
    --host 127.0.0.1 \
    --port 9999 \
    --model /path/to/model.m
```

### 場景 2: 多機部署 (Multi-Machine Deployment)

#### 步驟 1: 在所有 Worker 節點上啟動 Workers

**Worker 1** (192.168.1.11):
```bash
cd /home/runner/work/distributed-llama/distributed-llama/mix/target/distributed-llama.python
python -m worker \
    --host 0.0.0.0 \
    --port 9999 \
    --model /mnt/models/llama-8b-q40.m \
    --nthreads 4
```

**Worker 2** (192.168.1.12):
```bash
cd /home/runner/work/distributed-llama/distributed-llama/mix/target/distributed-llama.python
python -m worker \
    --host 0.0.0.0 \
    --port 9999 \
    --model /mnt/models/llama-8b-q40.m \
    --nthreads 4
```

#### 步驟 2: 在 Root 節點上啟動推理

**Root Node** (192.168.1.10):
```bash
cd /home/runner/work/distributed-llama/distributed-llama

# 推理模式 (inference mode)
./dllama inference \
    --model /mnt/models/llama-8b-q40.m \
    --tokenizer /path/to/tokenizer.t \
    --workers 192.168.1.11:9999 192.168.1.12:9999 \
    --prompt "Hello, how are you?" \
    --steps 100 \
    --nthreads 4

# 或者使用聊天模式 (chat mode)
./dllama chat \
    --model /mnt/models/llama-8b-q40.m \
    --tokenizer /path/to/tokenizer.t \
    --workers 192.168.1.11:9999 192.168.1.12:9999 \
    --nthreads 4
```

### 場景 3: 使用自動啟動腳本 (Using Launch Script)

最簡單的方式是使用項目的 `launch.py`:

```bash
# 這會自動下載模型、tokenizer 並啟動 root 節點
python launch.py llama3_1_8b_instruct_q40 \
    --workers 192.168.1.11:9999 192.168.1.12:9999
```

### 獲得 Distributed-AirLLM 效果的關鍵點

要充分發揮 Distributed-AirLLM 的優勢:

1. **確保共享存儲**: 所有節點必須有相同的模型文件
2. **使用 Python Workers**: Python workers 實現了分層加載和記憶體優化
3. **配置足夠的 RAM**: 每個 worker 需要足夠的 RAM 來加載分配的層
4. **使用 C++ 加速** (可選): 構建 cpp_ext 以獲得 5-15x 的計算加速

```bash
# 構建 C++ 擴展 (在每個 worker 節點上)
cd mix/target/airllm/cpp_ext
python setup.py build_ext --inplace
```

5. **啟用激活壓縮** (未來功能): Phase 3 已實現但需要集成

---

## 配置說明 (Configuration)

### Root 節點參數

| 參數 | 說明 | 示例 |
|------|------|------|
| `--model` | 模型文件路徑 | `/path/to/model.m` |
| `--tokenizer` | Tokenizer 文件路徑 | `/path/to/tokenizer.t` |
| `--workers` | Worker 地址列表 (空格分隔) | `192.168.1.2:9999 192.168.1.3:9999` |
| `--nthreads` | CPU 線程數 | `4` |
| `--max-seq-len` | 最大序列長度 | `4096` |
| `--buffer-float-type` | 緩衝區浮點類型 | `q80` (default) or `f32` |

### Python Worker 參數

| 參數 | 說明 | 示例 |
|------|------|------|
| `--host` | 綁定地址 | `0.0.0.0` (all interfaces) |
| `--port` | 綁定端口 | `9999` |
| `--model` | 模型文件路徑 | `/path/to/model.m` |
| `--nthreads` | CPU 線程數 | `4` |

### 環境變量

```bash
# 控制 OpenMP 線程數 (如果使用 C++ 擴展)
export OMP_NUM_THREADS=4

# 控制 NumPy 線程數
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 啟用調試輸出
export DLLAMA_DEBUG=1
```

---

## 故障排除 (Troubleshooting)

### 問題 1: Worker 無法連接到 Root

**症狀**: Worker 啟動後沒有任何輸出，或顯示連接超時。

**解決方案**:
```bash
# 1. 檢查 root 節點是否正在等待連接
# 確保 root 節點在 worker 之前或同時啟動

# 2. 檢查網絡連接
ping 192.168.1.10  # ping root 節點
nc -zv 192.168.1.10 9999  # 測試端口

# 3. 檢查防火牆
sudo ufw status
sudo ufw allow 9999/tcp

# 4. 檢查 IP 地址配置
# 確保 root 節點使用的 worker 地址正確
```

### 問題 2: 模型文件不匹配

**症狀**: Worker 啟動後報錯 "Model file mismatch" 或校驗和錯誤。

**解決方案**:
```bash
# 驗證所有節點的模型文件相同
md5sum /path/to/model.m  # 在所有節點上運行

# 如果不同，重新複製模型文件
scp root:/path/to/model.m worker:/path/to/model.m
```

### 問題 3: 記憶體不足 (Out of Memory)

**症狀**: Worker 啟動後崩潰或顯示 "OOM" 錯誤。

**解決方案**:
```bash
# 1. 增加交換空間
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 2. 減少分配的層數
# 增加更多 worker 節點以分散負載

# 3. 使用更小的模型或更高的量化
# Q40 vs F32: Q40 使用 ~40% 的記憶體
```

### 問題 4: 性能低於預期

**症狀**: 推理速度很慢，tokens/second 低於預期。

**解決方案**:
```bash
# 1. 構建 C++ 擴展
cd mix/target/airllm/cpp_ext
python setup.py build_ext --inplace

# 2. 驗證 SIMD 優化已啟用
python -c "import tensor_ops_cpp; print(tensor_ops_cpp.get_optimization_info())"

# 3. 調整線程數
# 設置為 CPU 核心數或略少
./dllama ... --nthreads $(nproc)

# 4. 使用更快的網絡
# 10GbE > 1GbE > WiFi
```

### 問題 5: "Illegal instruction" 錯誤

**症狀**: Worker 或 C++ 擴展崩潰並顯示 "Illegal instruction"。

**解決方案**:
```bash
# 這表示二進制文件使用了 CPU 不支持的指令
# 在目標機器上重新構建

# 對於 C++ 擴展:
cd mix/target/airllm/cpp_ext
rm -rf build *.so
python setup.py build_ext --inplace

# 對於 C++ root 節點:
cd /home/runner/work/distributed-llama/distributed-llama
make clean
make dllama
```

### 問題 6: Python Worker 沒有輸出

**症狀**: Worker 啟動後沒有任何日誌輸出。

**解決方案**:
```bash
# 啟用詳細日誌
python -m worker --host 0.0.0.0 --port 9999 --model /path/to/model.m -v

# 或設置環境變量
export PYTHONUNBUFFERED=1
python -m worker ...
```

---

## 性能優化 (Performance Optimization)

### 1. 硬件優化

#### CPU
- 使用支持 AVX2 的現代 CPU
- 啟用超線程 (Hyper-Threading)
- 確保 CPU 處於性能模式 (非節能模式)

```bash
# Linux: 設置 CPU 性能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

#### 存儲
- 使用 NVMe SSD 而非 SATA SSD
- 避免使用機械硬盤 (HDD)
- 確保模型文件在本地存儲 (不在網絡掛載上，除非使用高速 NFS)

#### 網絡
- 使用 10GbE 或更快的網絡
- 確保所有節點在同一交換機上
- 避免使用 WiFi 進行推理

### 2. 軟件優化

#### 構建 C++ 擴展 (關鍵!)

```bash
cd mix/target/airllm/cpp_ext
python setup.py build_ext --inplace

# 驗證優化已啟用
python -c "import tensor_ops_cpp; print(tensor_ops_cpp.get_optimization_info())"
# 應該顯示: "AVX2: enabled, FMA: enabled, OpenMP: enabled"
```

**預期加速**:
- RMS normalization: 5-15x
- SiLU/GELU activation: 2-4x
- 總體推理: 3-7x

#### 調整線程數

```bash
# 設置為 CPU 物理核心數
# 避免設置為超線程後的總核心數
export OMP_NUM_THREADS=8  # 例如: 8 物理核心
```

#### 使用適當的量化

| 量化級別 | 記憶體 | 質量 | 推薦場景 |
|----------|--------|------|----------|
| F32 | 100% | 最高 | 小模型，充足 RAM |
| Q80 | ~80% | 極高 | 平衡選擇 |
| Q40 | ~40% | 高 | 大模型，有限 RAM |

### 3. 分布式優化

#### 最優節點數

```bash
# 節點數必須是 2 的冪: 1, 2, 4, 8, 16...
# 最大節點數 = 模型的 KV heads 數量

# 例如: Llama 3.1 8B 有 8 個 KV heads
# 可以使用 1, 2, 4, 或 8 個節點
```

#### 平衡負載

```bash
# 確保每個 worker 有相似的硬件配置
# 避免混合快速和慢速節點
```

### 4. 監控和調試

#### 監控 Worker 性能

```bash
# 在 worker 節點上監控資源使用
htop  # CPU 和記憶體
iotop  # 磁盤 I/O
iftop  # 網絡流量
```

#### 性能分析

```bash
# 使用 profiling 腳本
cd mix/target
python profile_worker.py

# 這會顯示每個操作的耗時和瓶頸
```

---

## 附錄: 完整部署示例 (Appendix: Complete Deployment Example)

### 場景: 3 節點部署 Llama 3.1 8B

**硬件**:
- Node 1 (root): Intel i7-12700K, 32GB RAM, 1TB NVMe SSD
- Node 2 (worker): Intel i5-12400, 16GB RAM, 512GB NVMe SSD  
- Node 3 (worker): Intel i5-12400, 16GB RAM, 512GB NVMe SSD

**網絡**: 1 Gbps Ethernet

### 步驟 1: 準備模型 (在 root 節點)

```bash
cd /home/runner/work/distributed-llama/distributed-llama
python launch.py llama3_1_8b_instruct_q40 --download-only

# 模型下載到: /home/user/.cache/distributed-llama/
MODEL_PATH=/home/user/.cache/distributed-llama/dllama_model_meta-llama-3.1-8b-instruct_q40.m
TOKEN_PATH=/home/user/.cache/distributed-llama/dllama_tokenizer_llama3.t
```

### 步驟 2: 分發模型到 Workers

```bash
# 從 root 複製到 workers
scp $MODEL_PATH user@192.168.1.11:/data/models/llama-8b-q40.m
scp $MODEL_PATH user@192.168.1.12:/data/models/llama-8b-q40.m

# 驗證校驗和
md5sum $MODEL_PATH
ssh user@192.168.1.11 "md5sum /data/models/llama-8b-q40.m"
ssh user@192.168.1.12 "md5sum /data/models/llama-8b-q40.m"
```

### 步驟 3: 設置 Workers

```bash
# 在每個 worker 上
ssh user@192.168.1.11
cd /home/runner/work/distributed-llama/distributed-llama/mix/target/distributed-llama.python
pip install -r requirements.txt

# 構建 C++ 擴展
cd ../airllm/cpp_ext
python setup.py build_ext --inplace
```

### 步驟 4: 啟動服務

**Terminal 1** (Worker 1 - 192.168.1.11):
```bash
cd /home/runner/work/distributed-llama/distributed-llama/mix/target/distributed-llama.python
python -m worker \
    --host 0.0.0.0 \
    --port 9999 \
    --model /data/models/llama-8b-q40.m \
    --nthreads 6
```

**Terminal 2** (Worker 2 - 192.168.1.12):
```bash
cd /home/runner/work/distributed-llama/distributed-llama/mix/target/distributed-llama.python
python -m worker \
    --host 0.0.0.0 \
    --port 9999 \
    --model /data/models/llama-8b-q40.m \
    --nthreads 6
```

**Terminal 3** (Root - 192.168.1.10):
```bash
cd /home/runner/work/distributed-llama/distributed-llama
./dllama chat \
    --model $MODEL_PATH \
    --tokenizer $TOKEN_PATH \
    --workers 192.168.1.11:9999 192.168.1.12:9999 \
    --nthreads 8 \
    --buffer-float-type q80
```

### 步驟 5: 測試推理

```bash
# 在聊天界面中測試
> Hello, how are you?
# 應該看到模型生成回應

# 性能測試
./dllama inference \
    --model $MODEL_PATH \
    --tokenizer $TOKEN_PATH \
    --workers 192.168.1.11:9999 192.168.1.12:9999 \
    --prompt "Once upon a time" \
    --steps 100 \
    --nthreads 8
```

### 預期性能

- **吞吐量**: ~20-40 tokens/second (取決於網絡和 CPU)
- **記憶體使用**: 每個 worker ~4-6 GB
- **網絡流量**: ~100-500 KB/token (with compression)

---

## 參考資源 (References)

- [主項目文檔](../README.md)
- [C++ 擴展文檔](target/airllm/cpp_ext/README.md)
- [實現總結](IMPLEMENTATION_SUMMARY.md)
- [Phase 3 & 4 總結](PHASE3_4_SUMMARY.md)
- [原始 Distributed-Llama](https://github.com/b4rtaz/distributed-llama)
- [AirLLM 項目](https://github.com/lyogavin/airllm)

---

## 支持和貢獻 (Support and Contributing)

如有問題或建議，請：
1. 查看 [故障排除](#故障排除-troubleshooting) 部分
2. 查看項目 Issues 和 Discussions
3. 創建新的 Issue 報告問題
4. 提交 Pull Request 改進文檔或代碼

---

**License**: MIT (same as parent project)
