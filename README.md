# VLM_recognition


# 在CLI上 使用 LLaMA.cpp 跑 GGUF 模型（以 Gemma-3-27B 為例）

本教學將引導如何在任何支援 CUDA 的機器上，下載並執行 Hugging Face 上的 `.gguf` 模型，並結合影像輸入進行推理（以 Google 的 `gemma-3-27b-it-qat-q4_0-gguf` 為範例）。

---

## 環境需求

- 支援 CUDA 的 GPU（NVIDIA）
- 已安裝：
  - `git`
  - `gcc-11`, `g++-11`
  - `cmake`
  - `make`
  - `huggingface-cli`
  - CUDA 工具鏈

---

## 安裝步驟

### 1. 下載 `llama.cpp` 原始碼並設定環境

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```
```bash
git submodule update --init --recursive
sudo apt-get install libcurl4-openssl-dev
```

### 2. 建立 build 目錄

```bash
mkdir build
cd build
```
### 3. 執行 CMake，啟用 CUDA（GPU 加速）
```bash
cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11

make -j$(nproc)
```
### 如果g++-11 跑不起來 可以用g++ -10(需先將原先的build檔刪除重新建立再執行以下)
```bash
sudo apt install gcc-10 g++-10

cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_C_COMPILER=gcc-10 \
  -DCMAKE_CXX_COMPILER=g++-10 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 \
  -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80" \
```
### 4. 登入huggingface
```bash
huggingface-cli login
```
### 5. 下載gemma-3-27b-it-q4_0.gguf模型
```bash
huggingface-cli download google/gemma-3-27b-it-qat-q4_0-gguf gemma-3-27b-it-q4_0.gguf
huggingface-cli download google/gemma-3-27b-it-qat-q4_0-gguf mmproj-model-f16-27B.gguf
```
檔案會被儲存在 ~/.cache/huggingface/ 下自動建立的路徑中

### 6.載入 .gguf 模型，執行一次多模態（圖像 + prompt）推論：
```bash
./bin/llama-mtmd-cli \
  -m ~/.cache/huggingface/hub/models--google--gemma-3-27b-it-qat-q4_0-gguf/snapshots/<snapshot_hash>/gemma-3-27b-it-q4_0.gguf \
  --mmproj ~/.cache/huggingface/hub/models--google--gemma-3-27b-it-qat-q4_0-gguf/snapshots/<snapshot_hash>/mmproj-model-f16-27B.gguf \
  -p "以繁體中文回答，完整印出這張醫療收據的所有內容和文字，並保持原始排版和表格結構" \
  --image /path/to/your/image.jpg \
  --gpu-layers 99 \
  --temp 0.7
```
### <snapshot_hash>要改成實際的檔案位置
### 在4090的情況下可以全速跑，vram消耗約21G，大約20到25秒可以跑完一張醫療收據，在4080的環境下，gpu-layers只能大約設成35，剩下用cpu跑(執行時間會是原先的10倍以上!)



# 跑rolmocr模型

```bash
pip install vllm
pip install bitsandbytes
```

當執行 VLLM 命令（例如 vllm serve reducto/RolmOCR）時，可能遇到以下錯誤：
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory

可以執行
```bash
echo 'export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

啟動 RolmOCR server 
```bash
vllm serve reducto/RolmOCR  --quantization bitsandbytes --dtype auto --max-num-batched-tokens 64 --max-num-seqs 1   --max-model-len 5000 
```
參數	說明
vllm serve reducto/RolmOCR	啟動 vLLM 模型伺服器並載入 reducto/RolmOCR 模型。
--quantization bitsandbytes	啟用 4-bit 壓縮（quantization）來減少記憶體使用量。使用 BitsAndBytes 函式庫。適合在記憶體有限的 GPU 上部署。
--dtype auto	自動選擇最佳的數值精度（例如 float16、bfloat16、int8）來平衡效能與記憶體使用。
--max-num-batched-tokens 64	控制每批（batch）最多處理的 token 數量上限。
--max-num-seqs 1	每次推理最多只接受 1 筆請求
--max-model-len 5000	設定模型最多接受的 token 長度。

進行醫療收據辨識
```bash
python RolmOCR_medical_receipt.py
```


# 跑internVL模型
pip install decord
pip install timm
