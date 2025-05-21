# VLM_recognition

# ⚠️ 注意事項

- **RolmOCR 與 InternVL3 目前僅建議於 NVIDIA 4090 GPU 上執行。**
    - 雖然在 4090 上 VRAM 可壓到 13GB 以下，但在 4080 上執行 RolmOCR 只會出現「`!!!!!!!!!!!`」異常輸出，InternVL 也只會出現亂碼輸出，**目前尚未釐清原因**。

---

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



# RolmOCR 模型部署與使用說明

---

## 1. 安裝必要套件

```bash
pip install vllm
pip install bitsandbytes
```

---

## 2. 常見錯誤與解決方式

若執行 VLLM 命令（如 `vllm serve reducto/RolmOCR`）時遇到以下錯誤：
```bash
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```
請執行以下指令，設定 CUDA 路徑：

```bash
echo 'export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 3. 啟動 RolmOCR 伺服器

在辨識前，請先啟動 RolmOCR server：

```bash
vllm serve reducto/RolmOCR \
  --quantization bitsandbytes \
  --dtype auto \
  --max-num-batched-tokens 64 \
  --max-num-seqs 1 \
  --max-model-len 5000
  --gpu-memory-utilization 0.6 #可以降低vram消耗，最低13GB的vram就可以使用
```

### 參數說明

| 參數                        | 說明                                                                                   |
|-----------------------------|----------------------------------------------------------------------------------------|
| `vllm serve reducto/RolmOCR`| 啟動 vLLM 模型伺服器並載入 reducto/RolmOCR 模型                                        |
| `--quantization bitsandbytes`| 啟用 8-bit 壓縮（quantization），減少記憶體用量，加速推理速度                 |
| `--dtype auto`              | 自動選擇最佳數值精度（如 float16、bfloat16、int8）以平衡效能與記憶體                   |
| `--max-num-batched-tokens 64`| 每批最多處理 64 個 token                                                               |
| `--max-num-seqs 1`          | 每次推理僅接受 1 筆請求                                                                |
| `--max-model-len 5000`      | 設定模型最多接受的 token 長度                                                          |
| `--gpu-memory-utilization 0.6`| 設定模型最多能使用多少的vram                                                        |

---

## 4. 辨識流程

### 醫療收據辨識

```bash
python rolmocr_medical_receipt.py
```

### 發票辨識

```bash
python rolmocr_invoice.py
```

### 前端頁面進行發票辨識

```bash
python rolmocr_invoice_fronted_page.py
```

---

## 5. 前端頁面展示

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/55e66a2f-d606-46b7-87c7-e06c9b365023" width="700"/></td>
    <td><img src="https://github.com/user-attachments/assets/91d3bd7e-c0e8-4d02-9766-c248f4dbedec" width="700"/></td>
  </tr>
</table>


# InternVL 模型部署與使用說明
```bash
pip install decord
pip install timm
pip install accelerate
```

進行醫療收據辨識
```bash
python InternVL3-8B_medical_receipt.py
```
降精度 InternVL3進行醫療收據辨識(4 bit)
```bash
InternVL3-8B_load_4_bit_medical_receipt.py
```
### 模型效能比較表

| 模型名稱 | 醫療收據辨識時間 | 表現穩定性 | 幻覺情況 | 備註 |
|:---:|:---:|:---:|:---:|:---|
| **gemma-3-27b-it-qat-q4_0** | 最慢（通常 > 20 秒） | 穩定的爛 | 嚴重，常捏造內容 | 應該是訓練資料中較少醫療資訊，關於醫療品項，時常會輸出毫不相關的回答，例如：<br>藥事服務費→衛生紙、注射技術費→登錄費。<br>表現最差，不建議使用。 |
| **Rolmocr** | 大多圖片都能在 10 秒以內辨識完畢 | 尚可（少時候會出現連續輸出一排相同字的情況） | 較少 | 在表格辨識上，有時會有遺失的資訊，例如榮總的收據，一排是健保，一排是自費費用，但兩排都寫了重複的欄位，像是：<br>健保點數(含部分負擔) 自費費用<br>藥費 7023 藥費 0<br>這時他的輸出會遺漏健保費用資訊：<br> \| 健保點數(含部分負擔) \| 自費費用 \|<br> \| 藥費 \| 7023 \|<br>（本來是健保的藥費，變成了自費費用）<br>又或是在參差不齊的表格上（中間沒有格線）有連續三排資料時，第三排資訊往往會被忽略。<br>在表格以外的資訊上，大多表現不錯，因此在醫療辨識或是名片辨識上是目前最佳選擇。 |
| **InternVL3** | 輸出時間大概是 rolmocr 的 1.5 倍，大多圖片都能在 15 秒以內辨識完畢 | 普通（偶爾出現連續輸出一排相同字的情況） | 比 Rolmocr 多一點 | 因為 rolmocr 有時候表格資訊會被忽略，因此有些情況下，internVL 的表格輸出會優於 rolmocr，例如 InternVL_16bit 在榮總的收據辨識上表現比 Rolmocr_16bit 來得好（InternVL3_4bit 輸出格式有誤），但整體上 Rolmocr 還是優於 InternVL3。 |

---

辨識效果比較:https://miro.com/app/board/uXjVI8-si30=/
