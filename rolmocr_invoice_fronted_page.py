# HOST YOUR OPENAI COMPATIBLE API WITH THE FOLLOWING COMMAND in VLLM:
# export VLLM_USE_V1=1
# vllm serve reducto/RolmOCR 
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import subprocess
import time
import os
import uvicorn
import argparse
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import io
import base64
from openai import OpenAI

client = OpenAI(api_key="123", base_url="http://localhost:8000/v1")

model = "reducto/RolmOCR"

def compress_image(image_input, max_size=1600, quality=85):
    """
    壓縮圖片，保持原始寬高比，不強制調整為正方形。
    :param image_path: 圖片路徑
    :param max_size: 最大邊長（單邊），預設為 1600
    :param quality: JPEG 壓縮品質，預設為 85
    :return: base64 編碼的圖片數據
    """
    # 打開圖片
    if isinstance(image_input, str):
        # 如果是路徑，打開圖片
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        # 如果已經是 PIL Image 對象，直接使用
        img = image_input
    else:
        raise ValueError("輸入必須是圖片路徑或 PIL Image 對象")
    
    # 檢查圖片原始尺寸
    original_width, original_height = img.size
    #print(f"原始圖片尺寸：{original_width}x{original_height}")
    
    # 計算原始寬高比
    aspect_ratio = original_width / original_height
    #print(f"原始寬高比：{aspect_ratio:.2f}")
    
    # 如果圖片最大邊長小於 max_size，則不需要壓縮
    if max(original_width, original_height) <= max_size:
        print("圖片已是低解析度，跳過壓縮。")
    else:
        print(f"圖片最大邊長高於 {max_size}，進行壓縮...")
        # 使用 thumbnail 縮放圖片，保持寬高比
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    print("\n")
    # 獲取縮放後的尺寸
    new_width, new_height = img.size
   # print(f"縮放後圖片尺寸：{new_width}x{new_height}")
    
    # 保存圖片並轉為 base64
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=quality)
    compressed_base64 = base64.b64encode(output.getvalue()).decode("utf-8")
    
    # 打印壓縮後的資訊
   # print(f"壓縮後 base64 長度：{len(compressed_base64)}")
    
    return compressed_base64

def encode_image(image_path):
    """
    將圖片轉為 base64 編碼，並在需要時進行壓縮。
    :param image_path: 圖片路徑
    :return: base64 編碼的字符串
    """
    # 使用 compress_image 函數處理圖片
    img_base64 = compress_image(image_path, max_size=1600, quality=85)
    return img_base64
def ocr_total_only(img_base64, custom_prompt=None, temperature=0):
    # 使用默認提示詞或自定義提示詞
    prompt = custom_prompt if custom_prompt else """請你擷取下列三個欄位：「銷售額合計」、「營業稅」、「總計」，**僅當發票上明確出現該詞語或數字時，才允許輸出數字**，否則請填「無」，給我純文字就好，禁止輸出表格。
注意:總計不一定等於銷售額合計+營業稅，因此你**絕對不能進行任何推理、計算或合邏輯推測**，只能逐字匹配文字欄位。
正確範例：
銷售額合計: 2354
營業稅: 100
總計: 2354"""
    
    # 呼叫 OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        temperature=temperature,
        presence_penalty=0.7,
        max_tokens=300,
        stop=["END_OF_OUTPUT"],
    )
    
    # 返回結果
    return response.choices[0].message.content

def ocr_page_with_rolm(img_base64, custom_prompt=None, temperature=0):
    # 使用默認提示詞或自定義提示詞
    prompt = custom_prompt if custom_prompt else "完整印出這張發票的所有資訊，從上到下擷取所有文本，包括標題、發票號碼、日期、表格、賣方/買方資訊、備註、總計和商業印章上的資訊。發票可能包含印刷文字和手寫文字，並且會有部分英文。優先處理中文文字，除非英文是關鍵字段的一部分。請特別注意捕捉最上方的標題和最下方的文字，因為這些部分對發票的有效性至關重要。完成後，請寫下 'END_OF_OUTPUT'，請勿一直重複輸出相同資訊"
    
    # 呼叫 OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        temperature=temperature,
        presence_penalty=0.7,
        max_tokens=3000,
        stop=["END_OF_OUTPUT"],
    )

    # 返回結果
    return response.choices[0].message.content


# 創建 FastAPI 應用
app = FastAPI(title="OCR API with vLLM and RolmOCR")

# 定義首頁（HTML 介面）
@app.get("/", response_class=HTMLResponse)
async def get_ocr_page():
    html_content = """
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>發票 OCR 處理</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .upload-section {
            margin: 20px 0;
            text-align: center;
        }
        input[type="file"] {
            padding: 10px;
            margin-bottom: 10px;
        }
        button {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 5px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .results-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .image-preview {
            flex: 1;
            min-width: 300px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            padding: 15px;
            text-align: center;
        }
        .result {
            flex: 2;
            min-width: 300px;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            white-space: pre-wrap;
        }
        .preview-image {
            width: 100%;
            max-width: 100%;
            max-height: 90vh;
            display: none;
            cursor: pointer;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: transform 0.2s ease-in-out;
        }
        .loading {
            display: none;
            margin-top: 10px;
            color: #007bff;
        }
        .options-section {
            margin-top: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .prompt-textarea {
            width: 100%;
            height: 100px;
            margin-top: 5px;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
        }
        .option-row {
            margin-bottom: 15px;
        }
        
        /* 標籤欄樣式 */
        .tabs {
            display: flex;
            flex-wrap: nowrap;
            overflow-x: auto;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
            background-color: white;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            padding: 5px 5px 0 5px;
        }
        .tab {
            padding: 10px 20px;
            background-color: #f1f1f1;
            border: 1px solid #ddd;
            border-bottom: none;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            cursor: pointer;
            margin-right: 5px;
            white-space: nowrap;
            display: flex;
            align-items: center;
        }
        .tab.active {
            background-color: #fff;
            border-bottom: 1px solid white;
            margin-bottom: -1px;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .close-tab {
            margin-left: 10px;
            font-size: 16px;
            color: #888;
            background: none;
            border: none;
            padding: 0 5px;
            cursor: pointer;
            border-radius: 50%;
        }
        .close-tab:hover {
            background-color: #ddd;
            color: #333;
        }
        .new-tab-btn {
            padding: 8px 12px;
            background-color: #eee;
            border: 1px solid #ddd;
            border-bottom: none;
            border-radius: 5px 5px 0 0;
            cursor: pointer;
            margin-right: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .new-tab-btn:hover {
            background-color: #ddd;
        }
        .tab-title {
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        #image-modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(0,0,0,0.9);
            justify-content: center;
            align-items: center;
        }
        #modal-image {
            max-width: 90vw;
            max-height: 90vh;
            border-radius: 10px;
        }
        @media (max-width: 768px) {
            .results-container {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <h1>發票 OCR 處理</h1>
    
    <!-- 標籤欄 -->
    <div class="tabs" id="tabs-container">
        <button class="new-tab-btn" onclick="createNewTab()">+ 新標籤頁</button>
    </div>
    
    <!-- 標籤頁內容區 -->
    <div id="tab-contents">
        <!-- 標籤頁內容將動態生成 -->
    </div>
    
    <!-- 圖片模態框 -->
    <div id="image-modal">
        <img id="modal-image">
    </div>
    
    <script>
        let tabCounter = 0;
        let activeTabId = null;
        
        // 初始化頁面時創建第一個標籤頁
        document.addEventListener('DOMContentLoaded', function() {
            createNewTab();
        });
        
        // 創建新標籤頁
        function createNewTab() {
            tabCounter++;
            const tabId = `tab-${tabCounter}`;
            const tabTitle = `OCR ${tabCounter}`;
            
            // 創建標籤按鈕
            const tabElement = document.createElement('div');
            tabElement.className = 'tab';
            tabElement.id = `${tabId}-button`;
            tabElement.innerHTML = `
                <span class="tab-title">${tabTitle}</span>
                <button class="close-tab" onclick="closeTab('${tabId}')">&times;</button>
            `;
            tabElement.addEventListener('click', function() {
                activateTab(tabId);
            });
            
            // 將標籤按鈕添加到標籤欄
            const tabsContainer = document.getElementById('tabs-container');
            tabsContainer.insertBefore(tabElement, document.querySelector('.new-tab-btn'));
            
            // 創建標籤頁內容
            const tabContent = document.createElement('div');
            tabContent.className = 'tab-content';
            tabContent.id = tabId;
            tabContent.innerHTML = `
                <div class="upload-section">
                    <input type="file" id="${tabId}-imageInput" accept="image/*" onchange="previewImage('${tabId}')">
                    <br>
                    <button onclick="processImage('${tabId}')">上傳並處理</button>
                </div>
                <div id="${tabId}-loading" class="loading">處理中，請稍候...</div>
                
                <div class="results-container">
                    <div class="image-preview">
                        <h3>上傳圖片</h3>
                        <img id="${tabId}-preview-image" class="preview-image" alt="預覽圖片" onclick="showImageModal(this.src)">
                        <p id="${tabId}-no-image">尚未選擇圖片</p>
                    </div>
                    <div id="${tabId}-result" class="result">選擇一張圖片並點擊「上傳並處理」按鈕開始 OCR 識別。</div>
                </div>
                
                <div class="options-section">
                    <h3>OCR 處理選項</h3>
                    <div class="option-row">
                        <input type="checkbox" id="${tabId}-enableTotalExtraction" checked>
                        <label for="${tabId}-enableTotalExtraction">啟用總計金額提取</label>
                    </div>
                    
                    <div class="option-row">
                        <label for="${tabId}-totalPrompt">總計金額提取提示詞：</label>
                        <textarea id="${tabId}-totalPrompt" class="prompt-textarea">請你擷取下列三個欄位：「銷售額合計」、「營業稅」、「總計」，**僅當發票上明確出現該詞語或數字時，才允許輸出數字**，否則請填「無」，給我純文字就好，禁止輸出表格。
注意:總計不一定等於銷售額合計+營業稅，因此你**絕對不能進行任何推理、計算或邏輯推測**，只能逐字匹配文字欄位。
正確範例：
銷售額合計: 2354
營業稅: 100
總計: 2354
</textarea>
                    </div>
                    
                    <div class="option-row">
                        <label for="${tabId}-fullPrompt">完整發票提取提示詞：</label>
                        <textarea id="${tabId}-fullPrompt" class="prompt-textarea">完整印出這張發票的所有資訊，從上到下擷取所有文本，包括標題、發票號碼、日期、表格、賣方/買方資訊、備註、總計和商業印章上的資訊。發票可能包含印刷文字和手寫文字，並且會有部分英文。優先處理中文文字，除非英文是關鍵字段的一部分。請特別注意捕捉最上方的標題和最下方的文字，因為這些部分對發票的有效性至關重要。完成後，請寫下 'END_OF_OUTPUT'，請勿一直重複輸出相同資訊。</textarea>
                    </div>
                    
                    <div class="option-row">
                        <label for="${tabId}-temperature">溫度設定 (temperature):</label>
                        <input type="number" id="${tabId}-temperature" name="temperature" step="0.1" min="0" max="1" value="0">
                    </div>

                    <div class="option-row">
                        <input type="checkbox" id="${tabId}-enableFullExtraction" checked>
                        <label for="${tabId}-enableFullExtraction">啟用完整發票提取</label>
                    </div>
                </div>
            `;
            
            // 將標籤頁內容添加到頁面
            document.getElementById('tab-contents').appendChild(tabContent);
            
            // 激活新創建的標籤頁
            activateTab(tabId);
        }
        
        // 激活指定標籤頁
        function activateTab(tabId) {
            // 更新當前活動標籤頁
            activeTabId = tabId;
            
            // 移除所有標籤和內容的活動狀態
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // 設置當前標籤和內容為活動狀態
            document.getElementById(`${tabId}-button`).classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }
        
        // 關閉標籤頁
        function closeTab(tabId) {
            event.stopPropagation(); // 防止事件冒泡到標籤點擊
            
            // 移除標籤和內容
            document.getElementById(`${tabId}-button`).remove();
            document.getElementById(tabId).remove();
            
            // 檢查是否還有標籤頁
            const remainingTabs = document.querySelectorAll('.tab');
            if (remainingTabs.length > 0) {
                // 如果關閉的是當前活動標籤頁，則激活最後一個標籤頁
                if (activeTabId === tabId) {
                    const lastTabId = remainingTabs[remainingTabs.length - 1].id.replace('-button', '');
                    activateTab(lastTabId);
                }
            } else {
                // 如果沒有標籤頁了，則創建一個新的
                createNewTab();
            }
        }
        
        // 預覽圖片
        function previewImage(tabId) {
            const imageInput = document.getElementById(`${tabId}-imageInput`);
            const previewImage = document.getElementById(`${tabId}-preview-image`);
            const noImageText = document.getElementById(`${tabId}-no-image`);
            
            if (imageInput.files && imageInput.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewImage.style.display = 'block';
                    noImageText.style.display = 'none';
                    
                    // 更新標籤頁標題為文件名
                    const fileName = imageInput.files[0].name;
                    document.querySelector(`#${tabId}-button .tab-title`).textContent = fileName;
                }
                
                reader.readAsDataURL(imageInput.files[0]);
            }
        }
        
        // 處理圖片
        async function processImage(tabId) {
            const imageInput = document.getElementById(`${tabId}-imageInput`);
            const resultDiv = document.getElementById(`${tabId}-result`);
            const loadingDiv = document.getElementById(`${tabId}-loading`);
            const enableTotalExtraction = document.getElementById(`${tabId}-enableTotalExtraction`).checked;
            const totalPrompt = document.getElementById(`${tabId}-totalPrompt`).value.trim();
            const fullPrompt = document.getElementById(`${tabId}-fullPrompt`).value.trim();
            const enableFullExtraction = document.getElementById(`${tabId}-enableFullExtraction`).checked;
            const temperature = document.getElementById(`${tabId}-temperature`).value;

            if (!imageInput.files[0]) {
                alert('請選擇一張圖片！');
                return;
            }

            loadingDiv.style.display = 'block';
            resultDiv.textContent = '處理中...';

            const formData = new FormData();
            formData.append('file', imageInput.files[0]);
            formData.append('enableTotalExtraction', enableTotalExtraction);
            formData.append('temperature', temperature);
            formData.append('enableFullExtraction', enableFullExtraction);
            
            if (totalPrompt) {
                formData.append('totalPrompt', totalPrompt);
            }
            
            if (fullPrompt) {
                formData.append('fullPrompt', fullPrompt);
            }

            try {
                const response = await fetch('/ocr', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (response.ok) {
                    resultDiv.textContent = data.ocr_result;
                } else {
                    resultDiv.textContent = '錯誤：' + data.detail;
                }
            } catch (error) {
                resultDiv.textContent = '處理失敗：' + error.message;
            } finally {
                loadingDiv.style.display = 'none';
            }
        }
        
        // 顯示圖片模態框
        function showImageModal(src) {
            const modal = document.getElementById('image-modal');
            const modalImg = document.getElementById('modal-image');
            modalImg.src = src;
            modal.style.display = 'flex';
        }
        
        // 關閉圖片模態框
        document.getElementById('image-modal').addEventListener('click', function() {
            this.style.display = 'none';
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# 定義 OCR API 端點
@app.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    enableTotalExtraction: bool = Form(True),
    enableFullExtraction: bool = Form(True),
    totalPrompt: str = Form(None),
    fullPrompt: str = Form(None),
    temperature: float = Form(0),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="請上傳圖片文件（JPEG/PNG）")

    image_data = await file.read()
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法讀取圖片：{str(e)}")

    try:
        img_base64 = compress_image(image, max_size=1600, quality=85)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"圖片壓縮失敗：{str(e)}")

    # 初始化結果和時間
    result_1 = ""
    total_time = 0
    page_time=0
    # 如果啟用總計提取
    if enableTotalExtraction:
        total_start_time = time.time()
        # 使用自定義提示詞或默認提示詞
        result_1 = ocr_total_only(img_base64, totalPrompt, temperature)
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
    
    # 記錄完整處理開始時間
    if enableFullExtraction:
        page_start_time = time.time()
        # 使用自定義提示詞或默認提示詞
        result_2 = ocr_page_with_rolm(img_base64, fullPrompt, temperature)
        page_end_time = time.time()
        page_time = page_end_time - page_start_time
    
    # 計算總處理時間
    combined_time = total_time + page_time
    print("--------------------------------------------\n")
    # 構建結果文本
    if enableTotalExtraction:
        result_text = result_1 + "\n\n" + result_2
        time_info = f"\n計算價格時間: {total_time:.2f}秒\n\n計算整張時間: {page_time:.2f}秒\n\n總時間: {combined_time:.2f}秒"
    else:
        result_text = result_2
        time_info = f"\n計算整張時間: {page_time:.2f}秒"
    
    return JSONResponse(content={
        "ocr_result": result_text + time_info,
        "processing_times": {
            "total_price_time": total_time if enableTotalExtraction else None,
            "full_page_time": page_time,
            "combined_time": combined_time
        }
    })
# 支援命令列執行
def main():
    uvicorn.run(app, host="0.0.0.0", port=9005)

if __name__ == "__main__":
    main()