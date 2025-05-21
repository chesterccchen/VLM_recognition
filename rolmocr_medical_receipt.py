# HOST YOUR OPENAI COMPATIBLE API WITH THE FOLLOWING COMMAND in VLLM:
# export VLLM_USE_V1=1
# vllm serve reducto/RolmOCR 

from openai import OpenAI
import base64
import time
import threading
import argparse
from PIL import Image
import io

client = OpenAI(api_key="123", base_url="http://localhost:8000/v1")

model = "reducto/RolmOCR"

def compress_image(image_path, max_size=1700, quality=85):
    """
    壓縮圖片，保持原始寬高比，不強制調整為正方形。
    :param image_path: 圖片路徑
    :param max_size: 最大邊長（單邊），預設為 1600
    :param quality: JPEG 壓縮品質，預設為 85
    :return: base64 編碼的圖片數據
    """
    # 打開圖片
    img = Image.open(image_path)
    
    # 檢查圖片原始尺寸
    original_width, original_height = img.size
    #print(f"原始圖片尺寸：{original_width}x{original_height}")
    
    # 計算原始寬高比
    aspect_ratio = original_width / original_height
    #print(f"原始寬高比：{aspect_ratio:.2f}")
    
    # 如果圖片最大邊長小於 max_size，則不需要壓縮
    if max(original_width, original_height) >= max_size:
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
    img_base64 = compress_image(image_path, max_size=1700, quality=85)
    return img_base64

def VLM_ocr(img_base64):
    img_data = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_data))
    width, height = img.size
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
                        #"text": """完整的印出這張圖片上的所有內容，包含標題，病患資訊，表格，收費項目，金額，日期，序號，其他資訊等等，並保留表格資訊，結束後印出"END_OF_OUTPUT"  """
                        "text": """完整的從圖片的最上面到最下面印出這張圖片上的所有內容和文字，並保留表格資訊，結束後印出"END_OF_OUTPUT"  """
                        
                    },
                ],
            }
        ],
        stream=True,
        temperature=0.2,
        #presence_penalty=1,
        max_tokens=4000,
        stop=["END_OF_OUTPUT"],
        #frequency_penalty=1.0,
    )
    # 計算推理時間
    full_text = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
          partial = chunk.choices[0].delta.content
          print(partial, end="", flush=True)
          full_text += partial
   # print("\n📥 全文結束生成。")

    # 返回結果
    #return response.choices[0].message.content
    return full_text

def main():
    # 設定命令列參數解析
    parser = argparse.ArgumentParser(description='OCR processing with RolmOCR')
    parser.add_argument('--image', 
                        type=str, 
                        default="/home/chester/rolmocr/tw_einvoice/1005.jpg",
                        help='Path to the test image file')
    
    # 解析參數
    args = parser.parse_args()
    # 使用解析後的路徑
    test_img_path = args.image
    img_base64 = encode_image(test_img_path)

    print("\n📥 全文生成結果:\n")

    start_time = time.time()
    result_page = VLM_ocr(img_base64)
    end_time = time.time()
    print(f"計算整張時間: {end_time - start_time}")

if __name__ == "__main__":
    main()