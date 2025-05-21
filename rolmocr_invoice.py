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

def timeout_handler():
    raise TimeoutError("推理時間超過 10 秒，已自動停止")

def ocr_total_only(img_base64):
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
                        #"text": "只要給我這張發票的總計價格就可以，需要將所有品項加總起來，注意:請特別注意是否有營業稅並加總，確認圖片的總價與計算結果是否相同且正確，如果價格不合理就再利用驗算方式驗算，輸出價格前必須輸出\"總計\"二字。輸出範例1. 總計:2019  合理。輸出範例2. 總計:2019000 不合理", },
                       "text": """請你擷取下列三個欄位：「銷售額合計」、「營業稅」、「總計」，**僅當發票上明確出現該詞語或數字時，才允許輸出數字**，否則請填「無」，給我純文字就好，禁止輸出表格。
注意:總計不一定等於銷售額合計+營業稅，因此你**絕對不能進行任何推理、計算或合邏輯推測**，只能逐字匹配文字欄位。
正確範例：
銷售額合計: 2354
營業稅: 100
總計: 2354
""" , },
                        #"text": "只要給我這張發票的總計價格就可以:輸出價格前必須輸出\"總計\"二字。輸出範例:總計:2019", },
                ],
            }
        ],
        temperature=0,
        presence_penalty=0.7,
        max_tokens=300,
        stop=["END_OF_OUTPUT"]
    )
    # 計算推理時間

    # 返回結果
    return response.choices[0].message.content

def ocr_page_with_rolm(img_base64):
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
                        "text": f"完整印出這張發票的所有資訊，從上到下擷取所有文本，包括標題、發票號碼、日期、表格、賣方/買方資訊、備註、總計和商業印章上的資訊。發票可能包含印刷文字和手寫文字，並且會有部分英文。優先處理中文文字，除非英文是關鍵字段的一部分。請特別注意捕捉最上方的標題和最下方的文字，因為這些部分對發票的有效性至關重要。完成後，請寫下 'END_OF_OUTPUT'，請勿一直重複輸出相同資訊。", },
                        #"text": "輸出這張圖片的所有資訊，一個字都不能漏", },
                
                ],
            }
        ],
        stream=True,
        temperature=0.7,
        presence_penalty=0.7,
        max_tokens=3000,
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

    calculate_page_time=0
    
    # 使用解析後的路徑
    test_img_path = args.image
    img_base64 = encode_image(test_img_path)

    start_time = time.time()
    result_total=ocr_total_only(img_base64)
    end_time = time.time()
    calculate_total_time=end_time - start_time

    print("\n📥 價格生成結果:\n"+result_total)

    print("\n📥 全文生成結果:\n")

    start_time = time.time()
    result_page = ocr_page_with_rolm(img_base64)
    end_time = time.time()
    calculate_page_time=end_time - start_time
    end_time = time.time()


    #print("result_total:\n"+result_total+"\nresult_page: \n"+result_page)
    #print("result_page:\n"+result_page+"\nresult_total: \n"+result_total)

    #print("result_total: \n"+result_total)
    print(f"計算價格時間: {calculate_total_time} 計算整張時間: {calculate_page_time}\n總時間: {calculate_total_time + calculate_page_time}")

if __name__ == "__main__":
    main()