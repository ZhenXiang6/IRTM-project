import requests
from datetime import datetime, timedelta

# 設置下載起始日期和結束日期
start_date = datetime(2019, 12, 1)  # 起始日期
end_date = datetime(2019, 12, 3)    # 結束日期

# 基本下載 URL
base_url = "http://data.gdeltproject.org/gdeltv2/"

# 創建下載函數
def download_gkg(date):
    # 格式化日期為 YYYYMMDD 格式
    date_str = date.strftime("%Y%m%d")
    # 完整文件名
    filename = f"{date_str}.gkg.csv.zip"
    # 文件 URL
    file_url = base_url + filename
    print(f"正在下載: {file_url}")
    
    # 發送 HTTP 請求
    response = requests.get(file_url, stream=True)
    
    # 如果請求成功
    if response.status_code == 200:
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"下載完成: {filename}")
    else:
        print(f"無法下載: {file_url} (狀態碼: {response.status_code})")

# 循環下載指定日期範圍內的文件
current_date = start_date
while current_date <= end_date:
    download_gkg(current_date)
    current_date += timedelta(days=1)
