import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================
#  1) 讀取 & 前處理
# ==========================

def clean_text(txt: str) -> str:
    """簡單文本清理，去除多餘空白和雜訊。"""
    if not isinstance(txt, str):
        return ""
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

def parse_cnbc_time(t):
    """解析 CNBC 的時間格式: '7:51  PM ET Fri, 17 July 2020'"""
    if not isinstance(t, str):
        return None
    match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', t)
    if match:
        date_str = match.group(1)  # e.g., '17 July 2020'
        try:
            dt = datetime.strptime(date_str, '%d %B %Y')
            return dt.strftime('%Y-%m-%d')
        except:
            return None
    return None

def parse_guardian_time(t):
    """解析 Guardian 的時間格式: '18-Jul-20'"""
    if not isinstance(t, str):
        return None
    try:
        dt = datetime.strptime(t.strip(), '%d-%b-%y')
        return dt.strftime('%Y-%m-%d')
    except:
        return None

def parse_reuters_time(t):
    """解析 Reuters 的時間格式: 'Jul 18 2020'"""
    if not isinstance(t, str):
        return None
    try:
        dt = datetime.strptime(t.strip(), '%b %d %Y')
        return dt.strftime('%Y-%m-%d')
    except:
        return None

def load_cnbc(csv_path: str) -> pd.DataFrame:
    """讀取 CNBC 的 CSV 檔案"""
    df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
    print(f"CNBC 原始欄位: {df.columns.tolist()}")
    # 確認欄位名稱是否包含 'Headlines', 'Time', 'Description'
    expected_cols = ['Headlines', 'Time', 'Description']
    for col in expected_cols:
        if col not in df.columns:
            print(f"警告: CNBC CSV 缺少欄位 '{col}'")
    df['headline'] = df['Headlines'].fillna("").apply(clean_text)
    df['article_content'] = df['Description'].fillna("").apply(clean_text)
    df['date'] = df['Time'].apply(parse_cnbc_time)
    df = df[['date', 'headline', 'article_content']].dropna(subset=['date', 'headline'])
    df['source'] = 'CNBC'
    return df

def load_guardian(csv_path: str) -> pd.DataFrame:
    """讀取 Guardian 的 CSV 檔案"""
    df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
    print(f"Guardian 原始欄位: {df.columns.tolist()}")
    # 確認欄位名稱是否包含 'Time', 'Headlines'
    expected_cols = ['Time', 'Headlines']
    for col in expected_cols:
        if col not in df.columns:
            print(f"警告: Guardian CSV 缺少欄位 '{col}'")
    df['headline'] = df['Headlines'].fillna("").apply(clean_text)
    df['article_content'] = ""  # Guardian 無 Description
    df['date'] = df['Time'].apply(parse_guardian_time)
    df = df[['date', 'headline', 'article_content']].dropna(subset=['date', 'headline'])
    df['source'] = 'Guardian'
    return df

def load_reuters(csv_path: str) -> pd.DataFrame:
    """讀取 Reuters 的 CSV 檔案"""
    df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
    print(f"Reuters 原始欄位: {df.columns.tolist()}")
    # 確認欄位名稱是否包含 'Headlines', 'Time', 'Description'
    expected_cols = ['Headlines', 'Time', 'Description']
    for col in expected_cols:
        if col not in df.columns:
            print(f"警告: Reuters CSV 缺少欄位 '{col}'")
    df['headline'] = df['Headlines'].fillna("").apply(clean_text)
    df['article_content'] = df['Description'].fillna("").apply(clean_text)
    df['date'] = df['Time'].apply(parse_reuters_time)
    df = df[['date', 'headline', 'article_content']].dropna(subset=['date', 'headline'])
    df['source'] = 'Reuters'
    return df

def merge_news(cnbc_path, guardian_path, reuters_path) -> pd.DataFrame:
    """合併三家新聞資料"""
    df_cnbc = load_cnbc(cnbc_path)
    df_guardian = load_guardian(guardian_path)
    df_reuters = load_reuters(reuters_path)
    df_news = pd.concat([df_cnbc, df_guardian, df_reuters], ignore_index=True)
    df_news.dropna(subset=['date', 'headline'], inplace=True)
    df_news.sort_values('date', inplace=True)
    print(f"合併後新聞數量: {len(df_news)}")
    return df_news

# ==========================
#  2) FinBERT 情緒分析
# ==========================

def setup_finbert_pipeline():
    """初始化 FinBERT pipeline"""
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    return sentiment_pipeline

def get_finbert_sentiment(txt: str, sentiment_pipeline) -> float:
    """
    使用 FinBERT pipeline 得到新聞文本的情緒分數:
    - 正面: +score
    - 負面: -score
    - 中立:  0
    """
    if not isinstance(txt, str) or txt.strip() == '':
        return 0.0
    try:
        result = sentiment_pipeline(txt[:512])[0]  # 只取前 512 tokens
        label = result['label'].lower()
        score = result['score']
        if label == 'positive':
            return +score
        elif label == 'negative':
            return -score
        else:
            return 0.0
    except Exception as e:
        print(f"情緒分析錯誤: {e}")
        return 0.0

def compute_daily_sentiment(df_news: pd.DataFrame, sentiment_pipeline) -> pd.DataFrame:
    """
    對每則新聞的 headline 和 article_content 分別做情緒分析，再求平均。
    回傳: daily_sentiment_df，包含 date, mean_headline_sent, mean_content_sent
    """
    print("=> 開始計算 FinBERT 情緒 ...")
    sentiments_headline = []
    sentiments_content = []
    for i in tqdm(range(len(df_news)), desc="Processing News"):
        row = df_news.iloc[i]
        headline_score = get_finbert_sentiment(row['headline'], sentiment_pipeline)
        content_score = get_finbert_sentiment(row['article_content'], sentiment_pipeline)
        sentiments_headline.append(headline_score)
        sentiments_content.append(content_score)
    df_news['headline_sent'] = sentiments_headline
    df_news['content_sent'] = sentiments_content

    print("=> 彙整當日情緒 ...")
    daily_sentiment = (
        df_news
        .groupby('date')
        .agg({'headline_sent': 'mean', 'content_sent': 'mean'})
        .reset_index()
        .rename(columns={
            'headline_sent': 'mean_headline_sent',
            'content_sent': 'mean_content_sent'
        })
    )
    print(f"每日情緒表大小: {daily_sentiment.shape}")
    return daily_sentiment

# ==========================
#  3) 建立特徵 & 標籤
# ==========================

def load_sp500(sp_csv_path: str) -> pd.DataFrame:
    """
    讀取 S&P500 資料:
      Date,S&P500
      2014-12-22,2078.54
    轉成 date=YYYY-MM-DD, sp_close=數值
    """
    df_sp = pd.read_csv(sp_csv_path, on_bad_lines='skip', engine='python')
    print(f"S&P500 原始欄位: {df_sp.columns.tolist()}")
    df_sp.rename(columns={'Date': 'date', 'S&P500': 'sp_close'}, inplace=True)
    df_sp['date'] = pd.to_datetime(df_sp['date']).dt.strftime('%Y-%m-%d')
    df_sp.sort_values('date', inplace=True)
    print(f"S&P500 行數: {len(df_sp)}")
    return df_sp

def merge_with_sp(daily_sentiment_df: pd.DataFrame, sp_df: pd.DataFrame):
    """
    合併每日情緒與 S&P 收盤。回傳 df_merged。
    只保留有重疊的日期 (inner join)。
    """
    df_merged = pd.merge(daily_sentiment_df, sp_df, on='date', how='inner')
    df_merged.sort_values('date', inplace=True)
    df_merged.dropna(subset=['mean_headline_sent', 'mean_content_sent', 'sp_close'], inplace=True)
    print(f"合併後資料大小: {df_merged.shape}")
    return df_merged

def create_labels_for_prediction(df: pd.DataFrame, mode='classification'):
    """
    mode='classification': 建立二元標籤(隔日上漲=1、下跌=0)
    mode='regression': 直接預測隔日收盤價
    """
    df['sp_close_next'] = df['sp_close'].shift(-1)  # 隔日收盤
    df = df.dropna(subset=['sp_close_next'])

    if mode == 'classification':
        df['target'] = (df['sp_close_next'] > df['sp_close']).astype(int)
    elif mode == 'regression':
        df['target'] = df['sp_close_next']  # 直接預測數值

    print(f"建立標籤後資料大小: {df.shape}")
    return df

# ==========================
#  4) 建立多模型 (RandomForest, LSTM, Stacking/Boosting)
# ==========================

def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, num_classes=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # For classification

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]  # 取最後一個 time step
        out = self.fc(out)
        return out

def train_lstm_classification(X_train_ts, y_train_ts, input_size=2, epochs=20):
    """
    簡易 LSTM 訓練示範：針對 time-series classification。
    X_train_ts shape: (samples, seq_len, feature_dim)
    y_train_ts shape: (samples,)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMModel(input_size=input_size, hidden_size=32, num_layers=1, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X_train_ts, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train_ts, dtype=torch.long).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model

def predict_lstm(model, X_test_ts):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_tensor = torch.tensor(X_test_ts, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
    _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()

def get_stacking_model():
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    stacking_clf = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    return stacking_clf

# ==========================
#  5) 評估
# ==========================

def evaluate_classification(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"MSE : {mse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R^2 : {r2:.4f}")

# ==========================
#  主流程
# ==========================

def main():
    # 確保使用原始字串 (r"路徑") 或將 '\' 換成 '\\'
    cnbc_csv     = r"C:\Users\morri\Desktop\IRTM-project\test\cnbc_headlines.csv"
    guardian_csv = r"C:\Users\morri\Desktop\IRTM-project\test\guardian_headlines.csv"
    reuters_csv  = r"C:\Users\morri\Desktop\IRTM-project\test\reuters_headlines.csv"
    sp500_csv    = r"C:\Users\morri\Desktop\IRTM-project\test\sp500_index.csv"

    # 1) 讀取 & 合併新聞
    df_news = merge_news(cnbc_csv, guardian_csv, reuters_csv)

    # 2) 初始化 FinBERT，計算情緒
    finbert_pipeline = setup_finbert_pipeline()

    # 3) 計算每日情緒
    daily_sentiment_df = compute_daily_sentiment(df_news, finbert_pipeline)

    # 4) 讀取 S&P500 資料
    df_sp = load_sp500(sp500_csv)

    # 5) 合併情緒 & S&P
    df_merged = merge_with_sp(daily_sentiment_df, df_sp)

    # 6) 建立標籤 (以 classification 為例：隔日漲/跌)
    df_merged = create_labels_for_prediction(df_merged, mode='classification')

    # 7) 定義特徵與目標
    feature_cols = ['mean_headline_sent', 'mean_content_sent', 'sp_close']
    X = df_merged[feature_cols].values
    y = df_merged['target'].values

    # 8) 切分訓練/測試 (時間序列，不能 shuffle)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 9) 標準化特徵（可選）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # (A) Random Forest 分類
    print("\n=== Random Forest ===")
    rf_clf = train_random_forest(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    evaluate_classification(y_test, y_pred_rf)

    # (B) LSTM 分類
    print("\n=== LSTM ===")
    # LSTM 需要三維輸入 [samples, seq_len, features]
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    lstm_model   = train_lstm_classification(X_train_lstm, y_train, input_size=X_train.shape[1], epochs=20)
    y_pred_lstm  = predict_lstm(lstm_model, X_test_lstm)
    evaluate_classification(y_test, y_pred_lstm)

    # (C) Stacking (RandomForest + GradientBoosting)
    print("\n=== Stacking (RF + GB) ===")
    stack_clf = get_stacking_model()
    stack_clf.fit(X_train, y_train)
    y_pred_stack = stack_clf.predict(X_test)
    evaluate_classification(y_test, y_pred_stack)

    # 可視化
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, label='True (0=跌,1=漲)', marker='o')
    plt.plot(range(len(y_test)), y_pred_rf, label='RF Pred', linestyle='--')
    plt.plot(range(len(y_test)), y_pred_lstm, label='LSTM Pred', linestyle='--')
    plt.plot(range(len(y_test)), y_pred_stack, label='Stack Pred', linestyle='--')
    plt.title("Classification Results Comparison")
    plt.xlabel("Test Index")
    plt.ylabel("Label")
    plt.legend()
    plt.show()

    # （可選）輸出合併資料 CSV
    df_merged.to_csv("merged_data.csv", index=False)
    print("已輸出 merged_data.csv")

if __name__ == "__main__":
    main()
