import os
import re
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from tqdm import tqdm
import ta  # 技術指標

# ==========================
#  A. 讀取與前處理 (新聞部分)
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
        date_str = match.group(1)
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
    df_cnbc     = load_cnbc(cnbc_path)
    df_guardian = load_guardian(guardian_path)
    df_reuters  = load_reuters(reuters_path)
    df_news = pd.concat([df_cnbc, df_guardian, df_reuters], ignore_index=True)
    df_news.dropna(subset=['date','headline'], inplace=True)
    df_news.sort_values('date', inplace=True)
    print(f"合併後新聞數量: {len(df_news)}")
    return df_news

# ==========================
#  B. FinBERT 批次情緒分析
# ==========================

def setup_finbert_pipeline():
    """初始化 FinBERT pipeline"""
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    device = 0 if torch.cuda.is_available() else -1
    finbert_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return finbert_pipeline

def batch_sentiment(text_list, pipeline_fn, batch_size=16, max_length=128):
    """
    手動分批，並使用 tqdm 進度條顯示處理進度。
    除非 text_list 非常龐大，否則也可以一次性傳入 pipeline，
    但這裡透過手動分批可精確控制 batch_size。
    """
    results = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="Sentiment Analysis Batches"):
        batch_texts = text_list[i : i+batch_size]
        try:
            batch_out = pipeline_fn(
                batch_texts,
                truncation=True,
                max_length=max_length
            )
            for out in batch_out:
                label = out['label'].lower()
                score = out['score']
                if label == 'positive':
                    results.append(+score)
                elif label == 'negative':
                    results.append(-score)
                else:
                    results.append(0.0)
        except Exception as e:
            print(f"批次 {i//batch_size + 1} 解析錯誤: {e}")
            results.extend([0.0]*len(batch_texts))
    return results

def compute_daily_sentiment(df_news: pd.DataFrame, sentiment_pipeline, batch_size=16, max_length=128) -> pd.DataFrame:
    """
    對 headline & article_content 做批次情緒分析，顯示 tqdm 進度條，並彙整成每日平均 (mean_headline_sent, mean_content_sent)。
    """
    print("開始批次情緒分析 (FinBERT) ...")

    # headline
    headlines = df_news['headline'].tolist()
    print("-> 分析 Headline")
    headline_scores = batch_sentiment(headlines, sentiment_pipeline, batch_size=batch_size, max_length=max_length)

    # content
    contents = df_news['article_content'].tolist()
    print("-> 分析 Content")
    content_scores = batch_sentiment(contents, sentiment_pipeline, batch_size=batch_size, max_length=max_length)

    df_news['headline_sent'] = headline_scores
    df_news['content_sent']  = content_scores

    print("-> 彙整當日情緒...")
    daily_sentiment = (
        df_news
        .groupby('date')
        .agg({'headline_sent':'mean','content_sent':'mean'})
        .reset_index()
        .rename(columns={
            'headline_sent': 'mean_headline_sent',
            'content_sent': 'mean_content_sent'
        })
    )
    print("情緒分析完成。若出現負值，可能代表負面新聞居多。")
    return daily_sentiment

# ==========================
#  C. S&P500 + 技術指標
# ==========================

def load_sp500(sp_csv_path: str) -> pd.DataFrame:
    df_sp = pd.read_csv(sp_csv_path, on_bad_lines='skip', engine='python')
    print(f"S&P500 原始欄位: {df_sp.columns.tolist()}")
    df_sp.rename(columns={'Date':'date','S&P500':'sp_close'}, inplace=True)
    df_sp['date'] = pd.to_datetime(df_sp['date']).dt.strftime('%Y-%m-%d')
    df_sp.sort_values('date', inplace=True)
    print(f"S&P500 行數: {len(df_sp)}")
    return df_sp

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    temp_df['sp_close'] = pd.to_numeric(temp_df['sp_close'], errors='coerce')
    temp_df['Open'] = temp_df['sp_close']
    temp_df['High'] = temp_df['sp_close']
    temp_df['Low']  = temp_df['sp_close']
    temp_df['Volume'] = 1

    temp_df['SMA_5'] = ta.trend.sma_indicator(temp_df['sp_close'], window=5)
    temp_df['SMA_10'] = ta.trend.sma_indicator(temp_df['sp_close'], window=10)
    temp_df['RSI_14'] = ta.momentum.rsi(temp_df['sp_close'], window=14)
    macd = ta.trend.MACD(temp_df['sp_close'], window_slow=26, window_fast=12, window_sign=9)
    temp_df['MACD']        = macd.macd()
    temp_df['MACD_signal'] = macd.macd_signal()
    temp_df['MACD_diff']   = macd.macd_diff()

    temp_df.drop(['Open','High','Low','Volume'], axis=1, inplace=True)
    return temp_df

# ==========================
#  D. 建立標籤
# ==========================

def create_labels_for_prediction(df: pd.DataFrame, mode='classification'):
    df['sp_close_next'] = df['sp_close'].shift(-1)
    df.dropna(subset=['sp_close_next'], inplace=True)
    if mode == 'classification':
        df['target'] = (df['sp_close_next'] > df['sp_close']).astype(int)
    else:
        df['target'] = df['sp_close_next']
    return df

# ==========================
#  E. 模型定義
# ==========================

def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf

def train_gradient_boosting(X_train, y_train):
    clf = GradientBoostingClassifier(n_estimators=150, random_state=42, learning_rate=0.1, max_depth=5)
    clf.fit(X_train, y_train)
    return clf

def get_stacking_model():
    rf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    gb = GradientBoostingClassifier(n_estimators=150, random_state=42)
    stacking_clf = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    return stacking_clf

class LSTMModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, num_classes=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_lstm_classification(X_train, y_train, input_size=9, epochs=15, batch_size=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
    return model

def predict_lstm(model, X_test):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
    _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()

# ==========================
#  F. TimeSeriesSplit & 交叉驗證
# ==========================

def timeseries_cv_and_train(X, y, model_fn, n_splits=5, is_lstm=False):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    acc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 如果是 LSTM，需要 reshape => (samples, 1, features=9)
        if is_lstm:
            X_train_3d = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_val_3d   = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            model, pred_fn = model_fn(X_train_3d, y_train)
            y_pred = pred_fn(X_val_3d)
        else:
            model, pred_fn = model_fn(X_train, y_train)
            y_pred = pred_fn(X_val)

        acc = accuracy_score(y_val, y_pred)
        acc_scores.append(acc)
        print(f"Fold {fold_idx+1}/{n_splits}, ACC = {acc:.4f}")

    avg_acc = np.mean(acc_scores)
    print(f"=== TimeSeriesSplit Average ACC: {avg_acc:.4f} ===")
    return avg_acc

# ==========================
#  主流程
# ==========================

def main():
    # 讀取檔案
    cnbc_csv     = r"C:\Users\morri\Desktop\IRTM-project\test\cnbc_headlines.csv"
    guardian_csv = r"C:\Users\morri\Desktop\IRTM-project\test\guardian_headlines.csv"
    reuters_csv  = r"C:\Users\morri\Desktop\IRTM-project\test\reuters_headlines.csv"
    sp500_csv    = r"C:\Users\morri\Desktop\IRTM-project\test\sp500_index.csv"

    # 1) 讀取 & 合併新聞
    print("讀取並合併新聞資料...")
    df_news = merge_news(cnbc_csv, guardian_csv, reuters_csv)
    print(f"[News] total: {len(df_news)} rows from 3 sources.\n")

    # 2) FinBERT 批次情緒分析
    print("\n初始化 FinBERT pipeline...")
    finbert_pipe = setup_finbert_pipeline()

    print("\n計算每日情緒分數 (含 tqdm 進度條)...")
    # 使用 batch_size=32，你可依GPU記憶體自行調整
    daily_sentiment_df = compute_daily_sentiment(df_news, finbert_pipe, batch_size=32, max_length=128)
    daily_sentiment_df.to_csv("daily_sentiment.csv", index=False)
    print("[Output] daily_sentiment.csv 已輸出.\n")

    # 3) 讀取 S&P500 並加入技術指標
    print("\n讀取 S&P500 資料並加入技術指標...")
    df_sp = load_sp500(sp500_csv)
    df_sp_tech = add_technical_indicators(df_sp)
    print(f"[SP500] after tech indicators: {df_sp_tech.shape}\n")

    # 4) 合併 & 處理 NaN
    print("合併情緒分數與 S&P500 資料...")
    df_merged = pd.merge(daily_sentiment_df, df_sp_tech, on='date', how='inner')
    df_merged.sort_values('date', inplace=True)
    df_merged.dropna(inplace=True)
    print(f"[Merged] total: {len(df_merged)} rows.\n")

    # 5) 建立標籤 (隔日漲跌)
    print("建立標籤 (隔日漲跌)...")
    df_merged = create_labels_for_prediction(df_merged, mode='classification')
    print(f"[Merged+Target] total: {len(df_merged)} rows with target.\n")

    # 6) 特徵 & 標準化
    features = [
        'mean_headline_sent',
        'mean_content_sent',
        'sp_close',
        'SMA_5',
        'SMA_10',
        'RSI_14',
        'MACD',
        'MACD_signal',
        'MACD_diff'
    ]
    df_merged[features] = df_merged[features].astype(float).fillna(0)
    X_full = df_merged[features].values
    y_full = df_merged['target'].values

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    print("[Info] Features scaled.\n")

    # 7) TimeSeriesSplit 交叉驗證，比較三種模型
    print("=== TimeSeriesSplit: Random Forest ===")
    def model_fn_rf(X, y):
        model = train_random_forest(X, y)
        return model, lambda X_val: model.predict(X_val)

    avg_acc_rf = timeseries_cv_and_train(X_full_scaled, y_full, model_fn_rf, n_splits=5, is_lstm=False)

    print("\n=== TimeSeriesSplit: LSTM ===")
    def model_fn_lstm(X, y):
        # X shape = (samples, 1, 9)
        model = train_lstm_classification(X, y, input_size=X.shape[2], epochs=15, batch_size=32)
        return model, lambda X_val: predict_lstm(model, X_val)

    avg_acc_lstm = timeseries_cv_and_train(
        X_full_scaled, y_full, 
        model_fn=lambda X, y: model_fn_lstm(X.reshape((X.shape[0],1,X.shape[1])), y),
        n_splits=5,
        is_lstm=True
    )

    print("\n=== TimeSeriesSplit: Stacking ===")
    def model_fn_stack(X, y):
        stacking_clf = get_stacking_model()
        stacking_clf.fit(X, y)
        return stacking_clf, lambda X_val: stacking_clf.predict(X_val)

    avg_acc_stack = timeseries_cv_and_train(X_full_scaled, y_full, model_fn_stack, n_splits=5, is_lstm=False)

    # 選擇最佳模型
    best_acc = max([avg_acc_rf, avg_acc_lstm, avg_acc_stack])
    if best_acc == avg_acc_rf:
        best_model_name = "RandomForest"
    elif best_acc == avg_acc_lstm:
        best_model_name = "LSTM"
    else:
        best_model_name = "Stacking"

    print(f"\n[Best Model] {best_model_name}, ACC={best_acc:.4f}\n")

    # 8) 最終模型訓練
    print("=== 訓練最終模型 ===")
    if best_model_name=="RandomForest":
        final_model, final_predict = model_fn_rf(X_full_scaled, y_full)
        is_lstm_mode = False
    elif best_model_name=="LSTM":
        X_full_3d = X_full_scaled.reshape(X_full_scaled.shape[0], 1, X_full_scaled.shape[1])
        final_model, final_predict = model_fn_lstm(X_full_3d, y_full)
        is_lstm_mode = True
    else:
        final_model, final_predict = model_fn_stack(X_full_scaled, y_full)
        is_lstm_mode = False

    if is_lstm_mode:
        X_test_final = X_full_scaled.reshape(X_full_scaled.shape[0], 1, X_full_scaled.shape[1])
        y_pred_final = final_predict(X_test_final)
    else:
        y_pred_final = final_predict(X_full_scaled)

    acc_final = accuracy_score(y_full, y_pred_final)
    print(f"[Final Model: {best_model_name}] Accuracy on full data: {acc_final:.4f}")
    print(classification_report(y_full, y_pred_final))

    # 輸出 merged_data.csv
    df_merged.to_csv("merged_data.csv", index=False)
    print("merged_data.csv 已輸出。")

    # 視覺化
    plt.figure(figsize=(12,5))
    plt.plot(range(len(y_full)), y_full, label='True (0=跌,1=漲)', marker='o')
    plt.plot(range(len(y_full)), y_pred_final, label='Predicted', linestyle='--')
    plt.title(f"Final Model ({best_model_name}) on Full Data")
    plt.xlabel("Index (Time Order)")
    plt.ylabel("Label")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
