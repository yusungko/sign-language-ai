# 🤟 Sign Language AI Recognition System

本專案是一套端到端的 AI 手語辨識系統，整合影片資料處理、關鍵點擷取、3D CNN + ResNeXt 模型訓練與推論。應用於手語詞彙的分類與視訊理解任務。

---

## 📌 系統簡介

此系統包含以下核心模組：

1. **影片資料前處理**
   - 去除黑畫面 (`delBlack.ipynb`)
   - 使用 MediaPipe 擷取手部／臉部／身體 landmark (`Mediapipe.ipynb`)

2. **資料集整理**
   - 根據 JSON 或影片檔名進行分類與分割 (`Signbert.ipynb`)
   - 生成訓練集 / 驗證集 / 測試集

3. **模型訓練**
   - 使用 TensorFlow + Keras 建立 `3D CNN + ResNeXt` 架構 (`3DCNN.ipynb`)
   - 可擴充為句子辨識模型（支援 BERT 架構，訓練中）

---

## 🧠 模型架構

- Input：預處理後影片（frame-wise tensor 或 landmark 序列）
- 模型：
  - 空間特徵：採用 **ResNeXt** 模型作為 backbone
  - 時間特徵：使用 **3D Convolution** 探索 temporal pattern
- Output：多類別手語詞分類

架構圖（可替換為你產出的 demo.png）：

Video (mp4)
│  
├─▶ delBlack → Mediapipe Landmark 抽取  
├─▶ Signbert.ipynb 分類整理資料  
└─▶ 3DCNN (ResNeXt backbone)  
↓  
手語分類結果

---

## 📁 專案結構

```bash
sign-language-ai/
├── notebooks/                 # Jupyter Notebook 原始碼
│   ├── delBlack.ipynb
│   ├── Mediapipe.ipynb
│   ├── Signbert.ipynb
│   ├── 3DCNN.ipynb
│   └── ...
├── data/                      # 樣本 JSON 檔、少量測試影片
│   └── hospital.json
├── produced_videos/          # landmark/姿勢可視化影片
├── requirements.txt
└── README.md
```

## 🛠 安裝與執行方式

### 1️⃣ 安裝套件

pip install -r requirements.txt
內容包含：

tensorflow  
keras  
opencv-python  
mediapipe  
einops  
numpy  
matplotlib  
seaborn  
imageio  
tqdm

### 2️⃣ 執行 Notebook
依照順序執行以下 .ipynb：

delBlack.ipynb

Mediapipe.ipynb

Signbert.ipynb

3DCNN.ipynb

可另建 predict.ipynb 作為模型推論測試。

## 🔍 適用場景
手語識別／助聽應用

## 📫 聯絡方式
GitHub: [@yusungko](https://github.com/yusungko)

Email: st950157@gmail.com
