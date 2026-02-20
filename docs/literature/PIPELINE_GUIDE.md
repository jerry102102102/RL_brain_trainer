# Pipeline Reading Guide (L1/L2/L3)

## 你現在的研究假設
- L1 之後會換成 VLM（目前用 heuristic 代替）
- L2 必須是「局部路徑規劃器」，不是單純速度策略器
- L3 是 deterministic follower，根據回饋穩定追蹤

## 按環節看文獻

### L1：高層語意/全局規劃（VLM 假設層）
先讀：
1. `160f9154-ea6e-4e4c-8952-bd4fe027235a.pdf`（RoBridge）

要回答：
- L1 輸出應是「語意+約束」還是「稀疏 waypoint」？
- L1 與 L2 的責任切線在哪裡？

### L2：局部規劃 + 記憶
先讀：
1. `dee63f08-7672-497a-8d7c-3840e9166c60.pdf`（Memory-based constraints）
2. `d0072133-43f3-4a58-bf54-1f57e7511e1a.pdf`（optimization/control，待精讀）

要回答：
- L2 的 memory 何時應該介入？
- 介入形式是 path candidate ranking、warm-start，還是直接改控制？

### L3：穩定追蹤 + 魯棒執行
先讀：
1. `d2fa8f71-741b-49f4-b3f3-4f0c3080ccf4.pdf`（robotics empirical evidence，待精讀）

要回答：
- 追蹤器與環境擾動下的穩定性邊界如何量測？
- 什麼指標能早於 success_rate 反映崩潰前兆？

## 每輪研究輸出格式（固定）
1) 本輪改動對齊哪篇文獻
2) 改了哪個環節（L1/L2/L3）
3) 驗證指標（先可行、再最優）
4) 是否通過 Lv5 challenge gate
