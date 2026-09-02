"""Craigslist の `type` 列（ボディ形状）が取りうる値。

主環境（測定）と隔離環境 .venv-embed（埋め込み計算）の両方から読むので、
**依存を持たない小さな置き場**として切り出してある。両者で綴りがずれると
`PrecomputedEncoder` の引き当てに失敗するため、literal を二重に書かない。

綴りは**元データ（Kaggle の vehicles.csv）のまま**。`SUV` が大文字なのも
`mini-van` にハイフンが入るのも原典に合わせたもので、読み替えない。
並びは頻度順（sedan が 28.6%、bus が 0.1%）。
"""

TYPE_VALUES = ["sedan", "SUV", "pickup", "truck", "hatchback", "coupe",
               "other", "wagon", "van", "convertible", "mini-van",
               "offroad", "bus"]
