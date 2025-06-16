import pandas as pd

df = pd.read_csv("recalib_data_SH.csv")
df["dx"] = -df["dx"]
df.to_csv("recalib_data_SH_flipx.csv", index=False)
