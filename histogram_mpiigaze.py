import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mpiigaze/mpiigaze_labels_center_px.csv")
plt.hist(df["dx"], bins=50)
plt.title("X축 레이블 분포")
plt.show()
