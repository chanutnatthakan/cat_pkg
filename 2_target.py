import pandas as pd

df = pd.read_csv("clean_cat.csv")

tg_1 = ['opt1_buy', 'opt2_buy', 'opt3_buy', 'opt4_buy', 'opt5_buy', 'opt6_buy', 'opt7_buy', 'opt8_buy', 'opt9_buy', 'opt10_buy']

buy_top = df[tg_1].idxmax(axis=1) #เช็คทีละคน
df['target_buy'] = buy_top.str.replace('opt', '').str.replace('_buy', '').astype(int) #ดึงแค่ตัวเลขออกมา 

print("---- คนที่เลือกซื้อเพราะเห็นผลิตภัณฑ์ ----")
print(df['target_buy'].value_counts())

df.to_csv('clean_cat.csv', index=False)