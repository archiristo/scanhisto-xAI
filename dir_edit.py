import os
import pandas as pd
import shutil

csv_path = "archive/train_labels.csv"
image_dir = "archive/train/"
output_dir = "data/train/"

os.makedirs(output_dir + "cancer", exist_ok=True)
os.makedirs(output_dir + "normal", exist_ok=True)

df = pd.read_csv(csv_path)

df = df.sample(10000)  # 🔥 önce küçük subset

for _, row in df.iterrows():
    img_name = row['id'] + ".tif"
    label = row['label']

    src = os.path.join(image_dir, img_name)

    if label == 1:
        dst = os.path.join(output_dir, "cancer", img_name)
    else:
        dst = os.path.join(output_dir, "normal", img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)
