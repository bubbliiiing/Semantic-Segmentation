import os

jpgs = os.listdir("./dataset2/jpg")
pngs = os.listdir("./dataset2/png")

with open("./dataset2/train_data.txt","w") as f:
    for jpg in jpgs:
        png = jpg.replace("jpg","png")
        # 判断jpg是否存在对应的png
        if png in pngs:
            f.write(jpg+";"+png+"\n")