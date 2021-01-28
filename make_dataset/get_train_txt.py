import os

jpgs = os.listdir("./jpg")
pngs = os.listdir("./png")

with open("train.txt","w") as f:
    for jpg in jpgs:
        png = jpg.replace("jpg","png")
        # 判断jpg是否存在对应的png
        if png in pngs:
            f.write(jpg+";"+png+"\n")
