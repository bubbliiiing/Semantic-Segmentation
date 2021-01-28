import os

import numpy as np
from PIL import Image


def main():
    # 读取原文件夹
    count = os.listdir("./before/") 
    for i in range(0, len(count)):
        # 如果里的文件以jpg结尾
        # 则寻找它对应的png
        if count[i].endswith("jpg"):
            path = os.path.join("./before", count[i])
            img = Image.open(path)
            img.save(os.path.join("./jpg", count[i]))
            
            # 找到对应的png
            path = "./output/" + count[i].split(".")[0] + "_json/label.png"
            img = Image.open(path)

            # 找到全局的类
            class_txt = open("./before/class_name.txt","r")
            class_name = class_txt.read().splitlines()
            # ["bk","cat","dog"]
            # 打开json文件里面存在的类，称其为局部类
            with open("./output/" + count[i].split(".")[0] + "_json/label_names.txt","r") as f:
                names = f.read().splitlines()
                # ["bk","dog"]
                new = Image.new("RGB",[np.shape(img)[1],np.shape(img)[0]])
                for name in names:
                    # index_json是json文件里存在的类，局部类
                    index_json = names.index(name)
                    # index_all是全局的类
                    index_all = class_name.index(name)
                    # 将局部类转换成为全局类
                    
                    new = new + np.expand_dims(index_all*(np.array(img) == index_json),-1)

            new = Image.fromarray(np.uint8(new))        
            new.save(os.path.join("./png", count[i].replace("jpg","png")))
            print(np.max(new),np.min(new))

if __name__ == '__main__':
    main()
