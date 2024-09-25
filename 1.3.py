#%%
import numpy as np
import time
import cv2
from PIL import Image
#from numba import jit
import pandas as pd
import math


def img2array(img_path, img_name):
    # 读取图片并获取尺�?
    # img_path = "C:/Users/acer/Documents/Python Scripts/"
    # img_name = "pic1.tif"
    # im = Image.open(img_path + img_name)
    im = cv2.imread(img_path + img_name)  # array
    # img_rgb2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im)
    im = im.convert("RGB")
    width, lengh = im.size
    # 裁剪图片
    # im = im.crop((8, 3, width, lengh-50))
    # 得到蓝色�?
    b, g, r = im.split()
    th = 140  # 阈�?
    b_np = np.array(r)
    g_np = np.array(g)
    r_np = np.array(b)
    # 图片双值化处理
    raw, column = b_np.shape
    img_b = Image.fromarray(b_np)  # 蓝色值图�?
    #img_b.save(img_path + "pic_b_origin_.png")  # 保存图片
    img_g = Image.fromarray(g_np)  # 绿色值图�?
    #img_g.save(img_path + "pic_g_origin_.png")  # 保存图片
    img_r = Image.fromarray(r_np)  # 红色值图�?
    #img_r.save(img_path + "pic_r_origin_.png")  # 保存图片

    for i in range(raw):
        for j in range(column):
            if b_np[i, j] <= th:
                b_np[i, j] = 0
            elif g_np[i, j] >= 127:
                b_np[i, j] = 254
            else:
                b_np[i, j] = 254

    img_b = Image.fromarray(b_np)  # 蓝色值图�?
    img_b.save(img_path + "pic_b_.png")  # 保存图片
    img = b_np.astype('int') / 254  # 得到0,1双值的img数组
    return img

#@jit
def calcR(x,y):
    # R1
    r1_up = 0
    k = 0
    item = [x,y]
    for i in range(col):
        for j in range(row):
            if i+item[0]<col and j+item[1]<row:
                r1_up += (img[i][j]-x_ave)*(img[i+item[0]][j+item[1]]-x_ave)
                k += 1
            if item[0]!=item[1]:
                if i+item[1]<col and j+item[0]<row:
                    r1_up += (img[i][j]-x_ave)*(img[i+item[1]][j+item[0]]-x_ave)
                    k += 1
            if item[0]!=0 :
                if i-item[1]>=0 and j+item[0]<row:
                    r1_up += (img[i][j]-x_ave)*(img[i-item[1]][j+item[0]]-x_ave)
                    k += 1
                if item[0]!=item[1]:
                    if i-item[0]>=0 and j+item[1]<row:
                        r1_up += (img[i][j]-x_ave)*(img[i-item[0]][j+item[1]]-x_ave)
                        k += 1
    #r1 = r1_up/r1_down/k
    rpingfang = item[0]**2+item[1]**2
    r = math.sqrt(rpingfang)
    return [k,r1_up,rpingfang,r]




img = np.array([[0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 0, 0]
                ])
img_name = "1.tif"
img_path = ""  # 现在照片默认的目录是与python脚本在同一目录�?
img = img2array(img_path, img_name)
col,row = img.shape
x_ave = np.average(img)
r1_down = 0
r = []
rhang = []
rlie = []
k = 0
for i in range(col):
    for j in range(row):
        r1_down += (img[i][j]-x_ave)**2/(col*row)

starttime=time.time()
for j in range(1,row):
    print("\r已完成{:.4f}%   时间：{:.4f}".format((j+1)*100/row,time.time()-starttime),end='') 
    for i in range(col):
        k += 1 
        if i>j and i<row:
            continue
        if k%2!=1:
            continue
        r.append(calcR(i,j))
print("\nr计算已完成，排序中。。。")
r.sort(key=lambda x:x[2])
print("排序完成,合并数值中。。。 时间：{:.4f}".format(time.time()-starttime)) 
rnew = []
rnew2 = []
for i in range(0,len(r)-1):
    if r[i][2]!=r[i+1][2]:
        rnew.append(r[i])
    else:
        rnew2.append(r[i])
rnew.append(r[i+1])
rnew = np.array(rnew)
for i in rnew2:
    #index = list(rnew[:,2]).index(i[2])
    index = np.argwhere(rnew[:,2]==i[2])[0][0]
    rnew[index][0] += i[0]
    rnew[index][1] += i[1]
rnew = list(map(lambda x:[x[0],x[1]/x[0]/r1_down,x[3],x[2]],rnew))
#rnew = [j for i,j in enumerate(rnew) if i%138==0]
datar = pd.DataFrame(rnew,columns=['R总对数','R','距离r','距离r的平方'])
print("R数据正在保存为csv。。。 时间：{:.4f}".format(time.time()-starttime)) 

datar.to_csv("R.csv",encoding='ANSI')
print('-'*40)
# %%
