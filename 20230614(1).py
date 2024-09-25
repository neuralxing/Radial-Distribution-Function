import matplotlib.pyplot as plt
import numpy as np

def draw():
    n=8
    T = 4*n # number if iterations
    theta = 0 # initial values
    k = 0.02
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    z=[]
    z=[]
    #for I in [0.50]:
    for I_start in np.arange(1,2,0.01): # I的范围，从0.5到3.0，间隔0.04
        for theta_start in np.arange(0,3.14,0.01):
            I = I_start
            theta= theta_start
            x1 = []
            y1 = []
            x1.append(theta)
            y1.append(I) #将I,和theta的值储存起来，用于画图
            x2.append(theta)
            y2.append(I) #将I,和theta的值储存起来，用于保存
            c=360*theta/6.28
            z.append(c)                
            for t in range(T):
                I1 = I
                theta1 = theta
                I = I1 + k * np.sin(10**(2)*theta1)
                if I > 2 :
                    I = I - 2
                if I < 1:
                    I = 2 - I
                theta=theta1+(I1-1)*2*3.14*(1/n)+k*2*3.14*(1/n)*np.sin(10**(2)*theta1)
                theta = theta % (2 * 3.14)

                x1.append(theta)
                y1.append(I) #将I,和theta的值储存起来，用于画图
                #x2.append(theta)
                #y2.append(I) #将I,和theta的值储存起来，用于保存
                c=360*theta/6.28
                z.append(c)                

            #ax = plt.subplot(111, projection='polar')
            #ax.scatter(x1[-1], y1[-1], s=0.05)
            x3.append(x1[-1])
            y3.append(y1[-1])
    #plt.legend(loc=0)
    #ax = plt.subplot(111), projection='polar')
    #ax.scatter(x2, y2, s=0.05)
    #plt.show()

    import pandas as pd
    df = pd.DataFrame({'theta_last':x3,'I_last':y3})
    df.to_csv('output.csv', index=False)
    return x3,y3

x,y = draw()
x = np.array(list(map(round,np.array(x)*100))).astype(int)
y = np.array(list(map(round,np.array(y)*100))).astype(int)
# 获取x和y的最大值，用于确定数组的大小
max_x = max(x)
max_y = max(y)

# 创建一个全为0的二维数组，形状为(max_x + 1) x (max_y + 1)
arr = np.zeros((max_x + 1, max_y + 1))

# 遍历x和y的值，将数组中对应位置的元素赋值为1
for i in range(len(x)):
    arr[x[i], y[i]] = 1

# 保存数组
np.savetxt("arr.csv",arr,fmt='%1.0f',delimiter=',')

plt.imshow(arr[:,:],cmap='binary')
plt.show()


def numb_01(img):
    # 得到行和列
    raw = np.shape(img)[0]
    column = np.shape(img)[1]
    w_ij_sum = 4*2 + (raw*2+column*2-8)*3 + (raw-2)*(column-2)*4 # W_ij总的值
    w_lie_ij_sum = column*2 + (raw-2)*column*2 # W_hang_ij总的值
    w_hang_ij_sum = raw*2 + (column-2)*raw*2 # W_lie_ij总的值
    x_mean = np.mean(img)# x的平均值
    img_fla = img.flatten()
    # 求1和0的个数
    numb_1 = np.sum(img)
    numb_0 = len(img_fla)-numb_1
    print("0的个数：%s" % numb_0)
    print("1的个数：%s" % numb_1)

    I_up_sum = 0  # 求I时的分子
    I_up_list = []
    for i, x_i in enumerate(img.flatten(), 1):
        # i=17
        j_sum = [i-1, i+1, i-column, i+column]
        if i % column == 0:
            j_sum = [i-1, i-column, i+column]
        elif i % column == 1:
            j_sum = [i+1, i-column, i+column]
        I_up = 0
        for j in j_sum:
            if j > raw*column or j < 1:
                continue
            x_j = img_fla[j-1]
            I_up += (x_i - x_mean)*(x_j - x_mean)  # 公式在这里
        I_up_sum += I_up
        I_up_list.append(I_up)
    # 求I的值
    I = I_up_sum/(w_ij_sum*np.var(img))  # np.var为求方差
    print("I的值为 %s    <--" % I)

    I_hang_up_sum = 0  # 求I_hang时的分子
    I_hang_up_list = []
    for i, x_i in enumerate(img.flatten(), 1):
        j_hang_sum = [i-1, i+1]
        if i % column == 0:
            j_hang_sum = [i-1]
        elif i % column == 1:
            j_hang_sum = [i+1]
        I_hang_up = 0
        for j in j_hang_sum:
            if j > raw*column or j < 1:
                continue
            x_j = img_fla[j-1]
            I_hang_up += (x_i - x_mean)*(x_j - x_mean)  # 公式在这里
        I_hang_up_sum += I_hang_up
        I_hang_up_list.append(I_hang_up)
    # 求I_hang的值
    I_hang = I_hang_up_sum/(w_hang_ij_sum*np.var(img))  # np.var为求方差
    print("I_hang的值为 %s    <--" % I_hang)

    I_lie_up_sum = 0  # 求I_lie时的分子
    I_lie_up_list = []
    for i, x_i in enumerate(img.flatten(), 1):
        j_lie_sum = [i-column, i+column]
        I_lie_up = 0
        for j in j_lie_sum:
            if j > raw*column or j < 1:
                continue
            x_j = img_fla[j-1]
            I_lie_up += (x_i - x_mean)*(x_j - x_mean)  # 公式在这里
        I_lie_up_sum += I_lie_up
        I_lie_up_list.append(I_lie_up)
    # 求I_lie的值
    I_lie = I_lie_up_sum/(w_lie_ij_sum*np.var(img))  # np.var为求方差
    print("I_lie的值为 %s    <--" % I_lie)


'''
img = arr[:,min(y):]
xs,xe,ys,ye = 60,100,0,500
img = img[ys:ye,xs:xe]
plt.imshow(img,cmap='binary')
plt.show()
numb_01(img)
'''