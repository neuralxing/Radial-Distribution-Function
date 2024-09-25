import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw():
    n=1
    T = 1*n # number if iterations
    theta = 0 # initial values
    k = 10**(-1)
    m=5*10**(-1)
    R=2
    r=1
    d=R-r
    a=6.28
    #x2 = []
    #y2 = []
    x3 = []
    y3 = []
    h1 = []
    h2 = []
    #for I in [0.50]:
    for p_start in np.arange(r,R,0.01): # I的范围，从0.5到3.0，间隔0.04
        for theta_start in np.arange(0,3.14,0.01):
            p = p_start
            z = 0
            theta= theta_start
            x1 = []
            y1 = []
            x1.append(theta)
            y1.append(p) #将I,和theta的值储存起来，用于画图
            #x2.append(theta)
            #y2.append(p) #将I,和theta的值储存起来，用于保存
            h1.append(z)
            for t in range(T):
                p1 = p
                theta1 = theta
                p = p1 + k * np.sin(m*theta1)
                z1 = z
                z = z1 + k * np.sin(m*theta1)
                if p > R :
                    p = p - d
                if p < r:
                    p = p +d
                theta=theta1+(p1-r)/d*a*(1/n)+k/d*a*(1/n)*np.sin(m*theta1)
                theta = theta % (6.28)
                z = z % 6

                x1.append(theta)
                y1.append(p) #将I,和theta的值储存起来，用于画图
                h1.append(z)
                #x2.append(theta)
                #y2.append(p) #将I,和theta的值储存起来，用于保存

            x3.append(x1[-1])
            y3.append(y1[-1])
            h2.append(h1[-1])
            #ax = plt.subplot(111, projection='3d')
            #ax.scatter(x1[-1], y1[-1],h1[-1], s=1.06)
    #plt.legend(loc=0)
    #plt.show()
    ax = plt.subplot(111, projection="3d")
    ax.scatter(x3,y3,h2, s=1.05)
    plt.show()

    import pandas as pd
    df = pd.DataFrame({'theta_last':x3,'p_last':y3,'z_last':h2})
    df.to_csv('output.csv', index=False)

draw()