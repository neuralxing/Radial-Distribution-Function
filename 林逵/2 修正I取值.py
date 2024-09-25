import matplotlib.pyplot as plt
import numpy as np

def draw():
    T = 6# number if iterations
    theta = 0 # initial values
    k = 1
    x2 = []
    y2 = []
    z=[]
    #for I in [0.50]:
    for I_start in np.arange(3,5,1): # I的范围，从0.5到3.0，间隔0.04
        for theta_start in np.arange(0,np.pi,0.5*np.pi):
            I = I_start
            theta= theta_start
            x1 = []
            y1 = []
            for t in range(T):
                I1 = I
                theta1 = theta
                I = I1 + k * np.sin(theta1)
                theta = theta1 + I1 + k * np.sin(theta1)
                theta = theta % (2 * np.pi)
                '''
                while theta > 2 * np.pi:
                    theta = theta - 2.0 * np.pi
                while theta < 0.0:
                    theta = theta + 2.0 * np.pi
                '''
                x1.append(theta)
                y1.append(I) #将I,和theta的值储存起来，用于画图
                x2.append(theta)
                y2.append(I) #将I,和theta的值储存起来，用于保存
                c=360*theta/6.283185306
                z.append(c)                

            #plt.scatter(x1, y1, s=2.6)
            ax = plt.subplot(111, projection='polar')
            ax.scatter(x1, y1, s=6,label="I=%.2f,theta=%.2f" %(I_start,theta_start))
    plt.legend(loc=0)
    plt.show()

    import pandas as pd
    df = pd.DataFrame({'I':y2,'theta':x2,'jiaodu':z})
    df.to_csv('output.csv', index=False)

draw()
