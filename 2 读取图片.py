# build crystal
#https://zhuanlan.zhihu.com/p/481780242
import math
import matplotlib.pyplot as plt
import numpy as np
data = [[1,1,1,1,1,0,1,1],
        [1,1,1,1,0,1,1,1],
        [0,1,1,1,0,1,1,1],
        [1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,1,0,1,1,0,0,0],
        [1,1,1,1,1,0,1,1]
        ]

from PIL import Image  
  
# 读取图片并转换为numpy数组  
image_path = '1.tif'  # 替换为你的图片路径  
image = Image.open(image_path)  
image = image.convert('RGB')  
data = np.array(image)  
  
# 将红色识别为1，蓝色识别为0  
# 假设红色定义为RGB(255,0,0)，蓝色定义为RGB(0,0,255)  
# 你可以根据需要调整这些阈值  
red_threshold = (200, 0, 0)  
blue_threshold = (0, 0, 200)  
  
# 创建一个新的数组来存储转换后的数据  
converted_data = np.zeros_like(data[:, :, 0])  
for i in range(data.shape[0]):  
    for j in range(data.shape[1]):  
        pixel = data[i, j]  
        if all(pixel >= red_threshold):  
            converted_data[i, j] = 1  
        elif all(pixel <= blue_threshold):  
            converted_data[i, j] = 0  

data = converted_data

class Atom():
    def __init__(self):
        self.x_cor = None
        self.y_cor = None
    
    def pbc_boundary(self):
        if(self.x_cor<0): self.x_cor += alatt
        if(self.x_cor >= alatt): self.x_cor -= alatt
        if(self.y_cor<0): self.y_cor += alatt
        if(self.y_cor >= alatt): self.y_cor -= alatt
    

def build_crystral():
    atoms = []
    for ii in range(0, ndot): # id
        if data[ii//alatt][ii%alatt] == 0: continue
        this_atom = Atom()
        this_atom.x_cor = ii % alatt
        this_atom.y_cor = int(ii / alatt)
        this_atom.pbc_boundary()
        atoms.append(this_atom)
    return atoms

def calculate_distance(atomii, atomjj):
    dr = math.sqrt((atomii.x_cor - atomjj.x_cor)**2 + (atomii.y_cor - atomjj.y_cor)**2) 
    return dr

natom = sum([sum(i) for i in data])
alatt = 8
ndot = alatt**2
atoms = build_crystral()
for atom in atoms:
    plt.scatter(atom.x_cor,atom.y_cor, c='blue', marker='o')
number_density = natom / (alatt**2)
number_density = natom / ndot
dr = 1  # 半径
rmax = 8 # 最大半径
layer_number = int(rmax/dr) # total number of layers
atom_flag = atoms[0]

gr = np.zeros(layer_number, dtype='int') # number of atoms per layer

for atom in atoms:
    drij = calculate_distance(atom_flag, atom)
    if (drij < rmax):
        layer = int(drij/dr)
        r = layer * dr # inner radius
        gr[layer] += 1

count = len(gr)
x = []
y = []
for ii in range(0, count):
    r = ii * dr
    x.append(r)
    s = 2 * math.pi * r * dr 
    if(s > 0):
        rho = gr[ii] / s / natom
    else: rho = 0
    y.append(rho / number_density)

fig = plt.figure(figsize=(20,10))
ax0 = fig.add_subplot(1,2,1)
for atom in atoms:
    plt.scatter(atom.x_cor,atom.y_cor, c='blue', marker='o')
plt.scatter(atom_flag.x_cor, atom_flag.y_cor, c='red', marker='o')
for r in np.arange(1,8):
    circle = plt.Circle((atom_flag.x_cor, atom_flag.y_cor), r, color='red', fill=False)
    plt.gcf().gca().add_artist(circle)
plt.grid()

ax0 = fig.add_subplot(1,2,2)
plt.plot(x,y)
plt.show()