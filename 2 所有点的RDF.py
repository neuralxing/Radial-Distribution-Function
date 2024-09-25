import matplotlib.pyplot as plt  
import numpy as np  
import math  
import pandas as pd  
from PIL import Image  
  
# 读取图片并转换为numpy数组  
image_path = '1.tif'  # 替换为你的图片路径  
image = Image.open(image_path)  
image = image.convert('RGB')  
image_data = np.array(image)  
  
# 将红色识别为1，蓝色识别为0  
red_threshold = (200, 0, 0)  
blue_threshold = (0, 0, 200)  
  
converted_image_data = np.zeros_like(image_data[:, :, 0])  
for i in range(image_data.shape[0]):  
    for j in range(image_data.shape[1]):  
        pixel = image_data[i, j]  
        if all(pixel >= red_threshold):  
            converted_image_data[i, j] = 1  
        elif all(pixel <= blue_threshold):  
            converted_image_data[i, j] = 0  
  
data = converted_image_data  
  
class Atom():  
    def __init__(self):  
        self.x_cor = None  
        self.y_cor = None  
        self.type = None  
  
    def pbc_boundary(self, alatt):  
        if self.x_cor < 0:  
            self.x_cor += alatt  
        if self.x_cor >= alatt:  
            self.x_cor -= alatt  
        if self.y_cor < 0:  
            self.y_cor += alatt  
        if self.y_cor >= alatt:  
            self.y_cor -= alatt  
  
def build_crystral(data):  
    atoms = []  
    alatt = len(data)  
    for ii in range(len(data)):  
        for jj in range(len(data[0])):  
            this_atom = Atom()  
            this_atom.x_cor = jj  
            this_atom.y_cor = ii  
            this_atom.pbc_boundary(alatt)  
            if data[ii][jj] == 0:  
                this_atom.type = 'blue'  
            else:  
                this_atom.type = 'red'  
            atoms.append(this_atom)  
    return atoms, alatt  
  
def calculate_distance(atomii, atomjj):  
    dr = math.sqrt((atomii.x_cor - atomjj.x_cor)**2 + (atomii.y_cor - atomjj.y_cor)**2)  
    return dr  
  
atoms, alatt = build_crystral(data)  
ndot = alatt * len(data[0])  
natom = len(atoms)  
number_density = natom / ndot  
  
# Plotting and RDF calculation for all atoms  
rdf_results = []  
for atom_index, atom_flag in enumerate(atoms):  
    gr = np.zeros(int(alatt), dtype='int')  
    for atom in atoms:  
        if atom.type != atom_flag.type:  
            drij = calculate_distance(atom_flag, atom)  
            if drij < alatt:  
                layer = int(drij)  
                gr[layer] += 1  
  
    count = len(gr)  
    results = []  
    for ii in range(count):  
        r = ii  
        s = 2 * math.pi * r * 1  # dr is 1 here  
        if s > 0:  
            rho = gr[ii] / s / natom  
        else:  
            rho = 0  
        results.append([r, rho / number_density])  
    rdf_results.append(results)  

  
# Optionally save all RDF results to a single Excel file with different sheets  
with pd.ExcelWriter('all_rdf_results.xlsx') as writer:  
    for index, results in enumerate(rdf_results):  
        df = pd.DataFrame(results, columns=['Radius', 'RDF'])  
        df.to_excel(writer, sheet_name=f'Atom_{index}', index=False)  
  
# Plotting all RDFs on a single plot  
fig, ax = plt.subplots(figsize=(10, 6))  
for results in rdf_results:  
    df = pd.DataFrame(results, columns=['Radius', 'RDF'])  
    ax.plot(df['Radius'], df['RDF'], label=f'Atom RDF')  
ax.set_xlabel('Radius')  
ax.set_ylabel('Radial Distribution Function (RDF)')  
ax.legend()  
plt.show()