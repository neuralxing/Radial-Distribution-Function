import numpy as np
import matplotlib.pyplot as plt

# create a 10x10 array of random integers from 0 to 3
data = np.random.randint(0, 2, size=(1000, 1000))

# define a color map with four colors: black, red, green and blue
cmap = plt.cm.get_cmap("brg", 2)

# show the array as an image using the color map
plt.imshow(data, cmap=cmap)
plt.imshow(data)
plt.savefig("1.png")