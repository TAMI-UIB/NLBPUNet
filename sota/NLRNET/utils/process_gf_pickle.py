import pickle
from scipy.io import savemat
import matplotlib.pyplot as plt

# path = "cache/results/wv-real/pcnn/"
path = "/media/1/leidj/lx/code3/cache/"
# name = 'wv'
name = 'tem_net'
read_file = open(path+'img.pickle', 'rb')
data = pickle.load(read_file)
print(data['img'].shape, data['img'].dtype)

img = data['img'][:, :, [2,1,0]]
# img = data['img']
plt.imshow(img)
plt.axis('off')
plt.show()

savemat(path+"{}.mat".format(name), {'result': data['img']})
print("end")
