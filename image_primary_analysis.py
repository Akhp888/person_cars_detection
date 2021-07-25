# import modules
import imagesize
import glob
import matplotlib.pyplot as plt

#plt histogram
def plt_hist(par_list,plt_name):
    plt.hist(par_list, bins=30)
    plt.ylabel('count')
    plt.xlabel('width')
    plt.title(plt_name)
    plt.show()

#load all images
images_list = glob.glob('D:/projects/misc/ev_inference/trainval/images/*.jpg')
print('total images : ',len(images_list))

width_list = []
height_list = []

for imgs in images_list:
    width, height = imagesize.get(imgs)
    width_list.append(width)
    height_list.append(height)

print('minimum and maximum width of images are {:d},{:d} respectively'.format(min(width_list),max(width_list)))
print('minimum and maximum height of images are {:d},{:d} respectively'.format(min(height_list),max(height_list)))

plt_hist(width_list,'width')
plt_hist(height_list,'height')