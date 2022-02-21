import imageio
import numpy as np
root = 'hci_dataset/additional/'
imageId = 'dishes'
pfix = 'input_Cam0'
formt = '.png'
ang_res = 9
filenames = []

#Top Horizontal
for i in range(0,ang_res):
    if i<10:
        s = pfix+'0'+str(i)
    else:
        s = pfix+str(i)
    filenames.append(s)

#Right Vertical
j=ang_res-1
for i in range(ang_res,ang_res+ang_res-1):
    j+=ang_res
    if j<10:
        s = pfix+'0'+str(j)
    else:
        s = pfix+str(j)
    filenames.append(s)

#Bottom Horizontal
for i in range(0,ang_res-1):
    j-=1
    if j<10:
        s = pfix+'0'+str(j)
    else:
        s = pfix+str(j)
    filenames.append(s)

#Left Vertical
j=j
for i in range(0,ang_res-1):
    j-=ang_res
    if j<10:
        s = pfix+'0'+str(j)
    else:
        s = pfix+str(j)
    filenames.append(s)

images = []
for filename in filenames:
    images.append(imageio.imread(root+imageId+'/'+filename+formt))
imageio.mimsave(imageId+'.gif', images, format='GIF', duration=0.001, fps=60)

from pathlib import Path
import numpy as np
import struct


def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:
        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')

        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4

        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale


import matplotlib.pyplot as plt

depth_image = read_pfm(root+imageId+'/'+'gt_depth_lowres.pfm') #depth disp
print('Minimum Value: ',np.min(depth_image))
print('Maximum Value: ',np.max(depth_image))
plt.imsave(imageId+'_depth.jpeg',depth_image)


disp_image = read_pfm(root+imageId+'/'+'gt_disp_lowres.pfm') #depth disp
print('Minimum Value: ',np.min(disp_image))
print('Maximum Value: ',np.max(disp_image))
plt.imsave(imageId+'_disparity.jpeg',disp_image)