import h5py
import os
import numpy as np
import pandas as pd
import torch
import pickle
import random
from skimage import io, util, transform
from skimage import data,exposure,img_as_float
from PIL import Image,ImageFilter
from torch.utils.data import Dataset
from tqdm import tqdm


def clean_slidingWCcsv():
	with open('../data/dev_dataset_csv/newdata.csv', 'w') as outf:
		with open('../data/dev_dataset_csv/slidingWindowClean.csv', 'r') as infi:
			for line in tqdm(infi):
				line = line.rstrip().split(',')
				for i in line[1:-1]:
					outf.write(i+',')
				outf.write(line[-1]+'\n')


def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def shift_hue(arr,hout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb


############### Create data sample list #################
def create_samples(root, shuffle=True, nat_sort=False):
	print('Creating data sample list...')
	f = pd.read_csv(root)
	data_samples = []
	for idx, row in tqdm(f.iterrows(), desc='loading data sample', ncols=100, total=len(f)):
		beams = row.values[0:13].astype(np.float32)
		img_paths = row.values[13:]
		for i, path in enumerate(img_paths):
			img_paths[i] = path.replace("\\", "/")
			# img_paths[i] = '../data/dev_dataset_csv/'+path
		sample = list( zip(img_paths,beams) )
		data_samples.append(sample)

	if shuffle:
		random.shuffle(data_samples)
	print('list is ready')
	return data_samples


def create_images_arr():
	"""Create images array.
	Needed when no pickle file exists.
	"""
	dir_name = "../data/dev_dataset_csv/visual_data/instance"
	images = []
	for idx in range(1, 3997):
		# import pdb; pdb.set_trace()
		filename = dir_name+str(idx)+'/cam'
		for cam in range(1, 7):
			imgname = filename+str(cam)+'.jpg'
			image = io.imread(imgname)
			image = transform.resize(image, (160, 256))
			image = util.img_as_ubyte(image)
			images.append(image)
	images = np.stack(images, axis=0)
	outfile = '../data/dev_dataset_csv/train_data.pkl'
	with open(outfile, 'wb') as outputf:
		pickle.dump(images, outputf, protocol=pickle.HIGHEST_PROTOCOL)


#########################################################

class DataFeed(Dataset):
	"""
	A class fetching a PyTorch tensor of beam indices.
	"""

	def __init__(self, root_dir,
				n,
				img_dim,
				transform=None,
				init_shuflle=True,
				noise_dict=None):
		# # Noise
		# 'noise_dict' : {
		#     'dev_def_image_prob': 0,
		#     'dev_def_beam_prob': 0,
		#     'wh_noi_std_beam': 0,
		#     'wh_noi_std_image': 0,
		#     'brightness_change_var': 0,
		#     'brightness_change_prob': 0,
		#	  'hue_std': 0
		# }
		self.noise_dict = noise_dict
		self.root = root_dir
		self.samples = create_samples(self.root, shuffle=init_shuflle)
		self.inp_seq = 8
		self.transform = transform
		self.seq_len = n
		self.img_dim = img_dim
		# import pdb; pdb.set_trace()
		imagefile = '../data/dev_dataset_csv/train_data.pkl'
		if not os.path.isfile('../data/dev_dataset_csv/train_data.pkl'):
			create_images_arr()
		with open(imagefile, 'rb') as inputf:
			self.images = pickle.load(inputf)

	def __len__(self):
		return len(self.samples)

	def __brightness_change__(self, image):
		if np.random.rand() <= self.noise_dict['brightness_change_prob']:
			brightness_change_var = self.noise_dict['brightness_change_var']
		else:
			brightness_change_var = 0
		brightness = np.random.normal(1, brightness_change_var)
		image = exposure.adjust_gamma(image, brightness) 
		return image

	def __color_change__(self, image):
		if np.random.rand() <= self.noise_dict['hue_change_prob']:
			hue = np.random.normal(180, self.noise_dict['hue_std']) / 360.0
			image = shift_hue(image, hue)
		return image
		
	def __dev_def_beams__(self, beams):
		for i in range(len(beams)):
			if np.random.rand() <= self.noise_dict['dev_def_beams_prob']:
				beams[i] = torch.tensor(np.floor(np.random.rand() * 128))
		return beams
		
	def __dev_def_img__(self, image):
		if np.random.rand() <= self.noise_dict['dev_def_image_prob']:
			image = np.floor(np.random.rand(image.shape[0], image.shape[1], image.shape[2]) * 256).astype(int)
		return image

	def __getitem__(self, idx):
		sample = self.samples[idx] # Read one data sample
		assert len(sample) >= self.seq_len, 'Unsupported sequence length'
		sample = sample[:self.seq_len] # Read a sequence of tuples from a sample
		beams = torch.zeros((self.seq_len,))
		b_noise = torch.from_numpy(np.random.normal(0, self.noise_dict['wh_noi_std_beam'], (self.seq_len,)).astype(int))
		assert (beams + b_noise).shape == beams.shape, 'Unsupported noised shape'
		beams = b_noise + beams
		beams = self.__dev_def_beams__(beams)
		for i,s in enumerate( sample ):
			x = s[1] # Read only beams
			beams[i] = torch.tensor(x, requires_grad=False)
		images = []
		for dp in sample[:self.inp_seq]:
			instances = dp[0].find('instance')
			instanceb = dp[0].find('/cam')
			camb = dp[0].find('.jpg')
			instance = int(dp[0][instances + 8 : instanceb])
			cam = int(dp[0][instanceb + 4 : camb])
			image = self.images[(instance - 1) * 6 + cam - 1]
			i_noise = np.random.normal(0, self.noise_dict['wh_noi_std_image'], (image.shape[0], image.shape[1], image.shape[2])).astype(int)
			assert (image + i_noise).shape == image.shape, 'Unsupported noised shape'
			image = i_noise + image
			image[image>255] = 255
			image[image<0] = 0
			image = self.__dev_def_img__(image)
			image = self.__brightness_change__(image)
			image = self.__color_change__(image)
			image = util.img_as_float(image)
			image = torch.from_numpy(image)
			image = image.permute(2, 0, 1)
			image = self.transform(image)
			images.append(image)
		images = torch.stack(images, dim=0)
		return (beams, images)

#hf = h5py.File('data/dev_dataset_h5/dev_dataset_h5/train_set.h5','r') as f
# clean_slidingWCcsv()
#hf.close()
