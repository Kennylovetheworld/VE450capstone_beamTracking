import h5py
import os
import numpy as np
import pandas as pd
import torch
import cv2
import random
from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm


############### Create data sample list #################
def create_samples(root, shuffle=True, nat_sort=False):
	print('Creating data sample list...')
	f = pd.read_csv(root)
	data_samples = []
	for idx, row in tqdm(f.iterrows(), desc='loading data sample', ncols=100, total=len(f)):
		beams = row.values[0:13].astype(np.float32)
		img_paths = row.values[13:]
		for i, path in enumerate(img_paths):
			path = path.replace("\\", "/")
			img_paths[i] = '../data/dev_dataset_csv/'+path
		# import pdb; pdb.set_trace()
		sample = list( zip(img_paths,beams) )
		data_samples.append(sample)

	if shuffle:
		random.shuffle(data_samples)
	print('list is ready')
	return data_samples


#########################################################

class DataFeed(Dataset):
	"""
	A class fetching a PyTorch tensor of beam indices.
	"""

	def __init__(self, root_dir,
				n,
				img_dim,
				transform=None,
				init_shuflle=True):

		self.root = root_dir
		self.samples = create_samples(self.root, shuffle=init_shuflle)
		self.inp_seq = 8
		self.transform = transform
		self.seq_len = n
		self.img_dim = img_dim

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx] # Read one data sample
		assert len(sample) >= self.seq_len, 'Unsupported sequence length'
		sample = sample[:self.seq_len] # Read a sequence of tuples from a sample
		beams = torch.zeros((self.seq_len,))
		for i,s in enumerate( sample ):
			x = s[1] # Read only beams
			beams[i] = torch.tensor(x, requires_grad=False)
		images = []
		for dp in sample[:self.inp_seq]:
			image = io.imread(dp[0])
			# import pdb; pdb.set_trace()
			# image2 = cv2.imread(dp[0])
			image = self.transform(image)
			images.append(image)
		return beams

#hf = h5py.File('data/dev_dataset_h5/dev_dataset_h5/train_set.h5','r') as f

#hf.close()
