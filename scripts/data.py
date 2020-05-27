import h5py

with h5py.File('data/dev_dataset_h5/train_set.h5','r') as f:
    print(f.keys())
