import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

class Data(Dataset):
    def __init__(self, root_dir):
        self.files = []

        for filename in os.listdir(root_dir):
            base, _ = os.path.splitext(filename)
            try:
                filenum = int(base)
            except ValueError:
                continue
            
            if 0 <= filenum < 1600:
                self.files.append(root_dir + '/' + filename)
                # end if
            # end for files in root_dir
        # end if train split
    # end __init__()

    # len overload, returns the number of files in the loader
    def __len__(self):
        return len(self.files)
    # end __len__()

    # bracket [] overload
    def __getitem__(self, index):
        feat_vec = np.load(self.files[index])
        feat_vec = np.asarray(feat_vec)
        return feat_vec
    # end __getitem__()