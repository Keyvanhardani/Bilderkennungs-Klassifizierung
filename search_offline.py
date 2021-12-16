#!/usr/bin/env python
# coding: utf-8

# In[25]:


from PIL import Image
from pathlib import Path
import numpy as np


# In[27]:


# Adding feature extractor class
from feature_extracer import FeatureExtracer
#get_ipython().run_line_magic('run', 'feature_extracer.ipynb import FeatureExtracter # Adding Jupyter Notebook file')

# Read the images from Path
if __name__ == "__main__":
    fe = FeatureExtracer()
    
    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)
    
        # set feature path and convert the images to .npy and replace them into feature folder
        # extrac a the deep feature
        feature = fe.extract(img=Image.open(img_path))
        #print(type(feature), feature.shape)
        
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")
        print(feature_path)

        # Save the feature
        np.save(feature_path, feature)



