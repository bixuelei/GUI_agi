import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

ara=np.random.randint(15,size=(3,4))
index=np.argsort(ara[:,1])
ara_=ara[index,:]
