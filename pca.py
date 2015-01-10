import numpy as np

def pca():
	#get points
	pt = np.array([[0,1,2,3,4,3,3,5,2,6], [0,1,2,1,2,2,3,3,1,4]])
	pt = pt.T

	pt_centered = pt - np.mean(pt, axis=0)

	#covariance
	cv = np.dot(pt_centered.T,pt_centered)/(pt.shape[0]-1)

	eig_val, eig_vec = np.linalg.eig(cv)

	print("eig_val = ")
	print(eig_val)
	print("eig_vec = ")
	print(eig_vec)