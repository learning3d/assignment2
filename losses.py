import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.loss import mesh_laplacian_smoothing
# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# loss = 
	# implement some loss for binary voxel grids
	return prob_loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# loss_chamfer = 
	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss = 
	# implement laplacian smoothening loss
	return loss_laplacian