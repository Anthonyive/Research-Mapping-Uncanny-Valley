from numba import cuda


def release_vram():
	# clear memory of gpu
	device = cuda.get_current_device()
	device.reset()