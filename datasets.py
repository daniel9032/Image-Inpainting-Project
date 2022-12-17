import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision import transforms
import torchvision.transforms.functional as TF

class ImageDataset(Dataset):
	def __init__(self):
		self.image_names = []
		for (root, dirs, files) in os.walk('E:/train_large_places365standard', topdown=True):
			for names in files:
				self.image_names.append(os.path.join(root, names))

	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, idx):
		img_name = self.image_names[idx]
		img_name = str(img_name)
		img = Image.open(img_name).convert('RGB')
		img = transforms.Resize((256, 256))(img)
		img = TF.to_tensor(img)
		img = transforms.Lambda(lambda x: (x * 2) - 1)(img)     # Scale between [1, -1]
		return img

def random_bbox(img_shape=(256, 256)):
	vertical_margin = horizontal_margin = 0
	height = width = 128
	img_height, img_width = img_shape
	maxt = img_height - vertical_margin - height
	maxl = img_width - horizontal_margin - width
	t = np.random.randint(vertical_margin, maxt)
	l = np.random.randint(horizontal_margin, maxl)

	return (t, l, height, width)

def bbox2mask(bbox, img_shape=(256, 256)):
	img_height ,img_width = img_shape
	max_delta_height = max_delta_width = 32
	mask = torch.zeros((1, 1, img_height, img_width), dtype=torch.float32)
	h = np.random.randint(max_delta_height // 2 + 1)
	w = np.random.randint(max_delta_width // 2 + 1)
	mask[:, :, bbox[0]+h: bbox[0]+bbox[2]-h, bbox[1]+w: bbox[1]+bbox[3]-w] = 1.

	# mask shape [1, 1, H, W]
	return mask

def brush_stroke_mask(img_shape=(256, 256)):
	max_num_vertex = 12
	min_num_vertex = 4
	max_width = 40
	min_width = 12
	mean_angle = 2 * np.pi / 5
	angle_range = 2 * np.pi / 15

	H, W = img_shape
	average_radius = np.sqrt(H*H+W*W) / 8
	mask = Image.new('L', (W, H), 0)

	for _ in range(np.random.randint(1, 4)):
		num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
		angle_min = mean_angle - np.random.uniform(0, angle_range)
		angle_max = mean_angle + np.random.uniform(0, angle_range)
		angles = []
		vertex = []
		for i in range(num_vertex):
			if i % 2 == 0:
				angles.append(2*np.pi - np.random.uniform(angle_min, angle_max))
			else:
				angles.append(np.random.uniform(angle_min, angle_max))

		h, w = mask.size
		vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
		for i in range(num_vertex):
			r = np.clip(
				np.random.normal(loc=average_radius, scale=average_radius//2), 0, 2*average_radius)
			new_x = np.clip(vertex[-1][0] + r * np.cos(angles[i]), 0, w)
			new_y = np.clip(vertex[-1][1] + r * np.sin(angles[i]), 0, h)
			vertex.append((int(new_x), int(new_y)))

		draw = ImageDraw.Draw(mask)
		width = int(np.random.uniform(min_width, max_width))
		draw.line(vertex, fill=1, width=width)
		for v in vertex:
			draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                          fill=1)

	if np.random.normal() > 0:
		mask.transpose(Image.FLIP_LEFT_RIGHT)
	if np.random.normal() > 0:
		mask.transpose(Image.FLIP_TOP_BOTTOM)
	mask = np.asarray(mask, np.float32)
	mask = np.reshape(mask, (1, 1, H, W))

	# mask shape [1, 1, H, W]
	return torch.Tensor(mask)