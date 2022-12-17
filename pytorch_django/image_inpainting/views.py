from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from .forms import ImageUploadForm
from .networks import Generator
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
import torch
import torchvision
import numpy as np
import base64
import io
import os
from io import BytesIO
from django.http import JsonResponse

IMGSIZE = (512, 512)
ITER = 1560000
img_size = None
inpainted_img_url = None
image_byte = None
image_url = None
device = torch.device('cpu')

# Load pretrained model
generator = Generator(4, 64)
generator = generator
pretrained_state_dict = f'C:\\Users\\User\\Desktop\\pytorch_django\\image_inpainting\\Inpainting_model_state_dict_iter_{ITER}.pt'
checkpoint = torch.load(pretrained_state_dict, map_location=torch.device(device))
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

def image_to_byte_array(image: Image) -> bytes:
	# BytesIO is a fake file stored in memory
	imgByteArr = io.BytesIO()

	# image.save expects a file as a argument, passing a bytes io ins
	image.save(imgByteArr, format='JPEG')

	# Turn the BytesIO object back into a bytes object
	imgByteArr = imgByteArr.getvalue()
	return imgByteArr

# Preprocess the image
def transform_image(img):
	img = transforms.Resize(IMGSIZE)(img)
	img = transforms.Lambda(lambda x: (x * 2) - 1)(img)     # Scale between [1, -1]
	return img

def transform_mask(mask):
	mask = transforms.Resize(IMGSIZE)(mask)
	return mask

def inpaint(image, mask):
	orig_mask = mask  # Type: tensor
	orig_img = image  # Type: tensor

	img = transform_image(image)  # After resize
	mask = transform_mask(mask)   # After resize

	incomplete_img = img * (1.0 - mask)
	ones_x = torch.ones_like(incomplete_img)[:, 0:1, :, :]
	x = torch.cat([incomplete_img, ones_x * mask], axis=1)

	_, fine_img = generator(x, mask)

	fine_img = transforms.Resize(img_size)(fine_img)

	filled = ((fine_img + 1) * 127.5) * orig_mask
	result = filled + orig_img * (1.0 - orig_mask) * 255
	result = result.squeeze().detach().numpy()

	inpainted_img = (result).astype(np.uint8).transpose(1, 2, 0)
	inpainted_img = Image.fromarray(inpainted_img).convert('RGB')
	return inpainted_img

@csrf_exempt
def post_request(request):

	# Receive mask drawn by user, and inpaint the image accordingly
	global inpainted_img_url, img_size
	if request.method == "POST":
		canvas_url = request.POST['mask']
		new_img_url = request.POST['new_img']
		mask_data = canvas_url[22:]
		new_img_data = new_img_url[23:]

		mask_rgba = Image.open(BytesIO(base64.b64decode(mask_data)))  # Mask type: pil image
		mask_rgba_t = TF.to_tensor(mask_rgba)
		mask_a_t = mask_rgba_t[3]  # Extract the alpha channel
		mask_a_t = torch.where(mask_a_t != 0, 1, 0)  # If the value is not zero, change it to one
		mask = mask_a_t.unsqueeze(0).unsqueeze(0)  # Change the size to [1, 1, H, W]
		mask = mask.to(torch.float)

		img = Image.open(BytesIO(base64.b64decode(new_img_data))).convert('RGB')  # Image type: pil image
		img_size = [img.height, img.width]
		img = TF.to_tensor(img)
		img = img.unsqueeze(0)  # Image size: [1, 3, H, W]

		inpainted_img = inpaint(img, mask)
		inpainted_img = image_to_byte_array(inpainted_img)
		encoded_img = base64.b64encode(inpainted_img).decode('ascii')
		inpainted_img_url = f'data:image/jpeg;base64,{encoded_img}'
		
	return HttpResponse("")

def return_result(request):
	global inpainted_img_url
	data = {"img": inpainted_img_url}
	return JsonResponse(data)

def index(request):
	global image_byte, image_url
	if request.method == 'POST':
		# In case of POST: get the uploaded image from the form and process it
		img_form = ImageUploadForm(request.POST, request.FILES)

		if img_form.is_valid():
			# Retrieve the uploaded image and convert it to bytes (for pytorch)
			image = img_form.cleaned_data['image']
			image_byte = image.file.read()

			# Convert and pass the image as base64 string to avoid storing it to DB or file system
			encoded_img = base64.b64encode(image_byte).decode('ascii')
			image_url = f'data:image/jpeg;base64,{encoded_img}'

	else:
		# In case of GET: simply show the empty form for uploading images
		img_form = ImageUploadForm()

	# Pass the form, image url and inpainted image to the template to be rendered
	context = {
		'form': img_form,
		'image_url': image_url,
	}
	return render(request, 'image_inpainting/index.html', context)