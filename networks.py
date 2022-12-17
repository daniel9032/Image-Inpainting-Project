from .modules import *

class CoarseGenerator(nn.Module):
	def __init__(self, in_channels, channels):
		super().__init__()
		#self.offset_flow = None
		self.conv = GatedConv2d(in_channels, channels//2, 5, 1, 2)

		# downsampling
		self.down1 = GatedConv2d(channels//2, channels, 3, 2)
		self.down2 = GatedConv2d(channels, channels, 3, 1)
		self.down3 = GatedConv2d(channels, channels*2, 3, 2)
		self.down4 = GatedConv2d(channels*2, channels*2, 3, 1)

		# bottleneck
		self.bn1 = GatedConv2d(channels*2, channels*2, 3, 1)
		self.bn2 = GatedConv2d(channels*2, channels*2, 3, padding=2, dilation=2)
		self.bn3 = GatedConv2d(channels*2, channels*2, 3, padding=4, dilation=4)
		self.bn4 = GatedConv2d(channels*2, channels*2, 3, padding=8, dilation=8)
		self.bn5 = GatedConv2d(channels*2, channels*2, 3, padding=16, dilation=16)
		self.bn6 = GatedConv2d(channels*2, channels*2, 3, 1)
		self.bn7 = GatedConv2d(channels*2, channels*2, 3, 1)

		# upsampling
		self.up1 = TransposedGatedConv2d(channels*2, channels, 3, 1, 1)
		self.up2 = GatedConv2d(channels, channels, 3, 1)
		self.up3 = TransposedGatedConv2d(channels, channels//2, 3, 1, 1)
		self.up4 = GatedConv2d(channels//2, channels//4, 3, 1)

		#to RGB
		self.conv_to_rgb = GatedConv2d(channels//4, 3, 3, 1, activation=None)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = self.conv(x)

		# downsampling
		x = self.down1(x)
		x = self.down2(x)
		x = self.down3(x)
		x = self.down4(x)

		# bottleneck
		x = self.bn1(x)
		x = self.bn2(x)
		x = self.bn3(x)
		x = self.bn4(x)
		x = self.bn5(x)
		x = self.bn6(x)
		x = self.bn7(x)

		# upsampling
		x = self.up1(x)
		x = self.up2(x)
		x = self.up3(x)
		x = self.up4(x)

		#to RGB
		x = self.conv_to_rgb(x)
		x = self.tanh(x)

		return x

class FineGenerator(nn.Module):
	def __init__(self, channels, return_flow=False):
		super().__init__()
		#---Convolution Branch---
		self.conv = GatedConv2d(3, channels//2, 5, 1, 2)

		# downsampling
		self.down1 = GatedConv2d(channels//2, channels//2, 3, 2)
		self.down2 = GatedConv2d(channels//2, channels, 3, 1)
		self.down3 = GatedConv2d(channels, channels, 3, 2)
		self.down4 = GatedConv2d(channels, channels*2, 3, 1)

		# bottleneck
		self.bn1 = GatedConv2d(channels*2, channels*2, 3, 1)
		self.bn2 = GatedConv2d(channels*2, channels*2, 3, padding=2, dilation=2)
		self.bn3 = GatedConv2d(channels*2, channels*2, 3, padding=4, dilation=4)	
		self.bn4 = GatedConv2d(channels*2, channels*2, 3, padding=8, dilation=8)
		self.bn5 = GatedConv2d(channels*2, channels*2, 3, padding=16, dilation=16)

		#---Contextual Attention Branch---
		self.ca_conv = GatedConv2d(3, channels//2, 5, 1, 2)

		# downsampling
		self.ca_down1 = GatedConv2d(channels//2, channels//2, 3, 2)
		self.ca_down2 = GatedConv2d(channels//2, channels, 3, 1)
		self.ca_down3 = GatedConv2d(channels, channels*2, 3, 2)
		self.ca_down4 = GatedConv2d(channels*2, channels*2, 3, 1)

		# bottleneck
		self.ca_bn1 = GatedConv2d(channels*2, channels*2, 3, 1)
		self.contextual_attention = ContextualAttention(ksize=3, 
                                                        stride=1, 
                                                        rate=2, 
                                                        fuse_k=3,
                                                        softmax_scale=10,
                                                        fuse=True,
                                                        device_ids=None,
                                                        return_flow=return_flow,
                                                        n_down=2)
		self.ca_bn3 = GatedConv2d(channels*2, channels*2, 3, 1)
		self.ca_bn4 = GatedConv2d(channels*2, channels*2, 3, 1)

		#---Combined Branch---
		self.cb1 = GatedConv2d(channels*4, channels*2, 3, 1)
		self.cb2 = GatedConv2d(channels*2, channels*2, 3, 1)

		# upsampling
		self.up1 = TransposedGatedConv2d(channels*2, channels, 3, 1, 1)
		self.up2 = GatedConv2d(channels, channels, 3, 1)
		self.up3 = TransposedGatedConv2d(channels, channels//2, 3, 1, 1)
		self.up4 = GatedConv2d(channels//2, channels//4, 3, 1)

		# to RGB
		self.conv_to_rgb = GatedConv2d(channels//4, 3, 3, 1, activation=None)
		self.tanh = nn.Tanh()

	def forward(self, x, mask):
		x_tmp = x

		#---Convolution Branch---
		x = self.conv(x_tmp)

		#downsampling
		x = self.down1(x)
		x = self.down2(x)
		x = self.down3(x)
		x = self.down4(x)

		# bottleneck
		x = self.bn1(x)
		x = self.bn2(x)
		x = self.bn3(x)
		x = self.bn4(x)
		x = self.bn5(x)
		x_conv = x

		#---Contextual Attention---
		x = self.ca_conv(x_tmp)

		# downsampling
		x = self.ca_down1(x)
		x = self.ca_down2(x)
		x = self.ca_down3(x)
		x = self.ca_down4(x)

		# bottleneck
		x = self.ca_bn1(x)
		x, offset_flow = self.contextual_attention(x, x, mask)
		x = self.ca_bn3(x)
		x = self.ca_bn4(x)
		x_ca = x

		# concatenate outputs from both branches
		x = torch.cat([x_conv, x_ca], dim=1)

		#---Combined Branch---
		x = self.cb1(x)
		x = self.cb2(x)

		# upsampling
		x = self.up1(x)
		x = self.up2(x)
		x = self.up3(x)
		x = self.up4(x)

		# to RGB
		x = self.conv_to_rgb(x)
		x = self.tanh(x)

		return x, offset_flow

class Generator(nn.Module):
	def __init__(self, in_channels, channels, return_flow=False):
		super().__init__()
		self.coarse = CoarseGenerator(in_channels, channels)
		self.fine = FineGenerator(channels, return_flow)
		self.return_flow = return_flow

	def forward(self, x, mask):
		x_tmp = x
		x_coarse = self.coarse(x)
		x = x_coarse*mask + x_tmp[:, 0:3, :, :]*(1.0-mask)
		x_fine, offset_flow = self.fine(x, mask)

		if self.return_flow:
			return x_coarse, x_fine, offset_flow
			
		return x_coarse, x_fine

class Discriminator(nn.Module):
	# SN PATCH GAN Discriminator
	def __init__(self, in_channels, channels):
		super().__init__()
		self.conv1 = DConv(in_channels, channels)
		self.conv2 = DConv(channels, channels*2)
		self.conv3 = DConv(channels*2, channels*4)
		self.conv4 = DConv(channels*4, channels*4)
		self.conv5 = DConv(channels*4, channels*4)
		self.conv6 = DConv(channels*4, channels*4)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = nn.Flatten()(x)
		return x