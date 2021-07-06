''' Layers
    This file contains various layers for the BigGAN models.
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from sync_batchnorm import SynchronizedBatchNorm2d as SyncBN2d


# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
  def forward(self, input):
    return input
 

# Spectral normalization base class 
class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)


# Embedding layer with spectral norm
# We use num_embeddings as the dim instead of embedding_dim here
# for convenience sake
class SNEmbedding(nn.Embedding, SN):
  def __init__(self, num_embeddings, embedding_dim, padding_idx=None, 
               max_norm=None, norm_type=2, scale_grad_by_freq=False,
               sparse=False, _weight=None,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                          max_norm, norm_type, scale_grad_by_freq, 
                          sparse, _weight)
    SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)
  def forward(self, x):
    return F.embedding(x, self.W_())


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
  def __init__(self, ch, which_conv=SNConv2d, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])    
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x


# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
  # Apply scale and shift--if gain and bias are provided, fuse them here
  # Prepare scale
  scale = torch.rsqrt(var + eps)
  # If a gain is provided, use it
  if gain is not None:
    scale = scale * gain
  # Prepare shift
  shift = mean * scale
  # If bias is provided, use it
  if bias is not None:
    shift = shift - bias
  return x * scale - shift
  #return ((x - mean) / ((var + eps) ** 0.5)) * gain + bias # The unfused way.


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
  # Cast x to float32 if necessary
  float_x = x.float()
  # Calculate expected value of x (m) and expected value of x**2 (m2)  
  # Mean of x
  m = torch.mean(float_x, [0, 2, 3], keepdim=True)
  # Mean of x squared
  m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
  # Calculate variance as mean of squared minus mean squared.
  var = (m2 - m **2)
  # Cast back to float 16 if necessary
  var = var.type(x.type())
  m = m.type(x.type())
  # Return mean and variance for updating stored mean/var if requested  
  if return_mean_var:
    return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
  else:
    return fused_bn(x, m, var, gain, bias, eps)


# My batchnorm, supports standing stats    
class myBN(nn.Module):
  def __init__(self, num_channels, eps=1e-5, momentum=0.1):
    super(myBN, self).__init__()
    # momentum for updating running stats
    self.momentum = momentum
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Register buffers
    self.register_buffer('stored_mean', torch.zeros(num_channels))
    self.register_buffer('stored_var',  torch.ones(num_channels))
    self.register_buffer('accumulation_counter', torch.zeros(1))
    # Accumulate running means and vars
    self.accumulate_standing = False
    
  # reset standing stats
  def reset_stats(self):
    self.stored_mean[:] = 0
    self.stored_var[:] = 0
    self.accumulation_counter[:] = 0
    
  def forward(self, x, gain, bias):
    if self.training:
      out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
      # If accumulating standing stats, increment them
      if self.accumulate_standing:
        self.stored_mean[:] = self.stored_mean + mean.data
        self.stored_var[:] = self.stored_var + var.data
        self.accumulation_counter += 1.0
      # If not accumulating standing stats, take running averages
      else:
        self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
        self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
      return out
    # If not in training mode, use the stored statistics
    else:         
      mean = self.stored_mean.view(1, -1, 1, 1)
      var = self.stored_var.view(1, -1, 1, 1)
      # If using standing stats, divide them by the accumulation counter   
      if self.accumulate_standing:
        mean = mean / self.accumulation_counter
        var = var / self.accumulation_counter
      return fused_bn(x, mean, var, gain, bias, self.eps)


# Simple function to handle groupnorm norm stylization                      
def groupnorm(x, norm_style):
  # If number of channels specified in norm_style:
  if 'ch' in norm_style:
    ch = int(norm_style.split('_')[-1])
    groups = max(int(x.shape[1]) // ch, 1)
  # If number of groups specified in norm style
  elif 'grp' in norm_style:
    groups = int(norm_style.split('_')[-1])
  # If neither, default to groups = 16
  else:
    groups = 16
  return F.group_norm(x, groups)


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable). 
class ccbn(nn.Module):
  def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn',):
    super(ccbn, self).__init__()
    self.output_size, self.input_size = output_size, input_size
    # Prepare gain and bias layers
    self.gain = which_linear(input_size, output_size)
    self.bias = which_linear(input_size, output_size)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # Norm style?
    self.norm_style = norm_style
    
    if self.cross_replica:
      self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
    elif self.mybn:
      self.bn = myBN(output_size, self.eps, self.momentum)
    elif self.norm_style in ['bn', 'in']:
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size)) 
    
    
  def forward(self, x, y):
    # Calculate class-conditional gains and biases
    gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
    bias = self.bias(y).view(y.size(0), -1, 1, 1)
    # If using my batchnorm
    if self.mybn or self.cross_replica:
      return self.bn(x, gain=gain, bias=bias)
    # else:
    else:
      if self.norm_style == 'bn':
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'in':
        out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'gn':
        out = groupnorm(x, self.normstyle)
      elif self.norm_style == 'nonorm':
        out = x
      return out * gain + bias
  def extra_repr(self):
    s = 'out: {output_size}, in: {input_size},'
    s +=' cross_replica={cross_replica}'
    return s.format(**self.__dict__)


# Normal, non-class-conditional BN
class bn(nn.Module):
  def __init__(self, output_size,  eps=1e-5, momentum=0.1,
                cross_replica=False, mybn=False):
    super(bn, self).__init__()
    self.output_size= output_size
    # Prepare gain and bias layers
    self.gain = P(torch.ones(output_size), requires_grad=True)
    self.bias = P(torch.zeros(output_size), requires_grad=True)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    
    if self.cross_replica:
      self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)    
    elif mybn:
      self.bn = myBN(output_size, self.eps, self.momentum)
     # Register buffers if neither of the above
    else:     
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size))
    
  def forward(self, x, y=None):
    if self.cross_replica or self.mybn:
      gain = self.gain.view(1,-1,1,1)
      bias = self.bias.view(1,-1,1,1)
      return self.bn(x, gain=gain, bias=bias)
    else:
      return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                          self.bias, self.training, self.momentum, self.eps)

                          
# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must 
# be preselected)
class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=bn, activation=None, 
               upsample=None, attentive=False):
    super(GBlock, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    self.upsample = upsample
    self.attentive = attentive
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
    # Batchnorm layers
    self.bn1 = self.which_bn(in_channels)
    self.bn2 = self.which_bn(out_channels)
    # upsample layers
    self.upsample = upsample

    if attentive:
      # Attentive layers - https://github.com/dvlab-research/AttenNorm/blob/master/inpaint-attnorm/net/network.py
      # Variables
      self.nClass = 16
      self.kama = 10
      self.orth_lambda = 0.001
      # KQV
      self.conv_xk = self.which_conv(self.out_channels, int(self.out_channels / 8), kernel_size=1, padding=0)
      self.conv_xq = self.which_conv(self.out_channels, int(self.out_channels / 8), kernel_size=1, padding=0)
      self.conv_xv = self.which_conv(self.out_channels, self.out_channels, kernel_size=1, padding=0)
      # orthogonality
      # self.x_mask_filters = torch.normal(0, 1, size=(1, 1, self.out_channels, self.nClass))
      # self.x_mask_filters.requires_grad = True
      self.conv_x_mask = self.which_conv(self.out_channels, self.nClass, kernel_size=1, padding=0, bias=False)
      # self.conv_x_mask.weight = self.x_mask_filters
      # self-sampling
      self.alpha = nn.Parameter((torch.ones(1, self.nClass, 1, 1) * 0.1))
      self.softmax = nn.Softmax(dim=1)
      # residual
      self.sigma = nn.Parameter(torch.zeros(1))

  def forward(self, x, y, voxelwise_a_mod, voxelwise_b_mod, voxelwise_a1_mod, voxelwise_b1_mod):
    h = self.activation(self.bn1(x, y))
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)

    if self.attentive:
      # Attentive Spatial Self-modulation - https://github.com/dvlab-research/AttenNorm/blob/master/inpaint-attnorm/net/network.py
      # KQV
      xk = self.conv_xk(h)
      xq = self.conv_xq(h)
      xv = self.conv_xv(h)

      # orthogonality
      # PyTorch Feature Map  ->  BATCH x CHANNEL x WIDTH x HEIGHT
      # PyTorch Weight  ->  OUT_CHANNEL X IN_CHANNEL X KERNEL_SIZE X KERNEL_SIZE
      # TensorFlow Feature Map  ->  BATCH x WIDTH x HEIGHT x CHANNEL
      # TensorFlow Weight  ->  KERNEL_SIZE X KERNEL_SIZE X IN_CHANNEL X OUT_CHANNEL
      x_mask = self.conv_x_mask(h)
      mask_w = torch.reshape(self.conv_x_mask.weight, (self.out_channels, self.nClass))
      mask_w_sym_T = torch.reshape(mask_w.T, (1, self.nClass, self.out_channels))
      mask_w_sym = torch.reshape(mask_w, (1, self.out_channels, self.nClass))
      sym = torch.matmul(mask_w_sym_T, mask_w_sym)
      sym -= torch.reshape(torch.eye(self.nClass).cuda(), (1, self.nClass, self.nClass))
      # orthogonality loss function. if want to optimize, return and add to loss function
      ortho_loss = self.orth_lambda * torch.sum(torch.mean(sym, dim=0))

      # self-sampling
      # PyTorch Feature Map  ->  BATCH x CHANNEL x WIDTH x HEIGHT
      # PyTorch Weight  ->  OUT_CHANNEL X IN_CHANNEL X KERNEL_SIZE X KERNEL_SIZE
      # TensorFlow Feature Map  ->  BATCH x WIDTH x HEIGHT x CHANNEL
      # TensorFlow Weight  ->  KERNEL_SIZE X KERNEL_SIZE X IN_CHANNEL X OUT_CHANNEL
      sampling_pos = torch.multinomial(torch.ones(1, h.shape[2] * h.shape[3]) * 0.5, self.nClass).cuda()
      sampling_pos = torch.squeeze(sampling_pos, dim=0)

      xk_reshaped = torch.reshape(xk, (h.shape[0], h.shape[2] * h.shape[3], int(self.out_channels / 8)))
      fast_filters = torch.index_select(xk_reshaped, 1, sampling_pos)
      fast_filters = torch.reshape(fast_filters, (h.shape[0], int(self.out_channels / 8), self.nClass))

      xq_reshaped = torch.reshape(xq, (h.shape[0], h.shape[2] * h.shape[3], int(self.out_channels / 8)))
      fast_activations = torch.matmul(xq_reshaped, fast_filters)
      fast_activations = torch.reshape(fast_activations, (h.shape[0], self.nClass, h.shape[2], h.shape[3]))

      # calculate per-pixel class-included weights
      layout = self.softmax((torch.clamp(self.alpha, min=0.0, max=1.0) * fast_activations + x_mask) / self.kama)  # BATCH X NCLASS X WIDTH X HEIGHT

      # normalization
      layout_expand = torch.reshape(layout, (layout.shape[0], layout.shape[1], 1, layout.shape[2], layout.shape[3]))  # BATCH X NCLASS X 1 X WIDTH X HEIGHT
      cnt = torch.sum(layout_expand, (3, 4), keepdim=True)
      xv_expand = torch.reshape(xv, (xv.shape[0], 1, xv.shape[1], xv.shape[2], xv.shape[3])).repeat(1, self.nClass, 1, 1, 1)
      hot_area = xv_expand * layout_expand
      xv_mean = torch.mean(hot_area, (3, 4), keepdim=True) / cnt
      xv_std = torch.sqrt(torch.sum((hot_area - xv_mean) ** 2, (3, 4), keepdim=True) / cnt)
      xn = torch.sum((xv_expand - xv_mean) / xv_std * layout_expand, axis=1)

      # residual
      h = h + self.sigma * xn

      # modulation: normalization에서 feature map에 대해서 수행했던 operation을 modulation map에 대해서 그대로 수행하면 된다.
      # h_expand = torch.reshape(h, (h.shape[0], 1, h.shape[1], h.shape[2], h.shape[3])).repeat(1, self.nClass, 1, 1, 1)
      # voxelwise_a1_mod_expand = torch.reshape(voxelwise_a1_mod, (voxelwise_a1_mod.shape[0], 1, voxelwise_a1_mod.shape[1], voxelwise_a1_mod.shape[2], voxelwise_a1_mod.shape[3])).repeat(1, self.nClass, 1, 1, 1)
      # hot_area_a1 = voxelwise_a1_mod_expand * layout_expand
      # voxelwise_a1_modn = torch.mean(hot_area_a1, (3, 4), keepdim=True)
      # voxelwise_b1_mod_expand = torch.reshape(voxelwise_b1_mod, (voxelwise_b1_mod.shape[0], 1, voxelwise_b1_mod.shape[1], voxelwise_b1_mod.shape[2], voxelwise_b1_mod.shape[3])).repeat(1, self.nClass, 1, 1, 1)
      # hot_area_b1 = voxelwise_b1_mod_expand * layout_expand
      # voxelwise_b1_modn = torch.mean(hot_area_b1, (3, 4), keepdim=True)
      # h = torch.sum(((h_expand * (1 + voxelwise_a1_modn)) + voxelwise_b1_modn) * layout_expand, axis=1)

    h = self.activation(self.bn2(h, y))
    h = self.conv2(h)
    if self.learnable_sc:       
      x = self.conv_sc(x)
    h = h + x

    # Spatial Self-modulation
    h = (h - torch.mean(h, dim=(1, 2, 3), keepdim=True)) / torch.std(h, dim=(1, 2, 3), keepdim=True)
    h = h * (1 + voxelwise_a_mod) + voxelwise_b_mod
    return h
  

# SpatialModulationBlock for G.
class SpatialModulationGBlock(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=bn, activation=None, 
               upsample=None):
    super(SpatialModulationGBlock, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    self.upsample = upsample
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    # Modulation layers
    self.voxelwise_a_modulation = self.which_conv(self.out_channels, self.out_channels, kernel_size=1, padding=0)
    self.voxelwise_b_modulation = self.which_conv(self.out_channels, self.out_channels, kernel_size=1, padding=0)
    # self.learnable_sc = in_channels != out_channels or upsample
    # if self.learnable_sc:
    #   self.conv_sc = self.which_conv(in_channels, out_channels, 
    #                                  kernel_size=1, padding=0)
    # Batchnorm layers
    # self.bn1 = self.which_bn(in_channels)
    # self.bn2 = self.which_bn(out_channels)
    # upsample layers
    self.upsample = upsample

  def forward(self, x):
    h = self.activation(x)
    if self.upsample:
      h = self.upsample(h)
      # x = self.upsample(x)
    h = self.conv1(h)

    h = self.activation(h)
    h = self.conv2(h)

    voxelwise_a_mod = self.voxelwise_a_modulation(h)
    voxelwise_b_mod = self.voxelwise_b_modulation(h)
    # if self.learnable_sc:       
    #   x = self.conv_sc(x)
    return h, voxelwise_a_mod, voxelwise_b_mod


# AttentiveSpatialModulationBlock for G.
class AttentiveSpatialModulationGBlock(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=bn, activation=None, 
               upsample=None):
    super(AttentiveSpatialModulationGBlock, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    self.upsample = upsample
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    self.conv3 = self.which_conv(self.out_channels, self.out_channels)
    # Modulation layers
    self.voxelwise_a_modulation = self.which_conv(self.out_channels, self.out_channels, kernel_size=1, padding=0)
    self.voxelwise_b_modulation = self.which_conv(self.out_channels, self.out_channels, kernel_size=1, padding=0)
    # self.learnable_sc = in_channels != out_channels or upsample
    # if self.learnable_sc:
    #   self.conv_sc = self.which_conv(in_channels, out_channels, 
    #                                  kernel_size=1, padding=0)
    # Batchnorm layers
    # self.bn1 = self.which_bn(in_channels)
    # self.bn2 = self.which_bn(out_channels)
    # upsample layers
    self.upsample = upsample

  def forward(self, x):
    h = self.activation(x)
    # 4 to 8
    if self.upsample:
      h = self.upsample(h)
      # x = self.upsample(x)
    h = self.conv1(h)

    h = self.activation(h)
    # 8 to 16
    if self.upsample:
      h = self.upsample(h)
      # x = self.upsample(x)
    h = self.conv2(h)

    h = self.activation(h)
    # 16 to 32
    if self.upsample:
      h = self.upsample(h)
      # x = self.upsample(x)
    h = self.conv3(h)

    voxelwise_a_mod = self.voxelwise_a_modulation(h)
    voxelwise_b_mod = self.voxelwise_b_modulation(h)
    # if self.learnable_sc:       
    #   x = self.conv_sc(x)
    return voxelwise_a_mod, voxelwise_b_mod
    
    
# Residual block for the discriminator
class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
               preactivation=False, activation=None, downsample=None,):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels if wide else self.in_channels
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
    self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
    self.learnable_sc = True if (in_channels != out_channels) or downsample else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.preactivation:
      if self.learnable_sc:
        x = self.conv_sc(x)
      if self.downsample:
        x = self.downsample(x)
    else:
      if self.downsample:
        x = self.downsample(x)
      if self.learnable_sc:
        x = self.conv_sc(x)
    return x
    
  def forward(self, x):
    if self.preactivation:
      # h = self.activation(x) # NOT TODAY SATAN
      # Andy's note: This line *must* be an out-of-place ReLU or it 
      #              will negatively affect the shortcut connection.
      h = F.relu(x)
    else:
      h = x    
    h = self.conv1(h)
    h = self.conv2(self.activation(h))
    if self.downsample:
      h = self.downsample(h)     
        
    return h + self.shortcut(x)
    
# dogball