from networks import Fine_encoder_g, Coarse_encoder_g, Decoder_g, Discriminator, refinement_network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
import argparse
from tqdm import tqdm
import wandb

import face_alignment
from skimage import io
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
import torch
from networks_edge import EdgeGenerator

class GAN(tf.keras.Model):
    def __init__(self,image_shape = (256,256),dual=2,refinement=True,
                      pretrained_fine_encoder = False, attention=True):
        super(GAN, self).__init__()
        
        if dual == 0:
          self.coarse_size = (image_shape[0],image_shape[1],4)
          self.coarse_encoder = Coarse_encoder_g(self.coarse_size)
        elif dual == 1:
          self.fine_size = (image_shape[0],image_shape[1],3)
          self.fine_encoder = Fine_encoder_g(self.fine_size)
        else:
          self.fine_size = (image_shape[0],image_shape[1],3)
          self.coarse_size = (image_shape[0],image_shape[1],4)
          self.fine_encoder = Fine_encoder_g(self.fine_size)
          self.coarse_encoder = Coarse_encoder_g(self.coarse_size)

        if attention == True:
          if dual == 2:
            self.decoder = Decoder_g(input_shape=(32,32,512))
          else:
            self.decoder = Decoder_g(input_shape=(32,32,256))
        else:
          if dual == 2:
            self.decoder = Decoder_g_natt(input_shape=(32,32,512))
          else:
            self.decoder = Decoder_g_natt(input_shape=(32,32,256))

        if refinement == True :
          self.fine_size = (image_shape[0],image_shape[1],4)
          self.refinement_network = refinement_network(input_shape=self.fine_size)

    def fine_encode(self, x):
      return self.fine_encoder(x)
    
    def coarse_encode(self, x):
      return self.coarse_encoder(x)

    def decode(self, f1,f2):
      output = self.decoder([f1,f2])
      return output

def build_networks (pretrained_path, image_shape= (256,256) ,continue_training = False,dual=2,
                    refinement=True,pretrained_fine_encoder=True, attention=True):
  gan = GAN(image_shape= image_shape,dual=dual,refinement=refinement,attention=attention)
  discriminator_c = Discriminator(input_shape=fine_image_shape)
  discriminator_f = Discriminator(input_shape=fine_image_shape)

#   if pretrained_fine_encoder == True:
#     gan.fine_encoder.load_weights(f'./fine_encoder_100_weights.h5')

  if continue_training == True:
    if dual == 0:
      gan.coarse_encoder.load_weights(f'{pretrained_path}/coarse_encoder_latest_weights_dual{dual}.h5')
    elif dual == 1:
      gan.fine_encoder.load_weights(f'{pretrained_path}/fine_encoder_latest_weights_dual{dual}.h5')
    else:
      gan.fine_encoder.load_weights(f'{pretrained_path}/fine_encoder_latest_weights_dual{dual}.h5')
      gan.coarse_encoder.load_weights(f'{pretrained_path}/coarse_encoder_latest_weights_dual{dual}.h5')

    if refinement == True:
      gan.refinement_network.load_weights(f'{pretrained_path}/refinement_network_latest_weights_dual{dual}.h5')
    
    gan.decoder.load_weights(f'{pretrained_path}/decoder_latest_weights_dual{dual}.h5')

  return gan

   
if __name__ == '__main__':
  # Instantiate the parser
  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--name', type=str, default='gan',
                      help='name of project')

  parser.add_argument('--continue_training', type=bool, default=True,
                       help='continue training: load the latest model')

  parser.add_argument('--test_path', type=str,
                      help='Path of test images paths')
  parser.add_argument('--dual', type=int, default=2,
                      help='duality of encoder,0 for just coarse encoder,1 for just fine encoder,2 for both of them')
  parser.add_argument('--refinement', type=bool, default=True,
                      help='flag for refinement network')

  parser.add_argument('--save_path', type=str, default='./example/results',
                      help='outputs saving direction')
  parser.add_argument('--pretrained_path', type=str, default='./weights',
                      help='outputs saving direction')
  args = parser.parse_args()


  test_path = args.test_path

  fine_image_shape = (256,256,3)
  coarse_image_shape = (256,256,4)
  continue_training = args.continue_training

  dual = args.dual
  refinement = args.refinement
  save_path = args.save_path
  pretrained_path = args.pretrained_path
  try:
    os.mkdir(save_path)
  except:
    pass


  device = torch.device("cuda")
  edge_generator = EdgeGenerator(use_spectral_norm=True)
  edge_weights = torch.load('./EdgeModel_gen.pth')
  edge_generator.load_state_dict(edge_weights['generator'])
  edge_generator.to(device)


  gan = build_networks(pretrained_path=pretrained_path, image_shape = (256,256),
                                      continue_training = continue_training,
                                      dual=dual, refinement=refinement)
  
  list_name = listdir(test_path + '/' + 'images')

  for name in range(list_name):
    
    im = Image.open(test_path + '/' + 'images' + name)
    im = im.resize((256,256))
    im = np.array(im).astype(float) / 255.0
    mask = Image.open(test_path + '/' + 'images' + name)
    mask = mask.resize((256,256))
    mask = np.array(mask).astype(float) / 255.0
    image = np.expand_dims(im,axis=0)
    mask = np.expand_dims(mask,axis=0)
    img_gray = rgb2gray(im)
    mask_gray = rgb2gray(mask).astype(bool)
    edge = canny(img_gray,sigma=1,mask=mask_gray).astype(float)
    mask_gray = torch.Tensor.unsqueeze(torch.tensor(mask_gray.astype(float)),0).float().cuda()
    edge = torch.Tensor.unsqueeze(torch.tensor(edge),0).float().cuda()
    img_gray = torch.Tensor.unsqueeze(torch.tensor(img_gray),0).float().cuda()
    edges_masked = torch.Tensor.unsqueeze(edge * mask_gray,0)
    images_masked = torch.Tensor.unsqueeze((img_gray * mask_gray) + (1 - mask_gray),0)
    inputs = torch.cat((images_masked, edges_masked, torch.Tensor.unsqueeze(1 - mask_gray,0)), dim=1)
    output = edge_generator(inputs)
    pred_edge = torch.Tensor.squeeze(output).cpu().detach().numpy()
    scatter = np.expand_dims(pred_edge,axis=-1)
    scatter = np.expand_dims(scatter,axis=0)
    
    input1 = image*mask

    input2 = tf.concat([input1,scatter],-1)

    if dual == 0:
      f = gan.coarse_encoder(input2)
      
    elif dual == 1:
      f = gan.fine_encoder(input1)
    
    elif dual == 2:
      f1 = gan.fine_encoder(input1)
      f2 = gan.coarse_encoder(input2)
    
      f = Concatenate()([f2,f1])

    output_c = gan.decoder(f)

    if refinement == True:
      output_f = gan.refinement_network(tf.concat([output_c*(1-mask) + image*(mask),scatter],-1))
    
    out_c_path = save_path + 'out_c_' + name
    org_path = save_path + 'org_' + name
    out_f_path = save_path + 'out_f_' + name
    masked_path = save_path + 'masked_' + name
    
    out_c_img = Image.fromarray(np.uint8(output_c[0].numpy()*255.0)).convert('RGB')
    out_f_img = Image.fromarray(np.uint8(output_f[0].numpy()*255.0)).convert('RGB')
    org_img = Image.fromarray(np.uint8(img[0].numpy()*255.0)).convert('RGB')
    masked_img = Image.fromarray(np.uint8(img[0]*msk[0].numpy()*255.0)).convert('RGB')
    
    out_c_img.save(out_c_path)
    out_f_img.save(out_f_path)
    org_img.save(org_path)
    masked_img.save(masked_path)
