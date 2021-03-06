# -*- coding: utf-8 -*-

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

# import face_alignment
from skimage import io
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
import torch
from networks_edge import EdgeGenerator

# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

def im_file_to_tensor(img,mask,scatter):
  def _im_file_to_tensor(img,mask,scatter):
    path = f"{img.numpy().decode()}"
    im = Image.open(path)
    im = im.resize((256,256))
    im = np.array(im).astype(float) / 255.0
    path = f"{mask.numpy().decode()}"
    mask = Image.open(path)
    mask = mask.resize((256,256))
    mask = np.array(mask).astype(float) / 255.0
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
    return im, mask , scatter
  return tf.py_function(_im_file_to_tensor, 
                        inp=(img,mask,scatter), 
                        Tout=(tf.float32,tf.float32,tf.float32))

def Create_dataset(images_path,masks_path,batch_size = 8):
  f = open(images_path,'r')
  img_paths = f.read()
  img_paths = img_paths.split(sep='\n')
  f = open(masks_path,'r')
  mask_paths = f.read()
  mask_paths = mask_paths.split(sep='\n')
  mask_paths.pop()
  img_paths.pop()

  img_paths = np.array(img_paths)
  mask_paths = np.array(mask_paths)

  indx = np.asarray(range(len(img_paths)))
  np.random.shuffle(indx)
  img_paths = img_paths[indx]
  mask_paths = mask_paths[indx]

  #step 1
  img_names = tf.constant(img_paths)
  mask_names = tf.constant(mask_paths)
  scatter = tf.constant([0]*(len(img_paths)))

  # step 2: create a dataset returning slices of `filenames`
  dataset = tf.data.Dataset.from_tensor_slices((img_names,mask_names,scatter))

  dataset = dataset.map(im_file_to_tensor)
  batch_dataset = dataset.batch(batch_size)

  return batch_dataset

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

def build_networks (save_path, image_shape= (256,256) ,continue_training = False,dual=2,
                    refinement=True,pretrained_fine_encoder=True, attention=True):
  gan = GAN(image_shape= image_shape,dual=dual,refinement=refinement,
            pretrained_fine_encoder=pretrained_fine_encoder,attention=attention)
  discriminator_c = Discriminator(input_shape=fine_image_shape)
  discriminator_f = Discriminator(input_shape=fine_image_shape)

  if pretrained_fine_encoder == True:
    gan.fine_encoder.load_weights(f'./fine_encoder_100_weights.h5')

  if continue_training == True:
    if dual == 0:
      gan.coarse_encoder.load_weights(f'{save_path}/coarse_encoder_latest_weights_dual{dual}.h5')
    elif dual == 1:
      gan.fine_encoder.load_weights(f'{save_path}/fine_encoder_latest_weights_dual{dual}.h5')
    else:
      gan.fine_encoder.load_weights(f'{save_path}/fine_encoder_latest_weights_dual{dual}.h5')
      gan.coarse_encoder.load_weights(f'{save_path}/coarse_encoder_latest_weights_dual{dual}.h5')

    if refinement == True:
      gan.refinement_network.load_weights(f'{save_path}/refinement_network_latest_weights_dual{dual}.h5')
    
    gan.decoder.load_weights(f'{save_path}/decoder_latest_weights_dual{dual}.h5')
    discriminator_c.load_weights(f'{save_path}/discriminator_c_latest_weights_dual{dual}.h5')
    discriminator_f.load_weights(f'{save_path}/discriminator_f_latest_weights_dual{dual}.h5')

  return gan, discriminator_c, discriminator_f

def perc_model (vgg_model):
  
  output1 = vgg_model.layers[1].output
  output2 = vgg_model.layers[4].output
  output3 = vgg_model.layers[7].output
  output4 = vgg_model.layers[12].output
  output5 = vgg_model.layers[17].output
  perceptual_model = keras.Model(inputs=vgg_model.input,outputs=[output1,output2,output3,output4,output5])
  return perceptual_model

def style_model (vgg_model):
  
  # output1 = vgg_model.layers[1].output
  output2 = vgg_model.layers[5].output
  output3 = vgg_model.layers[10].output
  output4 = vgg_model.layers[15].output
  output5 = vgg_model.layers[18].output
  style_model = keras.Model(inputs=vgg_model.input,outputs=[output2,output3,output4,output5])
  return style_model

def gram_matrix(features):
  f_size = [features.shape[0],features.shape[-1],features.shape[1]*features.shape[2]]
  features = tf.reshape(features,f_size)
  result = tf.matmul(features, features, transpose_b=True)
  result = result / (features.shape[-1]*((features.shape[1]*features.shape[2])**2))
  return result

def perc_style_loss(image: tf.Tensor,
                    output: tf.Tensor,
                    perceptual_model: tf.keras.Model,
                    style_model: tf.keras.Model) -> tf.Tensor:

  image_v = keras.applications.vgg19.preprocess_input(image * 255.0)
  output_v = keras.applications.vgg19.preprocess_input(output * 255.0)
  
  output_f1, output_f2, output_f3, output_f4, output_f5 = perceptual_model(output_v) 
  image_f1, image_f2, image_f3, image_f4, image_f5 = perceptual_model(image_v)

  perc_f1 = tf.reduce_mean(tf.reduce_mean(tf.abs(image_f1-output_f1),axis=(1,2,3)))
  perc_f2 = tf.reduce_mean(tf.reduce_mean(tf.abs(image_f2-output_f2),axis=(1,2,3)))
  perc_f3 = tf.reduce_mean(tf.reduce_mean(tf.abs(image_f3-output_f3),axis=(1,2,3)))
  perc_f4 = tf.reduce_mean(tf.reduce_mean(tf.abs(image_f4-output_f4),axis=(1,2,3)))
  perc_f5 = tf.reduce_mean(tf.reduce_mean(tf.abs(image_f5-output_f5),axis=(1,2,3)))
  perceptual_loss = perc_f1 + perc_f2 + perc_f3 + perc_f4 + perc_f5

  output_s_f2, output_s_f3, output_s_f4, output_s_f5 = style_model(output_v) 
  image_s_f2, image_s_f3, image_s_f4, image_s_f5 = style_model(image_v)

  img_gram_f2 = gram_matrix(image_s_f2)
  out_gram_f2 = gram_matrix(output_s_f2)
  img_gram_f3 = gram_matrix(image_s_f3)
  out_gram_f3 = gram_matrix(output_s_f3)
  img_gram_f4 = gram_matrix(image_s_f4)
  out_gram_f4 = gram_matrix(output_s_f4)
  img_gram_f5 = gram_matrix(image_s_f5)
  out_gram_f5 = gram_matrix(output_s_f5)


  style_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(img_gram_f3-out_gram_f3),axis=(1,2)))

  style_loss += tf.reduce_mean(tf.reduce_mean(tf.abs(img_gram_f2-out_gram_f2),axis=(1,2)))

  style_loss += tf.reduce_mean(tf.reduce_mean(tf.abs(img_gram_f4-out_gram_f4),axis=(1,2)))

  style_loss += tf.reduce_mean(tf.reduce_mean(tf.abs(img_gram_f5-out_gram_f5),axis=(1,2)))


  return perceptual_loss, style_loss


@tf.function
def train_g (gan: tf.keras.Model,
           discriminator_c: tf.keras.Model,
           discriminator_f: tf.keras.Model,
           g_opt: tf.keras.optimizers.Optimizer,
           d_opt_c: tf.keras.optimizers.Optimizer,
           d_opt_f: tf.keras.optimizers.Optimizer,
           image: tf.Tensor,mask: tf.Tensor,scatter: tf.Tensor,
           epoch: int, pre_epoch: int, dual, refinement) -> tf.Tensor:
  g_loss = 0
  d_loss = 0
  output_c = 0
  output_f = 0
  with tf.GradientTape() as tape1,tf.GradientTape() as tape2:
    input1 = image*mask

    input2 = tf.concat([input1,scatter],-1)

    if dual == 0:
      f = gan.coarse_encoder(input2,training=True)
      
    elif dual == 1:
      f = gan.fine_encoder(input1,training=True)
    
    elif dual == 2:
      f1 = gan.fine_encoder(input1,training=True)
      f2 = gan.coarse_encoder(input2,training=True)
      
      f = Concatenate()([f2,f1])

    output_c = gan.decoder(f,training=True)

    if refinement == True:
      output_f = gan.refinement_network(tf.concat([output_c*(1-mask) + image*(mask),scatter],-1),training=True)  

      perceptual_loss_f, _ = perc_style_loss(image,output_f,perceptual_model,styles_model)

      l1_loss_f = tf.reduce_mean(tf.reduce_mean(tf.abs(image-output_f),axis=(1,2,3)))
    
    l1_loss_c = tf.reduce_mean(tf.reduce_mean(tf.abs(image-output_c),axis=(1,2,3)))
    
    perceptual_loss_c, style_loss_c = perc_style_loss(image,output_c,perceptual_model,styles_model)

    if refinement == True:
      g_loss = l1_loss_c + 0.1*perceptual_loss_c  +  250 * style_loss_c + l1_loss_f + 0.1*perceptual_loss_f
    else:
      g_loss = l1_loss_c + 0.1*perceptual_loss_c  +  250 * style_loss_c 
 

  g_grads = tape1.gradient(g_loss,gan.trainable_variables)
  g_opt.apply_gradients(zip(g_grads,gan.trainable_variables))

  return g_loss , d_loss, output_c, output_f

@tf.function
def train_g_d (gan: tf.keras.Model,
           discriminator_c: tf.keras.Model,
           discriminator_f: tf.keras.Model,
           g_opt: tf.keras.optimizers.Optimizer,
           d_opt_c: tf.keras.optimizers.Optimizer,
           d_opt_f: tf.keras.optimizers.Optimizer,
           image: tf.Tensor,mask: tf.Tensor,scatter: tf.Tensor,
           epoch: int, pre_epoch: int,
           dual, refinement) -> tf.Tensor:
  g_loss = 0
  d_loss = 0
  output_c = 0
  output_f = 0
  with tf.GradientTape() as tape1,tf.GradientTape() as tape2,tf.GradientTape() as tape3:
    input1 = image*mask

    input2 = tf.concat([input1,scatter],-1)

    if dual == 0:
      f = gan.coarse_encoder(input2,training=True)
      
    elif dual == 1:
      f = gan.fine_encoder(input1,training=True)
    
    elif dual == 2:
      f1 = gan.fine_encoder(input1,training=True)
      f2 = gan.coarse_encoder(input2,training=True)
    
      f = Concatenate()([f2,f1])

    output_c = gan.decoder(f,training=True)

    if refinement == True:
      output_f = gan.refinement_network(tf.concat([output_c*(1-mask) + image*(mask),scatter],-1),training=True)      

      perceptual_loss_f, _ = perc_style_loss(image,output_f,perceptual_model,styles_model)

      l1_loss_f = tf.reduce_mean(tf.reduce_mean(tf.abs(image-output_f),axis=(1,2,3)))

    l1_loss_c = tf.reduce_mean(tf.reduce_mean(tf.abs(image-output_c),axis=(1,2,3)))
    
    perceptual_loss_c, style_loss_c = perc_style_loss(image,output_c,perceptual_model,styles_model)

    if refinement == True:
      g_loss = l1_loss_c + 0.1*perceptual_loss_c  +  250 * style_loss_c + l1_loss_f + 0.1*perceptual_loss_f
    else:
      g_loss = l1_loss_c + 0.1*perceptual_loss_c  +  250 * style_loss_c 
    
    d_fake_f = discriminator_f(output_f,training=True)
    adv_g_loss_f = -tf.reduce_mean(tf.reduce_mean(tf.math.log(tf.squeeze(d_fake_f)),axis=(1,2)))

    d_fake_c = discriminator_c(output_c,training=True)
    adv_g_loss_c = -tf.reduce_mean(tf.reduce_mean(tf.math.log(tf.squeeze(d_fake_c)),axis=(1,2)))

    d_real_c = discriminator_c(image,training=True)
    d_real_f = discriminator_f(image,training=True)

    d_loss_fake_c = tf.reduce_mean(tf.reduce_mean(tf.nn.relu(1.0 + tf.squeeze(d_fake_c)),axis=(1,2)))
    d_loss_real_c = tf.reduce_mean(tf.reduce_mean(tf.nn.relu(1.0 - tf.squeeze(d_real_c)),axis=(1,2)))

    d_loss_fake_f = tf.reduce_mean(tf.reduce_mean(tf.nn.relu(1.0 + tf.squeeze(d_fake_f)),axis=(1,2)))
    d_loss_real_f = tf.reduce_mean(tf.reduce_mean(tf.nn.relu(1.0 - tf.squeeze(d_real_f)),axis=(1,2)))

    d_loss = d_loss_fake_c + d_loss_real_c + d_loss_fake_f + d_loss_real_f

    d_g_loss = g_loss + 0.1 * adv_g_loss_f + 0.1 * adv_g_loss_c


  d_grads_f = tape2.gradient(d_loss,discriminator_f.trainable_variables)
  d_opt_f.apply_gradients(zip(d_grads_f,discriminator_f.trainable_variables))

  d_grads_c = tape3.gradient(d_loss,discriminator_c.trainable_variables)
  d_opt_c.apply_gradients(zip(d_grads_c,discriminator_c.trainable_variables))

  g_grads = tape1.gradient(d_g_loss,gan.trainable_variables)
  g_opt.apply_gradients(zip(g_grads,gan.trainable_variables))
  

  return g_loss , d_loss ,output_c, output_f

def high_pass_x_y(image):
  x_var = image - tf.roll(image,shift=-1,axis=1)
  y_var = image - tf.roll(image,shift=-1,axis=2)

  return x_var, y_var

@tf.function
def validation_batch (gan: tf.keras.Model,
                      img, msk,sc,dual,refinement) -> tf.Tensor:

  input1 = img*msk

  input2 = tf.concat([input1,sc],-1)

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
    output_f = gan.refinement_network(tf.concat([output_c*(1-msk) + img*(msk),sc],-1))   

    l1_loss_f = tf.reduce_sum(tf.reduce_mean(tf.abs(img-output_f),axis=(1,2,3))) 

    mse_f = tf.reduce_mean((output_f*255.0 - img*255.0) ** 2,axis=(1,2,3))

    psnr_f = 20 * tf.experimental.numpy.log10(255.0 / tf.sqrt(mse_f))

    PSNR_f = tf.reduce_sum(psnr_f,axis=0)

  l1_loss_c = tf.reduce_sum(tf.reduce_mean(tf.abs(img-output_c),axis=(1,2,3)))
  
  gx , gy = high_pass_x_y(output_c)
  grad_norm2 = gx**2 + gy**2
  TV = tf.reduce_sum(tf.reduce_mean(tf.sqrt(grad_norm2),axis=(1,2,3)),axis=0)

  mse_c = tf.reduce_mean((output_c*255.0 - img*255.0) ** 2,axis=(1,2,3))

  psnr_c = 20 * tf.experimental.numpy.log10(255.0 / tf.sqrt(mse_c))

  PSNR_c = tf.reduce_sum(psnr_c,axis=0)


  if refinement == False:
    out_dic = {'total_l1_loss_c': l1_loss_c,
              'PSNR_c' : PSNR_c,
              'TV_loss' : TV}
  else:

    out_dic = {'total_l1_loss_c': l1_loss_c,
              'total_l1_loss_f': l1_loss_f,
              'PSNR_c' : PSNR_c,
              'PSNR_f' : PSNR_f,
              'TV_loss' : TV}   

  return out_dic

def validation (gan: tf.keras.Model,
                val_dataset,dual,refinement) -> tf.Tensor:

  total_l1_loss_c = 0
  total_l1_loss_f = 0
  TV = 0
  PSNR_c = 0
  PSNR_f = 0
  total_size = 0
  for s, (img,msk,sc) in enumerate(val_dataset):
    total_size += img.shape[0]
    out_dic_batch = validation_batch(gan = gan,
                                     img = img, msk = msk,
                                     sc = sc,refinement =refinement,dual=dual)
    if refinement == False:
      PSNR_c += out_dic_batch['PSNR_c']
      TV += out_dic_batch['TV_loss']
      total_l1_loss_c += out_dic_batch['total_l1_loss_c']
    
      out_dic = {'total_l1_loss_c': total_l1_loss_c.numpy()/total_size,
                'PSNR_c' : PSNR_c.numpy()/total_size,
                'TV_loss' : TV.numpy()/total_size}
    else:
      PSNR_c += out_dic_batch['PSNR_c']
      PSNR_f += out_dic_batch['PSNR_f']
      TV += out_dic_batch['TV_loss']
      total_l1_loss_c += out_dic_batch['total_l1_loss_c']
      total_l1_loss_f += out_dic_batch['total_l1_loss_f']
    
      out_dic = {'total_l1_loss_c': total_l1_loss_c.numpy()/total_size,
                'total_l1_loss_f': total_l1_loss_f.numpy()/total_size,
                'PSNR_c' : PSNR_c.numpy()/total_size,
                'PSNR_f' : PSNR_f.numpy()/total_size,
                'TV_loss' : TV.numpy()/total_size}

  return out_dic

if __name__ == '__main__':
  # Instantiate the parser
  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--name', type=str, default='gan',
                      help='name of project')

  parser.add_argument('--batch_size', type=int, default=8,
                      help='batch size of training and evaluation')
  parser.add_argument('--epochs', type=int, default=120,
                      help='number epochs of training and evaluation')
  parser.add_argument('--pre_epoch', type=int, default=10,
                      help='number epochs without discriminator')
  parser.add_argument('--initial_epoch', type=int, default=1,
                      help='initial_epoch')
  parser.add_argument('--continue_training', type=bool, default=False,
                       help='continue training: load the latest model')

  parser.add_argument('--train_images_path', type=str,
                      help='Path of text file of train images paths')
  parser.add_argument('--train_masks_path', type=str,
                      help='Path of text file of train masks paths')
  parser.add_argument('--val_images_path', type=str,
                      help='Path of text file of val images paths')
  parser.add_argument('--val_masks_path', type=str,
                      help='Path of text file of val images paths')
  parser.add_argument('--dual', type=int, default=2,
                      help='duality of encoder,0 for just coarse encoder,1 for just fine encoder,2 for both of them')
  parser.add_argument('--refinement', type=bool, default=True,
                      help='flag for refinement network')

  parser.add_argument('--pretrained_fine_encoder', type=bool, default=False,
                      help='flag for refinement network')
  parser.add_argument('--attention', type=bool, default=True,
                      help='flag for attention block')
  parser.add_argument('--save_path', type=str, default='./gan',
                      help='model saving direction')
  parser.add_argument('--run_id', type=str,
                      help='run id of Wandb')
  args = parser.parse_args()


  val_images_path_txt = args.val_images_path
  train_images_path_txt = args.train_images_path
  val_masks_path_txt = args.val_masks_path
  train_masks_path_txt = args.train_masks_path

  fine_image_shape = (256,256,3)
  coarse_image_shape = (256,256,4)
  batch_size = args.batch_size
  val_batch_size = args.batch_size
  epochs = args.epochs
  pre_epoch = args.pre_epoch
  initial_epoch = args.initial_epoch
  continue_training = args.continue_training
  if args.initial_epoch > 1:
    continue_training = True

  dual = args.dual
  refinement = args.refinement
  pretrained_fine_encoder = args.pretrained_fine_encoder
  save_path = args.save_path
  attention = args.attention
  try:
    os.mkdir(save_path)
  except:
    pass

  wandb.login()

  # 1. Start a W&B run
  if continue_training == False:
    wandb.init(project=args.name, entity='amiretefaghi')
  else:
    wandb.init(project=args.name, entity='amiretefaghi',resume=args.run_id)
  
  g_learning_rate = 1e-4
  d_learning_rate = 5e-5

  configs = {
                "g_learning_rate": g_learning_rate,
                "d_learning_rate": d_learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                # "log_step": epochs,
                # "val_log_step": 50,
                "architecture": "CNN",
                "dataset": "CelebA"
            }

  config = wandb.config

  device = torch.device("cuda")
  edge_generator = EdgeGenerator(use_spectral_norm=True)
  edge_weights = torch.load('./EdgeModel_gen.pth')
  edge_generator.load_state_dict(edge_weights['generator'])
  edge_generator.to(device)

  train_dataset = Create_dataset(train_images_path_txt,
                                train_masks_path_txt,
                                batch_size = batch_size)
  val_dataset = Create_dataset(val_images_path_txt,
                              val_masks_path_txt,
                              batch_size = val_batch_size)

  gan, discriminator_c, discriminator_f = build_networks(save_path=save_path, image_shape = (256,256),
                                      continue_training = continue_training,
                                      dual=dual, refinement=refinement,
                                      pretrained_fine_encoder = pretrained_fine_encoder)

  vgg19_model = keras.applications.VGG19(include_top=False,input_shape=fine_image_shape)

  g_opt = keras.optimizers.Adam(g_learning_rate)
  d_opt_f = keras.optimizers.Adam(d_learning_rate)
  d_opt_c = keras.optimizers.Adam(d_learning_rate)

  perceptual_model = perc_model(vgg19_model)
  styles_model = style_model(vgg19_model)

  val_out_dic = validation(gan = gan,
                          val_dataset = val_dataset,dual=dual, refinement=refinement)

  print(f'validation loss before start training : {val_out_dic}')

  if args.refinement == False:
    val_PSNR_c = [val_out_dic['PSNR_c']]
    val_L1_loss_c = [val_out_dic['total_l1_loss_c']]
    val_TV_loss = [val_out_dic['TV_loss']]
  else:
    val_PSNR_c = [val_out_dic['PSNR_c']]
    val_PSNR_f = [val_out_dic['PSNR_f']]
    val_L1_loss_c = [val_out_dic['total_l1_loss_c']]
    val_L1_loss_f = [val_out_dic['total_l1_loss_f']]
    val_TV_loss = [val_out_dic['TV_loss']]

  if args.refinement == False:
    val_PSNR_c.append(val_out_dic['PSNR_c'])
    val_L1_loss_c.append(val_out_dic['total_l1_loss_c'])
    val_TV_loss.append(val_out_dic['TV_loss'])
    wandb.log({'val_PSNR_c': val_out_dic['PSNR_c'],
                'total_l1_loss_c': val_out_dic['total_l1_loss_c']})
  else:
    val_PSNR_c.append(val_out_dic['PSNR_c'])
    val_L1_loss_c.append(val_out_dic['total_l1_loss_c'])
    val_PSNR_f.append(val_out_dic['PSNR_f'])
    val_L1_loss_f.append(val_out_dic['total_l1_loss_f'])
    val_TV_loss.append(val_out_dic['TV_loss'])
    wandb.log({'val_PSNR_c': val_out_dic['PSNR_c'],
                'val_PSNR_f': val_out_dic['PSNR_f'],
                'total_l1_loss_c': val_out_dic['total_l1_loss_c'],
                'total_l1_loss_f': val_out_dic['total_l1_loss_f']})
  g_losses = []
  d_losses = []
  dataset = val_dataset.as_numpy_iterator()

  for epoch in range(epochs):
    epoch += initial_epoch
    for step, (image,mask,scatter) in tqdm(enumerate(train_dataset)):
      if epoch <= pre_epoch:
        g_loss , d_loss, output_c, output_f = train_g(gan=gan,discriminator_c=discriminator_c,
                                discriminator_f=discriminator_f,
                                g_opt=g_opt,d_opt_c=d_opt_c,d_opt_f=d_opt_f,
                                image=image,mask=mask,scatter=scatter,
                                epoch=epoch,pre_epoch=pre_epoch,dual=dual, 
                                refinement=refinement)
      else:
        g_loss , d_loss, output_c, output_f = train_g_d(gan=gan,discriminator_c=discriminator_c,
                                    discriminator_f=discriminator_f,
                                    g_opt=g_opt,d_opt_c=d_opt_c,d_opt_f=d_opt_f,
                                    image=image,mask=mask,scatter=scatter,
                                    epoch=epoch,pre_epoch=pre_epoch,dual=dual, refinement=refinement)        
      if step%100 == 0:
        g_losses.append(g_loss)
        print(f'g loss epoch {epoch} step {step} : {g_loss} ')
        if epoch > pre_epoch:
          d_losses.append(d_loss)
          print(f'd loss epoch {epoch} step {step} : {d_loss} ')
      if epoch > pre_epoch:
        wandb.log({'epochs': epoch,
                    'g_loss': g_loss,
                    'd_loss': d_loss})
      else:
        wandb.log({'epochs': epoch,
                    'g_loss': g_loss})        
      if tf.random.uniform((1,))>0.997:  
        # print(d_fake_c)     
        # print(d_real_c)     
        if args.refinement == False:
          images_c = wandb.Image(output_c[0], caption="Coarse output")
          org_images = wandb.Image(image[0], caption="original image")
          masked_org_images = wandb.Image(image[0]*mask[0], caption="masked_original image")
          edges = wandb.Image(scatter[0], caption="edges")
          wandb.log({"images_c": images_c,
                    "org_image": org_images,
                    "masked_image": masked_org_images,
                    "edges": edges})
          
        else:
          images_c = wandb.Image(output_c[0], caption="Coarse output")
          images_f = wandb.Image(output_f[0], caption="refinement output")
          org_images = wandb.Image(image[0], caption="original image")
          masked_org_images = wandb.Image(image[0]*mask[0], caption="masked original image")
          edges = wandb.Image(scatter[0], caption="edges")
          wandb.log({"images_c": images_c,
                    "images_f": images_f,
                    "org_image": org_images,
                    "masked_image": masked_org_images,
                    "edges": edges}) 

        test_img, test_msk,test_sc = dataset.next()

        test_input1 = test_img*test_msk

        test_input2 = tf.concat([test_input1,test_sc],-1)

        if dual == 0:
          f = gan.coarse_encoder(test_input2)
          
        elif dual == 1:
          f = gan.fine_encoder(test_input1)
        
        elif dual == 2:
          f1 = gan.fine_encoder(test_input1)
          f2 = gan.coarse_encoder(test_input2)
          
          f = Concatenate()([f2,f1])
        
        test_output_c = gan.decoder(f)
        if args.refinement == True:
          test_output_f = gan.refinement_network(tf.concat([test_output_c*(1-test_msk) + test_img*test_msk,test_sc],-1))

        if args.refinement == False:
          test_images_c = wandb.Image(test_output_c[0], caption="Val Coarse output")
          test_org_images = wandb.Image(test_img[0], caption="Val original image"),
          masked_test_org_images = wandb.Image(test_img[0]*test_msk[0], caption="masked Val original image")
          val_edges = wandb.Image(test_sc[0], caption="val edges")
          wandb.log({"val images_c": test_images_c,
                    "val org_image": test_org_images,
                    "masked image": masked_test_org_images,
                    "val edges": val_edges})
        else:
          test_images_c = wandb.Image(test_output_c[0], caption="Val Coarse output")
          test_images_f = wandb.Image(test_output_f[0], caption="Val refinement output")
          test_org_images = wandb.Image(test_img[0], caption="Val original image")
          masked_test_org_images = wandb.Image(test_img[0]*test_msk[0], caption="masked Val original image")
          val_edges = wandb.Image(test_sc[0], caption="val edges")
          wandb.log({"val images_c": test_images_c,
                    "val images_f": test_images_f,
                    "val org_image": test_org_images,
                    "masked image": masked_test_org_images,
                    "val edges": val_edges})  


    val_out_dic = validation(gan = gan,
                            val_dataset = val_dataset, dual=dual, refinement=refinement)
    if args.refinement == False:
      val_PSNR_c.append(val_out_dic['PSNR_c'])
      val_L1_loss_c.append(val_out_dic['total_l1_loss_c'])
      val_TV_loss.append(val_out_dic['TV_loss'])
      wandb.log({'val_PSNR_c': val_out_dic['PSNR_c'],
                  'total_l1_loss_c': val_out_dic['total_l1_loss_c']})
    else:
      val_PSNR_c.append(val_out_dic['PSNR_c'])
      val_L1_loss_c.append(val_out_dic['total_l1_loss_c'])
      val_PSNR_f.append(val_out_dic['PSNR_f'])
      val_L1_loss_f.append(val_out_dic['total_l1_loss_f'])
      val_TV_loss.append(val_out_dic['TV_loss'])
      wandb.log({'val_PSNR_c': val_out_dic['PSNR_c'],
                  'val_PSNR_f': val_out_dic['PSNR_f'],
                  'total_l1_loss_c': val_out_dic['total_l1_loss_c'],
                  'total_l1_loss_f': val_out_dic['total_l1_loss_f']})
    
    print(f'validation loss after epoch {epoch} : {val_out_dic}')

    if epoch >= 5 and epoch%5 == 0:
      if dual == 0:
        gan.coarse_encoder.save_weights(f'{save_path}/coarse_encoder_{epoch}_weights_dual{dual}.h5')
      elif dual == 1:
        gan.fine_encoder.save_weights(f'{save_path}/fine_encoder_{epoch}_weights_dual{dual}.h5')
      else:
        gan.fine_encoder.save_weights(f'{save_path}/fine_encoder_{epoch}_weights_dual{dual}.h5')
        gan.coarse_encoder.save_weights(f'{save_path}/coarse_encoder_{epoch}_weights_dual{dual}.h5')
      if args.refinement == True:
        gan.refinement_network.save_weights(f'{save_path}/refinement_network_{epoch}_weights_dual{dual}.h5')
      gan.decoder.save_weights(f'{save_path}/decoder_{epoch}_weights_dual{dual}.h5')
      discriminator_c.save_weights(f'{save_path}/discriminator_c_{epoch}_weights_dual{dual}.h5')
      discriminator_f.save_weights(f'{save_path}/discriminator_f_{epoch}_weights_dual{dual}.h5')

    if dual == 0:
      gan.coarse_encoder.save_weights(f'{save_path}/coarse_encoder_latest_weights_dual{dual}.h5')
    elif dual == 1:
      gan.fine_encoder.save_weights(f'{save_path}/fine_encoder_latest_weights_dual{dual}.h5')
    else:
      gan.fine_encoder.save_weights(f'{save_path}/fine_encoder_latest_weights_dual{dual}.h5')
      gan.coarse_encoder.save_weights(f'{save_path}/coarse_encoder_latest_weights_dual{dual}.h5')

    gan.decoder.save_weights(f'{save_path}/decoder_latest_weights_dual{dual}.h5')
    discriminator_c.save_weights(f'{save_path}/discriminator_c_latest_weights_dual{dual}.h5')
    discriminator_f.save_weights(f'{save_path}/discriminator_f_latest_weights_dual{dual}.h5')
    if args.refinement == True:
      gan.refinement_network.save_weights(f'{save_path}/refinement_network_latest_weights_dual{dual}.h5')
