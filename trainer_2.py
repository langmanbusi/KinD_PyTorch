"""
Trainer for KinD
"""
import os
import math
import json
import logging
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import atexit
import time
from tqdm import tqdm
import random
import numpy as np
import torch.nn.functional as F
from utils import MSE, SSIM, PSNR, LPIPS
from loss import mutual_consistency, illumination_smoothness, grad_loss, SSIM_loss, grad_loss_1
from PIL import Image


class Base_Trainer:
    """

    """
    def __init__(self, model, ckpt_dir,
                 train_dataloader, valid_dataloader,
                 lr, val_every_epoch, gpu_id,
                 phase_name, phase_epoch, vis_dir):

        self.model = model
        self.ckpt_dir = ckpt_dir
        self.vis_dir = vis_dir
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr = lr
        self.val_every_epoch = val_every_epoch
        if gpu_id is not None:
            gpu_id = 0
            self.gpu_id = torch.device('cuda:' + str(gpu_id))
            self.model = self.model.to(self.gpu_id)

        self.phase_name = phase_name
        self.phase_epoch = phase_epoch

        self.MSEloss = torch.nn.MSELoss()
        self.SSIMloss = SSIM_loss()
        # Create the optimizers
        self.optimizer_Decom = optim.Adam(self.model.DecomNet.parameters(), lr=self.lr[0], betas=(0.9, 0.999))
        self.optimizer_Restore = optim.Adam(self.model.RestoreNet.parameters(), lr=self.lr[1], betas=(0.9, 0.999))
        self.optimizer_Relight = optim.Adam(self.model.RelightNet.parameters(), lr=self.lr[2], betas=(0.9, 0.999))

        if ckpt_dir:
            tsbd_dir = os.path.join(ckpt_dir, 'tensorboard/1')
            self.writer = SummaryWriter(log_dir=tsbd_dir)
            atexit.register(self.cleanup)

    def cleanup(self):
        self.writer.close()

    def save(self, iter_num, ckpt_dir, train_phase):
        save_dir = ckpt_dir + '/' + train_phase + '/'
        save_name= save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if train_phase == 'Decom':
            torch.save(self.model.DecomNet.state_dict(), save_name)
        elif train_phase == 'Restore':
            torch.save(self.model.RestoreNet.state_dict(), save_name)
        elif train_phase == 'Relight':
            torch.save(self.model.RelightNet.state_dict(),save_name)

    def load(self, ckpt_dir, train_phase):
        load_dir   = ckpt_dir + '/' + train_phase + '/'
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts)>0:
                load_ckpt  = load_ckpts[-1]
                global_step= int(load_ckpt[:-4])
                ckpt_dict  = torch.load(load_dir + load_ckpt)
                if train_phase == 'Decom':
                    self.model.DecomNet.load_state_dict(ckpt_dict)
                elif train_phase == 'Restore':
                    self.model.RestoreNet.load_state_dict(ckpt_dict)
                elif train_phase == 'Relight':
                    self.model.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0  # TODO 异常

    def gradient(self, input_tensor, direction):
        smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def calculate_ratio(self, input_low, input_high):
        N, C, H, W = input_low.shape
        batch_ratio = torch.zeros(N, 1, H, W).to(self.gpu_id)  # TODO 放在初始化里
        for i in range(N):
            ratio = torch.mean(torch.div(input_low, input_high + 0.0001))
            i_low_data_ratio = torch.ones(H, W).to(self.gpu_id) * (1/ratio + 0.0001)
            i_low_ratio_expand = torch.unsqueeze(i_low_data_ratio, dim=0)
            batch_ratio[i, :, :, :] = i_low_ratio_expand

        return batch_ratio

    def calculate_loss(self, input_low, input_high, R_low, I_low, R_high, I_high, I_delta, I_low_3, I_high_3, I_delta_3,
                       train_phase, ratio, R_denoise=None):
        # Compute losses
        if train_phase == 'Decom':
            # recon_loss_low = F.l1_loss(R_low * I_low_3, input_low)
            # recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)
            # recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, input_low)
            # recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)
            # equal_R_loss = F.l1_loss(R_low, R_high.detach())
            # Ismooth_loss_low = self.smooth(I_low, R_low)
            # Ismooth_loss_high = self.smooth(I_high, R_high)
            # loss = recon_loss_low + recon_loss_high + 0.001 * recon_loss_mutal_low + 0.001 * recon_loss_mutal_high + \
            #          0.1 * Ismooth_loss_low + 0.1 * Ismooth_loss_high + 0.01 * equal_R_loss

            recon_loss_low = F.l1_loss(R_low * I_low_3, input_low)
            recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)
            i_mutual_loss = mutual_consistency(I_low, I_high)
            i_mutual_loss_high = illumination_smoothness(I_high, input_high)
            i_mutual_loss_low = illumination_smoothness(I_low, input_low)
            equal_R_loss = F.l1_loss(R_low, R_high.detach())
            loss = recon_loss_low + recon_loss_high + 0.01 * equal_R_loss + 0.2 * i_mutual_loss \
               + 0.15 * i_mutual_loss_high + 0.15 * i_mutual_loss_low

        if train_phase == "Restore":
            denoise_ssim_loss = 1 - self.SSIMloss(R_denoise, R_high)
            denoise_mse_loss = self.MSEloss(R_denoise, R_high)
            denoise_grad_loss = grad_loss(R_denoise, R_high)
            # loss = denoise_mse_loss + 0.5 * denoise_grad_loss + 0.5 * denoise_ssim_loss
            loss = denoise_mse_loss + denoise_grad_loss + denoise_ssim_loss

        if train_phase == 'Relight':
            relight_mse_loss = self.MSEloss(I_delta, I_high)
            relight_grad_loss = grad_loss_1(I_delta, I_high)
            loss = relight_grad_loss + relight_mse_loss

            # relight_mse_loss = self.MSEloss(R_denoise * I_delta_3, input_high)
            # relight_grad_loss = grad_loss(R_denoise * I_delta_3, input_high)
            # relight_ssim_loss = 1 - self.SSIMloss(R_denoise * I_delta_3, input_high)
            #
            # relight_ratio_loss = torch.abs(torch.mean(ratio * input_low - I_delta_3 * R_denoise))
            # loss = relight_mse_loss + 0.5 * relight_grad_loss + 0.5 * relight_ssim_loss + 0.1 * relight_ratio_loss

            # relight_loss = F.l1_loss(R_denoise * I_delta_3, input_high)
            # Ismooth_loss_delta = self.smooth(I_delta, R_denoise)
            # loss = relight_loss + 3 * Ismooth_loss_delta

        '''self.output_R_low = R_low.detach().cpu()
        self.output_I_low = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S = self.output_R_low * self.output_I_delta
        self.input_high = input_high.detach().cpu()'''

        return loss

    def train(self):
        numbatch = len(self.train_dataloader)

        for phase in range(len(self.phase_name)):
            epoch = self.phase_epoch[phase]
            train_phase = self.phase_name[phase]
            num_iter = numbatch * epoch

            # set lr
            if train_phase == "Decom":
                lr_list = self.lr[0] * np.ones([epoch])
                # lr_list[300:] = lr_list[0] / 10.0
            elif train_phase == "Restore":
                lr_list = self.lr[1] * np.ones([epoch])
                lr_list[800:] = lr_list[0] / 10.0
            elif train_phase == "Relight":
                lr_list = self.lr[2] * np.ones([epoch])
                # lr_list[300:] = lr_list[0] / 10.0

            # Initialize a network if its checkpoint is available
            load_model_status, global_step = self.load(self.ckpt_dir, train_phase)
            if load_model_status:
                iter_num = global_step
                start_epoch = global_step // numbatch
                print("Model restore success!")
            else:
                iter_num = 0
                start_epoch = 0
                print("No pretrained model to restore!")

            print("Start training for phase %s, with start epoch %d start iter %d : " %
                  (train_phase, start_epoch, iter_num))

            # start_time = time.time()
            image_id = 0
            for epoch in range(start_epoch, epoch):
                lr = lr_list[epoch]
                loss_all = 0
                # Adjust learning rate
                if train_phase == "Decom":
                    for param_group in self.optimizer_Decom.param_groups:
                        param_group['lr'] = lr
                elif train_phase == "Restore":
                    for param_group in self.optimizer_Restore.param_groups:
                        param_group['lr'] = lr
                elif train_phase == "Relight":
                    for param_group in self.optimizer_Decom.param_groups:    # TODO lr_scheduler
                        param_group['lr'] = lr
                with tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)) as tepoch:
                    for batch_id, item in tepoch:
                        tepoch.set_description(f"Train {train_phase} : Epoch {epoch}")
                        '''# Generate training data for a batch
                        batch_input_low = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                        batch_input_high= np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")

                        for patch_id in range(batch_size):
                            # Load images
                            train_low_img = Image.open(train_low_data_names[image_id])
                            train_low_img = np.array(train_low_img, dtype='float32')/255.0
                            train_high_img= Image.open(train_high_data_names[image_id])
                            train_high_img= np.array(train_high_img, dtype='float32')/255.0
                            # Take random crops
                            h, w, _        = train_low_img.shape
                            x = random.randint(0, h - patch_size)
                            y = random.randint(0, w - patch_size)
                            train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                            train_high_img= train_high_img[x: x + patch_size, y: y + patch_size, :]
                            # Data augmentation
                            if random.random() < 0.5:
                                train_low_img = np.flipud(train_low_img)
                                train_high_img= np.flipud(train_high_img)
                            if random.random() < 0.5:
                                train_low_img = np.fliplr(train_low_img)
                                train_high_img= np.fliplr(train_high_img)
                            rot_type = random.randint(1, 4)
                            if random.random() < 0.5:
                                train_low_img = np.rot90(train_low_img, rot_type)
                                train_high_img= np.rot90(train_high_img, rot_type)
                            # Permute the images to tensor format
                            train_low_img = np.transpose(train_low_img, (2, 0, 1))
                            train_high_img= np.transpose(train_high_img, (2, 0, 1))
                            # Prepare the batch
                            batch_input_low[patch_id, :, :, :] = train_low_img
                            batch_input_high[patch_id, :, :, :]= train_high_img
                            self.input_low = batch_input_low
                            self.input_high= batch_input_high

                            image_id = (image_id + 1) % len(train_low_data_names)
                            if image_id == 0:
                                tmp = list(zip(train_low_data_names, train_high_data_names))
                                random.shuffle(list(tmp))
                                train_low_data_names, train_high_data_names = zip(*tmp)'''

                        input_low = item[0].to(self.gpu_id)
                        input_high = item[1].to(self.gpu_id)
                        ratio = self.calculate_ratio(input_low, input_high)

                        # Feed-Forward to the network and obtain loss
                        input_low, input_high, R_low, I_low, R_high, I_high, I_delta, I_low_3, I_high_3, I_delta_3, \
                        R_denoise = self.model.forward(input_low, input_high, ratio)

                        loss = self.calculate_loss(input_low, input_high, R_low, I_low, R_high, I_high, I_delta,
                                                   I_low_3, I_high_3, I_delta_3, train_phase, ratio, R_denoise)

                        if train_phase == "Decom":
                            self.optimizer_Decom.zero_grad()
                            loss.backward()
                            self.optimizer_Decom.step()
                            loss = loss.item()
                        elif train_phase == "Restore":
                            self.optimizer_Restore.zero_grad()
                            loss.backward()
                            self.optimizer_Restore.step()
                            loss = loss.item()
                        elif train_phase == "Relight":
                            self.optimizer_Relight.zero_grad()
                            loss.backward()
                            self.optimizer_Relight.step()
                            loss = loss.item() # TODO 增加网络块

                        iter_num += 1
                        loss_all += loss
                        loss = loss_all / (batch_id + 1)
                        tepoch.set_postfix({'Iter': '{:d}/{:d}'.format(iter_num, num_iter),
                                            'Loss': '{:.4f}'.format(loss)})     # TODO 改成tqdm的输出

                # Evaluate the model and save a checkpoint file for it
                if (epoch + 1) % self.val_every_epoch == 0:
                    self.valid(train_phase, epoch + 1, ) # TODO 增加评价指标
                    self.save(iter_num, self.ckpt_dir, train_phase) # TODO 改ckpt命名格式

                self.writer.add_scalar(train_phase + '/Loss', loss, epoch)

            print("Finished training for phase %s." % train_phase)

    def valid(self, train_phase, epoch_num):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        save_val = False
        MSE_output = []
        SSIM_output = []
        PSNR_output = []
        LPIPS_output = []
        N = len(self.valid_dataloader)

        with torch.no_grad():
            with tqdm(enumerate(self.valid_dataloader), total=N) as tepoch:
                for id, item in tepoch:
                    tepoch.set_description(f"Valid {train_phase} : Epoch {epoch_num}")

                    input_low = item[0].to(self.gpu_id)
                    input_high = item[1].to(self.gpu_id)
                    ratio = self.calculate_ratio(input_low, input_high)

                    # Feed-Forward to the network and obtain loss
                    input_low, input_high, R_low, I_low, R_high, I_high, I_delta, I_low_3, I_high_3, I_delta_3, \
                    R_denoise = self.model.forward(input_low, input_high, ratio)

                    loss = self.calculate_loss(input_low, input_high,
                                               R_low, I_low,
                                               R_high, I_high, I_delta,
                                               I_low_3, I_high_3, I_delta_3, train_phase, ratio, R_denoise)

                    output = I_delta_3 * R_denoise

                    # metric
                    mse = torch.nn.MSELoss(reduction='mean')
                    MSE_output += [mse(output, input_high).item()]
                    ssim = SSIM_loss()
                    SSIM_output += [ssim(output, input_high).item()]
                    PSNR_output += [PSNR(MSE_output[-1])]
                    LPIPS_1 = LPIPS(output, input_high, self.gpu_id)
                    LPIPS_2 = torch.mean(torch.squeeze(LPIPS_1))
                    LPIPS_4 = float(LPIPS_2)
                    # LPIPS_4 = float(LPIPS_3)
                    # LPIPS_batch = float(torch.squeeze(LPIPS(output.cpu(), input_high.cpu())))
                    LPIPS_output += [LPIPS_4]

                    if save_val:
                        cat_image = np.concatenate([input_low, output, input_high], axis=2)
                        cat_image = np.transpose(cat_image, (1, 2, 0))
                        # print(cat_image.shape)
                        im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
                        filepath = os.path.join(self.vis_dir, 'eval_%s_%d_%d.png' %
                                                (train_phase, id + 1, epoch_num))
                        im.save(filepath[:-4] + '.jpg')

            MSE_avg = sum(MSE_output) / N * 255 * 255 / 1000
            SSIM_avg = sum(SSIM_output) / N
            PSNR_avg = sum(PSNR_output) / N
            LPIPS_avg = sum(LPIPS_output) / N

            self.writer.add_scalar(train_phase + '/MSE', MSE_avg, epoch_num)
            self.writer.add_scalar(train_phase + '/SSIM', SSIM_avg, epoch_num)
            self.writer.add_scalar(train_phase + '/PSNR', PSNR_avg, epoch_num)
            self.writer.add_scalar(train_phase + '/LPIPS', LPIPS_avg, epoch_num)

            print('MSE = ', MSE_avg,
                  'SSIM = ', SSIM_avg,
                  'PSNR = ', PSNR_avg,
                  'LPIPS = ', LPIPS_avg,
                  train_phase + '_loss = ', loss.item())
            '''if train_phase == "Decom":
                self.model.forward(input_low_eval, input_high_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                if save_eval:
                    input = np.squeeze(input_low_eval)
                    result_1 = np.squeeze(result_1)
                    result_2 = np.squeeze(result_2)
                    cat_image = np.concatenate([input, result_1, result_2], axis=2)
            if train_phase == "Relight":
                self.forward(input_low_eval, input_high_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                result_GT = self.input_high
                if save_eval:
                    input = np.squeeze(input_low_eval)
                    result_1 = np.squeeze(result_1)
                    result_2 = np.squeeze(result_2)
                    result_3 = np.squeeze(result_3)
                    result_4 = np.squeeze(result_4)
                    cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)
                # TODO 增加评价指标，不保存图片

                # metric
                MSE_output += [MSE(result_4.numpy(), result_GT.numpy())]
                SSIM_output += [SSIM(result_4.numpy(), result_GT.numpy())]
                PSNR_output += [PSNR(result_4.numpy(), result_GT.numpy())]
                # LPIPS_output += [LPIPS(result_4, result_GT)]

            if save_eval:
                cat_image = np.transpose(cat_image, (1, 2, 0))
                # print(cat_image.shape)
                im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
                filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png' %
                                        (train_phase, idx + 1, epoch_num))
                im.save(filepath[:-4] + '.jpg')

        if self.train_phase == 'Relight':
            # metric
            MSE_avg = sum(MSE_output) / N * 255 * 255 / 1000
            SSIM_avg = sum(SSIM_output) / N
            PSNR_avg = sum(PSNR_output) / N
            LPIPS_avg = sum(LPIPS_output) / N

            self.writer.add_scalar(self.train_phase + '/MSE', MSE_avg, epoch_num)
            self.writer.add_scalar(self.train_phase + '/SSIM', SSIM_avg, epoch_num)
            self.writer.add_scalar(self.train_phase + '/PSNR', PSNR_avg, epoch_num)
            self.writer.add_scalar(self.train_phase + '/LPIPS', LPIPS_avg, epoch_num)  # TODO 改文件夹

            print('MSE = ', MSE_avg,
                  'SSIM = ', SSIM_avg,
                  'PSNR = ', PSNR_avg,
                  'LPIPS = ', LPIPS_avg)'''
