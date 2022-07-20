import os
import argparse
from glob import glob
import numpy as np
from model_KinD import RetinexNet, Mymodel
import torch
from tqdm import tqdm
from PIL import Image
from utils import MSE, SSIM, PSNR, LPIPS, PSNR_1
from loss import SSIM_loss


def main(data_dir, ckpt_dir, res_dir, gpu_id):

    test_low_data_names = glob(data_dir + 'test/low/' + '*.*')
    test_high_data_names = glob(data_dir + 'test/high/' + '*.*')
    test_low_data_names.sort()
    test_high_data_names.sort()
    use_seg = False      # 是否使用语义先验
    print('Number of evaluation images: %d' % len(test_low_data_names))
    phase_name = ['Decom', 'Restore', 'Relight']

    model = Mymodel(gpu_id)
    predict(model, test_low_data_names, test_high_data_names, res_dir=res_dir, ckpt_dir=ckpt_dir, save_predict=False,
            gpu_id=gpu_id, phase_name=phase_name)


def calculate_ratio(input_low, input_high, gpu_id):
    N, C, H, W = input_low.shape
    batch_ratio = torch.zeros(N, 1, H, W).to(gpu_id)  # TODO 放在初始化里
    for i in range(N):
        ratio = torch.mean(torch.div(input_low, input_high + 0.0001))
        i_low_data_ratio = torch.ones(H, W).to(gpu_id) * (1/ratio + 0.0001)
        i_low_ratio_expand = torch.unsqueeze(i_low_data_ratio, dim=0)
        batch_ratio[i, :, :, :] = i_low_ratio_expand

    return batch_ratio


def predict(model, test_low_data_names, test_high_data_names, res_dir, ckpt_dir, save_predict=False, gpu_id=None,
            phase_name=None):
    with torch.no_grad():
        if gpu_id is not None:
            gpu_id = 0
            gpu_id = torch.device('cuda:' + str(gpu_id))
            model = model.to(gpu_id)

        # Load the network with a pre-trained checkpoint
        for train_phase in phase_name:
            load_model_status, _ = load(model, ckpt_dir, train_phase)
            if load_model_status:
                print(train_phase, ": Model restore success!")
            else:
                print(train_phase, ": No pretrained model to restore!")
                raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False

        # init metric
        MSE_output = []
        SSIM_output = []
        PSNR_output = []
        LPIPS_output = []
        N = len(test_low_data_names)

        # Predict for the test images
        for idx in tqdm(range(N)):
            # if idx < 82:
            #     continue
            test_low_img_path = test_low_data_names[idx]
            test_high_img_path = test_high_data_names[idx]
            # show name of result image
            test_img_name = test_low_img_path.split('/')[-1]
            # print('Processing ', test_img_name)
            # change dim
            test_low_img   = Image.open(test_low_img_path)
            test_low_img   = np.array(test_low_img, dtype="float32")/255.0
            test_low_img   = np.transpose(test_low_img, (2, 0, 1))

            test_high_img = Image.open(test_high_img_path)
            test_high_img = np.array(test_high_img, dtype="float32") / 255.0
            test_high_img = np.transpose(test_high_img, (2, 0, 1))
            _, H, W = test_low_img.shape
            if H % 16 != 0:
                h = H % 16
                test_low_img = test_low_img[:, :-h, :]
                test_high_img = test_high_img[:, :-h, :]
            if W % 16 != 0:
                w = W % 16
                test_low_img = test_low_img[:, :, :-w]
                test_high_img = test_high_img[:, :, :-w]

            input_low_test = torch.from_numpy(np.expand_dims(test_low_img, axis=0)).to(gpu_id)
            input_high_test = torch.from_numpy(np.expand_dims(test_high_img, axis=0)).to(gpu_id)

            ratio = calculate_ratio(input_low_test, input_high_test, gpu_id)

            input_low, input_high, R_low, _, _, _, _, I_low_3, _, I_delta_3, R_denoise = \
                model.forward(input_low_test, input_high_test, ratio)

            output = I_delta_3 * R_denoise

            # metric
            mse = torch.nn.MSELoss(reduction='mean')
            MSE_output += [mse(output, input_high).item()]
            ssim = SSIM_loss()
            SSIM_output += [ssim(output, input_high).item()]
            PSNR_output += [PSNR(MSE_output[-1])]
            LPIPS_1 = LPIPS(output, input_high, gpu_id)
            LPIPS_2 = torch.squeeze(LPIPS_1)
            LPIPS_4 = float(LPIPS_2)
            # LPIPS_4 = float(LPIPS_3)
            # LPIPS_batch = float(torch.squeeze(LPIPS(output.cpu(), input_high.cpu())))
            LPIPS_output += [LPIPS_4]

            if save_predict:  # and idx+1 % 10 == 0:
                # prepare for save
                input_low = np.squeeze(input_low.cpu().numpy())
                result_1 = np.squeeze(R_low.cpu().numpy())
                result_2 = np.squeeze(I_low_3.cpu().numpy())
                result_3 = np.squeeze(I_delta_3.cpu().numpy())
                result_4 = np.squeeze(output.cpu().numpy())
                result_GT = np.squeeze(input_high.cpu().numpy())

                if save_R_L:
                    cat_image_1 = np.concatenate([input_low, result_4, result_GT], axis=2)
                    cat_image_2 = np.concatenate([result_1, result_2, result_3], axis=2)
                    cat_image = np.concatenate([cat_image_1, cat_image_2], axis=1)
                else:
                    cat_image = np.concatenate([input_low, result_4, result_GT], axis=2)

                cat_image = np.transpose(cat_image, (1, 2, 0))
                # print(cat_image.shape)
                im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
                filepath = res_dir + '/' + test_img_name
                im.save(filepath[:-4] + '.jpg')

        # metric
        MSE_avg = sum(MSE_output) / N * 255 * 255 / 1000
        SSIM_avg = sum(SSIM_output) / N
        PSNR_avg = sum(PSNR_output) / N
        LPIPS_avg = sum(LPIPS_output) / N

        print('MSE = ', MSE_avg,
              'SSIM = ', SSIM_avg,
              'PSNR = ', PSNR_avg,
              'LPIPS = ', LPIPS_avg)


def load(model, ckpt_dir, train_phase):
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
                model.DecomNet.load_state_dict(ckpt_dict)
            elif train_phase == 'Seg':
                model.LayerSegNet.load_state_dict(ckpt_dict)
            elif train_phase == 'Relight':
                model.RelightNet.load_state_dict(ckpt_dict)
            elif train_phase == 'Denoise':
                model.DenoiseNet.load_state_dict(ckpt_dict)
            elif train_phase == 'Restore':
                model.RestoreNet.load_state_dict(ckpt_dict)
            return True, global_step
        else:
            return False, 0
    else:
        return False, 0  # TODO 异常


if __name__ == '__main__':
    # TODO logger

    parser = argparse.ArgumentParser(description='Learning Low Light Image Enhancement')

    parser.add_argument('--gpu_id', dest='gpu_id',
                        default="5",
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--data_dir', dest='data_dir',
                        default='./LOL/',
                        help='directory storing the test data')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir',
                        default='./ckpts/MyReNet_LOL_KinD_2/',
                        help='directory for checkpoints')
    parser.add_argument('--res_dir', dest='res_dir',
                        default='./results/test_ISSR/low/',
                        help='directory for saving the results')

    args = parser.parse_args()
    if args.gpu_id != "-1":
        # Create directories for saving the results
        if not os.path.exists(args.res_dir):
            os.makedirs(args.res_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        # Test the model
        main(args.data_dir, args.ckpt_dir, args.res_dir, args.gpu_id)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError
