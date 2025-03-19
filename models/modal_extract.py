"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import torch
import torch.nn as nn
from DnCNN_noiseprint import make_net
import logging
import os
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# from torchvision.transforms import ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor, to_pil_image

class SRMFilter(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]]

        f2 = [[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]

        f3 = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]

        q = torch.tensor([[4.], [12.], [2.]]).unsqueeze(-1).unsqueeze(-1)
        filters = torch.tensor([[f1, f1, f1], [f2, f2, f2], [f3, f3, f3]], dtype=torch.float) / q
        self.register_buffer('filters', filters)
        self.truc = nn.Hardtanh(-2, 2)

    def forward(self, x):
        x = F.conv2d(x, self.filters, padding='same', stride=1)
        x = self.truc(x)
        return x


class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super().__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)

    def bayarConstraint(self):
        self.kernel.data = self.kernel.data.div(self.kernel.data.sum(-1, keepdims=True))
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x

class ModalitiesExtractor(nn.Module):
    def __init__(self,
                 modals: list = ('noiseprint', 'bayar', 'srm'),
                 noiseprint_path: str = None):
        super().__init__()
        self.mod_extract = []
        if 'noiseprint' in modals:
            num_levels = 17
            out_channel = 1
            self.noiseprint = make_net(3, kernels=[3, ] * num_levels,
                                  features=[64, ] * (num_levels - 1) + [out_channel],
                                  bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                                  acts=['relu', ] * (num_levels - 1) + ['linear', ],
                                  dilats=[1, ] * num_levels,
                                  bn_momentum=0.1, padding=1)

            if noiseprint_path:
                np_weights = noiseprint_path
                assert os.path.isfile(np_weights)
                dat = torch.load(np_weights, map_location=torch.device('cpu'))
                logging.info(f'Noiseprint++ weights: {np_weights}')
                self.noiseprint.load_state_dict(dat)

            self.noiseprint.eval()
            for param in self.noiseprint.parameters():
                param.requires_grad = False
            self.mod_extract.append(self.noiseprint)
        if 'bayar' in modals:
            self.bayar = BayarConv2d(3, 3, padding=2)
            self.mod_extract.append(self.bayar)
        if 'srm' in modals:
            self.srm = SRMFilter()
            self.mod_extract.append(self.srm)

    def set_train(self):
        if hasattr(self, 'bayar'):
            self.bayar.train()

    def set_val(self):
        if hasattr(self, 'bayar'):
            self.bayar.eval()

    def forward(self, x) -> list:
        out = []
        for mod in self.mod_extract:
            y = mod(x)
            if y.size()[-3] == 1:
                y = torch.tile(y, (3, 1, 1))
            out.append(y)

        return out


if __name__ == '__main__':
    modal_ext = ModalitiesExtractor(['noiseprint', 'bayar', 'srm'], '/data/gk/Vmamba_CAB/weights/np++.pth')

    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision.transforms import ToPILImage, ToTensor


    # 输入文件夹路径
    dataset_name = "Columbia"
    input_folder = f'/data/gk/dataset/{dataset_name}/tamper/'  # 原始图像所在文件夹
    output_dir = f'/data/gk/Vmamba_CAB/results/{dataset_name}_modals/'  # 确保此路径存在

    # 遍历输入文件夹中的所有图像文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif')):  # 支持的图像格式
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('RGB')
            inp = ToTensor()(img).unsqueeze(0)

            out = modal_ext(inp)
            # 创建以原始文件名（无后缀）命名的文件夹
            base_name = os.path.splitext(filename)[0]  # 获取无后缀的文件名
            output_folder = os.path.join(output_dir, base_name)  # 将输出文件夹设置为原始图像名称
            os.makedirs(output_folder, exist_ok=True)  # 创建文件夹（如果不存在）


            # 保存原始图像
            # plt.imsave(os.path.join(output_folder, f'{base_name}_original.png'), img)
            img.save(os.path.join(output_folder, f'{base_name}_original.png'))

            # 保存 NoisePrint++
            noiseprint = out[0][:, 0].squeeze().numpy()
            noiseprint_cropped = noiseprint[16:-16:4, 16:-16:4]  # 裁剪后的噪声图
            plt.imsave(os.path.join(output_folder, f'{base_name}_noiseprint.png'), noiseprint_cropped, cmap='gray')

            # 保存 Bayar
            bayar = ToTensor()(ToPILImage()(out[1].squeeze())).permute(1, 2, 0).numpy()
            plt.imsave(os.path.join(output_folder, f'{base_name}_bayar.png'), bayar)

            # 保存 SRM
            srm = ToTensor()(ToPILImage()(out[2].squeeze())).permute(1, 2, 0).numpy()
            plt.imsave(os.path.join(output_folder, f'{base_name}_srm.png'), srm)

    print("处理完成，所有图像已保存。")


    #
    # # 输入文件夹路径
    # dataset_name = "Columbia"
    # input_folder = f'/data/gk/dataset/{dataset_name}/tamper/'  # 原始图像所在文件夹
    # output_dir = f'/data/gk/Vmamba_CAB/results/{dataset_name}_modals/'  # 确保此路径存在
    #
    # # 遍历输入文件夹中的所有图像文件
    # for filename in tqdm(os.listdir(input_folder)):
    #     if filename.endswith(('.png', '.jpg', '.jpeg', '.tif')):  # 支持的图像格式
    #         img_path = os.path.join(input_folder, filename)
    #         img = Image.open(img_path).convert('RGB')
    #         inp = to_tensor(img).unsqueeze(0)
    #
    #         out = modal_ext(inp)
    #
    #         # 创建以原始文件名（无后缀）命名的文件夹
    #         base_name = os.path.splitext(filename)[0]  # 获取无后缀的文件名
    #         output_folder = os.path.join(output_dir, base_name)  # 将输出文件夹设置为原始图像名称
    #         os.makedirs(output_folder, exist_ok=True)  # 创建文件夹（如果不存在）
    #
    #         # 保存原始图像
    #         img.save(os.path.join(output_folder, f'{base_name}_original.png'))
    #
    #         # 保存 NoisePrint++
    #         noiseprint = out[0][:, 0].squeeze()
    #         noiseprint_image = Image.fromarray((noiseprint.numpy() * 255).astype(np.uint8)[16:-16:4, 16:-16:4])  # * 255
    #         noiseprint_image = noiseprint_image.convert('L')
    #         noiseprint_image.save(os.path.join(output_folder, f'{base_name}_noiseprint.png'))
    #
    #         # 保存 Bayar
    #         bayar = to_tensor(to_pil_image(out[1].squeeze())).permute(1, 2, 0).numpy()
    #         bayar_image = Image.fromarray((bayar * 255).astype(np.uint8))
    #         bayar_image.save(os.path.join(output_folder, f'{base_name}_bayar.png'))
    #
    #         # 保存 SRM
    #         srm = to_tensor(to_pil_image(out[2].squeeze())).permute(1, 2, 0).numpy()
    #         srm_image = Image.fromarray((srm * 255).astype(np.uint8))
    #         srm_image.save(os.path.join(output_folder, f'{base_name}_srm.png'))
    #
    # print("处理完成，所有图像已保存。")