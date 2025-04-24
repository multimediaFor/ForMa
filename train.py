import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP: known incorrect sRGB profile.*")
import numpy as np
from mydatasets import MyDataset, thresholding
from torch.utils.data import DataLoader
import yaml
import logging as logger
import torch.optim as optim
import torch.nn as nn
import torch
import os
import cv2
import shutil
from losses import MyLoss
from models.vmamba_pixelshuf_modals import Forensic_Vmamba
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metric import calc_fixed_f1_iou
import datetime
from get_device import get_device

with open('config_pixelshuf_modals.yaml', 'r') as f:
    args = yaml.safe_load(f)
device = get_device(args["cuda_idx"])
print(f"Using device: {device}")
date_now = datetime.datetime.now()
date_now = 'Log_v%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
args["out_dir"] = args["out_dir"] + date_now
np.random.seed(666666)
torch.manual_seed(666666)
torch.cuda.manual_seed(666666)
torch.backends.cudnn.deterministic = True
logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s] %(message)s',
                   datefmt='%m-%d %H:%M:%S')


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def convert(x):
    x = x * 255.
    return x.permute(0, 2, 3, 1).cpu().detach().numpy()


class MyVmamba(nn.Module):
    def __init__(self,
                 net_weight="",continue_=False):
        super(MyVmamba, self).__init__()
        self.cur_net = Forensic_Vmamba().to(device)
        if continue_ == True:
            self.epoch_iteration_1w = 0
            self.count = 0
            self.lr = 1e-4
            weights = torch.load(net_weight)["model"]
            self.load(self.cur_net, weights)
            self.extractor_optimizer = optim.AdamW(self.cur_net.parameters(), lr=self.lr)
        else:
            checkpoint = torch.load(net_weight)
            self.extractor_optimizer = optim.AdamW(self.cur_net.parameters(), lr=0)
            self.load(self.cur_net, checkpoint['model_state_dict'])
            self.extractor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch_iteration_1w = checkpoint['epoch_iteration_1w']
            self.count = checkpoint['count']
            print("self.epoch_iteration_1w", self.epoch_iteration_1w)
            print("self.count", self.count)
            print("['lr']", self.extractor_optimizer.param_groups[0]['lr'])

        self.save_dir = 'weights/' + args["out_dir"]
        if args["type"] == 'train':
            rm_and_make_dir(self.save_dir)
        self.myLoss = MyLoss()

    def process(self, Ii, Mg, isTrain=False):
        self.extractor_optimizer.zero_grad()

        if isTrain:
            Fo = self.cur_net(Ii)
            batch_loss = self.myLoss(Fo, Mg)
            self.backward(batch_loss)
            return batch_loss
        else:
            with torch.no_grad():
                Fo = self.cur_net(Ii)
            return Fo

    def backward(self, batch_loss=None):
        if batch_loss:
            batch_loss.backward(retain_graph=False)
            self.extractor_optimizer.step()

    def save(self, path='', epoch_iteration_1w=0, count=0):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        checkpoint = {
            'model_state_dict': self.cur_net.state_dict(),
            'optimizer_state_dict': self.extractor_optimizer.state_dict(),
            "epoch_iteration_1w": epoch_iteration_1w,
            "count": count,
        }
        torch.save(checkpoint, self.save_dir + path + '%s_weights.pth' % self.cur_net.name)

    def load(self, model, weights):


        model_state_dict = model.state_dict()

        loaded_layers = []
        missing_layers = []
        mismatched_shapes = []

        for name, param in weights.items():
            name = "backbone." + name
            if name in model_state_dict:
                if param.shape == model_state_dict[name].shape:
                    model_state_dict[name].copy_(param)
                    loaded_layers.append(name)
                else:
                    print(name + ' has shape ' + str(param.size()))
                    mismatched_shapes.append(name)
            else:
                print(name + ' is not ')
                missing_layers.append(name)

        model.load_state_dict(model_state_dict, strict=False)


class ForgeryForensics():
    def __init__(self):
        self.train_npy_list = [
            # name, repeat_time
            ("sp_images_199999.npy", 1),
            ("cm_images_199429.npy", 1),
            ("bcm_images_199443.npy", 1),
            ("CASIA2_5123.npy", 40),
            ('IMD_2010.npy', 20),
        ]
        self.train_file = None
        for item in self.train_npy_list:
            self.train_file_tmp = np.load(args["flist_path"] + item[0])
            for _ in range(item[1]):
                self.train_file = np.concatenate(
                    [self.train_file, self.train_file_tmp]) if self.train_file is not None else self.train_file_tmp
        self.train_num = len(self.train_file)
        train_dataset = MyDataset(num=self.train_num, file=self.train_file, choice='train',
                                  input_size=args["input_size"], gt_ratio=args["gt_ratio"])

        self.val_npy_list = [
            # name, nickname
            # Validation Dataset:
            ('Columbia_160.npy', 'Columbia'),
            ('DSO_100.npy', 'DSO'),
            ('CASIAv1_920.npy', 'CASIAv1'),
            ('NIST_564.npy', 'NIST'),
            ('Coverage_100.npy', 'Coverage'),
            ('Korus_220.npy', 'Korus'),
            ('In_the_wild_201.npy', 'In_the_wild'),
            ('CoCoGlide_512.npy', 'CoCoGlide'),
            ('MISD_227.npy', 'MISD'),
            ('FFpp_1000.npy', 'FFpp'),
        ]
        self.val_file_list = []
        for item in self.val_npy_list:
            self.val_file_tmp = np.load(args["flist_path"] + item[0])
            self.val_file_list.append(self.val_file_tmp)

        self.train_bs = args["train_bs"]
        self.test_bs = args["test_bs"]
        self.ProMamba = MyVmamba(r"weights/vssm1_tiny_0230s_ckpt_epoch_264.pth",
                                  continue_=False).to(device)
        self.n_epochs = 99999
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_bs, shuffle=True)
        logger.info('Train on %d images.' % self.train_num)
        for idx, file_list in enumerate(self.val_file_list):
            logger.info('Test on %s (#%d).' % (self.val_npy_list[idx][0], len(file_list)))

    def train(self):
        train_writer = SummaryWriter(log_dir=os.path.join(self.ProMamba.save_dir, 'runs'))
        batch_losses = []
        best_score = 0
        scheduler = ReduceLROnPlateau(self.ProMamba.extractor_optimizer, mode='max', factor=0.8, patience=3,
                                      min_lr=1e-8)
        self.ProMamba.train()
        epoch_iteration_1w = self.ProMamba.epoch_iteration_1w
        count = self.ProMamba.count
        for epoch in range(1, self.n_epochs + 1):
            for items in self.train_loader:
                count += self.train_bs
                Ii, Mg = (item.to(device) for item in items[:2])  # Input, Ground-truth Mask
                batch_loss = self.ProMamba.process(Ii, Mg, isTrain=True)
                batch_losses.append(batch_loss.item())
                if count % (self.train_bs * 20) == 0:
                    current_lr = self.ProMamba.extractor_optimizer.param_groups[0]['lr']
                    logger.info('Train Num (%6d/%6d), Loss:%5.4f,LR: %5.8f' % (
                    count, self.train_num, np.mean(batch_losses), current_lr))
                    train_writer.add_scalar('Loss/train', np.mean(batch_losses), count, current_lr)
                if count % int((self.train_loader.dataset.__len__() / 100) // self.train_bs * self.train_bs) == 0:
                    epoch_iteration_1w += 1
                    self.ProMamba.save('latest/', epoch_iteration_1w, count)
                    current_lr = self.ProMamba.extractor_optimizer.param_groups[0]['lr']
                    logger.info('Ep%03d(%6d/%6d): Tra: Loss :%5.4f,LR: %5.8f' % (
                    epoch, count, self.train_num, np.mean(batch_losses), current_lr))
                    train_writer.add_scalar('Loss/train', np.mean(batch_losses), count)
                    tmp_score = self.val(epoch, epoch_iteration_1w)
                    scheduler.step(tmp_score)
                    if tmp_score > best_score:
                        train_writer.add_scalar('Score/train', tmp_score, count)
                        best_score = tmp_score
                        logger.info('Score: %5.4f (Best)' % best_score)
                        train_writer.add_scalar('Score/train(Best)', best_score, count)
                        self.ProMamba.save('Ep%03d_%5.4f/' % (epoch, tmp_score), epoch_iteration_1w, count)
                    else:
                        logger.info('Score: %5.4f' % tmp_score)
                        train_writer.add_scalar('Score/train', tmp_score, count)
                    self.ProMamba.train()
                    batch_losses = []
            count = 0

    def val(self, epoch, epoch_iteration_1w):
        tmp_F1 = []
        tmp_IOU = []
        test_nums = []
        result_file_path = os.path.join(self.ProMamba.save_dir, 'result.txt')
        with open(result_file_path, 'a') as result_file:
            result_file.write(f"Epoch {epoch}:\n")
            for idx in range(len(self.val_file_list)):
                P_F1, P_IOU, test_num = ForensicTesting(self.ProMamba, bs=self.test_bs,
                                                        test_npy=self.val_npy_list[idx][0],
                                                        test_file=self.val_file_list[idx],
                                                        epoch_iteration_1w=epoch_iteration_1w)
                tmp_IOU.append(P_IOU)
                tmp_F1.append(P_F1)
                test_nums.append(test_num)
                result_str = '%s(#%d): F1:%5.4f, PIOU:%5.4f\n' % (
                    self.val_npy_list[idx][1],  # Dataset name (e.g., CASIAv1)
                    len(self.val_file_list[idx]),  # Length of the dataset (or number of files)
                    P_F1,  # F1 score
                    P_IOU  # IoU score
                )
                result_file.write(result_str)
            # average
            avg_F1 = np.mean(tmp_F1)
            avg_IOU = np.mean(tmp_IOU)
            logger.info('Average F1: %5.4f' % avg_F1)
            logger.info('Average IoU: %5.4f' % avg_IOU)
            avg_result_str = 'Average F1: %5.4f\nAverage IoU: %5.4f\n\n' % (avg_F1, avg_IOU)
            result_file.write(avg_result_str)
            # Weighted Average
            total_samples = sum(test_nums)
            weighted_avg_F1 = np.sum(np.array(tmp_F1) * np.array(test_nums)) / total_samples
            weighted_avg_IOU = np.sum(np.array(tmp_IOU) * np.array(test_nums)) / total_samples
            logger.info('Average weight F1: %5.4f' % weighted_avg_F1)
            logger.info('Average weight IoU: %5.4f' % weighted_avg_IOU)
            avg_weight_result_str = 'Average weight F1: %5.4f\nAverage weight IoU: %5.4f\n\n' % (
            weighted_avg_F1, weighted_avg_IOU)
            result_file.write(avg_weight_result_str)

            current_lr = self.ProMamba.extractor_optimizer.param_groups[0]['lr']
            current_lr_str = 'current_lr : %5.8f' % (current_lr)
            result_file.write(current_lr_str)

        return (weighted_avg_F1 + weighted_avg_IOU) / 2.0


# test
def ForensicTesting(model, bs=1, test_npy='', test_file=None, epoch_iteration_1w=0):
    if test_file is None:
        test_file = np.load(args["flist_path"] + test_npy)
    test_num = len(test_file)
    test_dataset = MyDataset(test_num, test_file, choice='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=min(48, 2), shuffle=False)
    model.eval()
    f1, iou = [], []
    if args["save_res"]:
        path_out = args["path_out"]
        rm_and_make_dir(path_out)

    for items in test_loader:
        Ii, Mg, Hg, Wg = (item.to(device) for item in items[:-1])
        filename = items[-1]

        Mo = model.process(Ii, None, isTrain=False)
        for i in range(Mo.shape[0]):
            fixed_f1, iou_score = calc_fixed_f1_iou(Mo[i], Mg[i])
            f1.append(fixed_f1.cpu())
            iou.append(iou_score.cpu())

        if args["save_res"]:
            Mo = torch.softmax(Mo, dim=1)
            Mo = torch.argmax(Mo, dim=1)
            Mo = Mo.unsqueeze(1)
            Mg, Mo = convert(Mg), convert(Mo)
            Hg, Wg = Hg.cpu().numpy(), Wg.cpu().numpy()
            for i in range(Ii.shape[0]):
                res = cv2.resize(Mo[i], (Wg[i].item(), Hg[i].item()))
                res = thresholding(res)
                cv2.imwrite(path_out + filename[i][:-4] + '.png', res.astype(np.uint8))

    Pixel_F1 = np.mean(f1)
    Pixel_IOU = np.mean(iou)
    if args["type"] == 'test_single':
        logger.info('Score: F1: %5.4f, IoU: %5.4f' % (Pixel_F1, Pixel_IOU))
    return Pixel_F1, Pixel_IOU, test_num


if __name__ == '__main__':
    model = ForgeryForensics()
    model.train()
