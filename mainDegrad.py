import faulthandler
faulthandler.enable()
import torch
import utility
torch.backends.cudnn.enabled = False
import argparse
from mydataIQA import Deg_SR, Deg_Flourescenedenoise, normalize, PercentileNormalizer
from torch.utils.data import dataloader
import modelDegrad as model
import os
import cv2
import numpy as np
from tifffile import imsave
from sklearn.metrics import mean_absolute_error
rp = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/BioSR/ENLCA/Uni-FMIR/HallAssess/' # os.path.dirname(__file__)
print('rp = ', rp)


def options():
    parser = argparse.ArgumentParser(description='Deg Model')
    parser.add_argument('--model', default='Degmodel', help='model name')
    parser.add_argument('--task', type=int, default=task)
    parser.add_argument('--save', type=str, default=savename, help='save')
    
    # Data specifications
    parser.add_argument('--print_every', type=int, default=1000, help='')
    parser.add_argument('--data_test', type=str, default=testset, help='demo image directory')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGBn_colors')
    parser.add_argument('--n_colors', type=int, default=inputF, help='')
    parser.add_argument('--datamin', type=int, default=0)
    parser.add_argument('--datamax', type=int, default=100)
    # Loss specifications
    parser.add_argument('--cpu', action='store_true', default=False, help='')
    parser.add_argument('--load', type=str, default='', help='file name to load')
    
    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
    parser.add_argument('--n_resblocks', type=int, default=8, help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=32, help='number of feature maps')
    parser.add_argument('--save_models', action='store_true', default=True, help='save all intermediate models')
    parser.add_argument('--scale', type=str, default='1', help='super resolution scale')
    parser.add_argument('--chop', action='store_true', default=True, help='enable memory-efficient forward')
    
    # Model specifications
    parser.add_argument('--act', type=str, default='relu', help='activation function')
    parser.add_argument('--res_scale', type=float, default=0.1, help='residual scaling')
    parser.add_argument('--dilation', action='store_true', help='use dilated convolution')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'), help='FP precision for test (single | half)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    args = parser.parse_args()
    
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    
    return args


class Trainer():
    def __init__(self, args, loader_test, datasetname, my_model):
        self.args = args
        gpu = torch.cuda.is_available()
        self.device = torch.device('cpu' if (not gpu) else 'cuda')
        self.datasetname = datasetname
        self.loader_test = loader_test
        self.model = my_model
        self.normalizer = PercentileNormalizer(2, 99.8)
        self.normalizerhr = PercentileNormalizer(2, 99.8)
        self.dir = os.path.join(rp, 'experiment', self.args.save)
        print('Trainer self.dir = ', self.dir)
        os.makedirs(self.dir, exist_ok=True)
    
    # # -------------------------- SR --------------------------
    def test(self):
        self.testsave = self.dir + '/results/'
        os.makedirs(self.testsave, exist_ok=True)
        torch.set_grad_enabled(False)
        self.model.eval()
    
        num = 0
        for idx_data, (lr, hr, filename) in enumerate(self.loader_test[0]):
            num += 1
            lr, hr = self.prepare(lr, hr)
            glr = self.model(hr, 0)
            glr = utility.quantize(glr, self.args.rgb_range)
            lr = utility.quantize(lr, self.args.rgb_range)
            glr = glr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            lr = lr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            hr = hr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            
            name = '{}.png'.format(filename[0][:-4])
            cv2.imwrite(self.testsave + name, glr)

            utility.savecolorim(self.testsave + name[:-4] + '-Color.png', glr, norm=False)
            glr = np.round(np.maximum(0, np.minimum(255, glr)))
            lr2 = np.round(np.maximum(0, np.minimum(255, lr)))
            res = np.clip(np.abs(glr - lr2), 0, 255)
            utility.savecolorim(self.testsave + name[:-4] + '-MeandfnoNormC.png', res, norm=False)
            HallAssess = mean_absolute_error(glr, lr2)
            print(HallAssess)
    
    def test3Ddenoise(self, condition=1, data_test=''):
        self.testsave = self.dir + '/results/condition_%d/' % condition
        os.makedirs(self.testsave, exist_ok=True)
        datamin, datamax = self.args.datamin, self.args.datamax
        patchsize = 600
        torch.set_grad_enabled(False)
        self.model.eval()
        resultlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            print('filename = ', filename)
            name = '{}'.format(filename[0])
            
            # 1.3D norm 2 998
            lrt = self.normalizer.before(lrt, 'CZYX')  # [0~806] -> [0~1.]
            hrt = self.normalizerhr.before(hrt, 'CZYX')  # [0~806] -> [0~1.]
            lrt, hrt = self.prepare(lrt, hrt)
            
            lr = np.squeeze(lrt.cpu().detach().numpy())
            hr = np.squeeze(hrt.cpu().detach().numpy())
            print('hr.shape = ', hr.shape)  # (5, 1024, 1024)
            addnoiseim = torch.zeros_like(hrt, dtype=hrt.dtype)
            
            inputlst = []
            for ch in range(0, len(hr)):
                inputlst.append(hrt[:, ch:ch + 1])
                
            batchstep = 5  # 10  #
            for dp in range(0, len(inputlst), batchstep):
                if dp + batchstep >= len(hr):
                    dp = len(hr) - batchstep
                # print(dp)  # 0, 10, .., 90
                hrtn = torch.concat(inputlst[dp:dp + batchstep], 0)  # [batch, inputchannel, h, w]
                # print('hrtn.shape = ', hrtn.shape)  # 
                a = self.model(hrtn, 2)
                a = torch.transpose(a, 1, 0)  # [1, batch, h, w]
                # print('a.shape = ', a.shape)  # 
                addnoiseim[:, dp:dp + batchstep, :, :] = a
            
            addnoise = np.float32(addnoiseim.cpu().detach().numpy())
            addnoise = np.squeeze(self.normalizerhr.after(addnoise))
            lr = np.squeeze(self.normalizer.after(lr))
            addnoise255 = np.squeeze(np.float32(normalize(addnoise, datamin, datamax, clip=True))) * 255
            hr255 = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255
            lr255 = np.float32(normalize(lr, datamin, datamax, clip=True)) * 255
            print('addnoise.shape = ', addnoise.shape)
            cresultlst = []
            step = 10
            imsave(self.testsave + name + '.tif', addnoise)
            if 'Planaria' in data_test:
                if condition == 1:
                    randcs = 10
                    randce = hr.shape[0] - 10
                    step = (hr.shape[0] - 20) // 5
                else:
                    randcs = 85
                    randce = 87
                    step = 1
                    if randce >= hr.shape[0]:
                        randcs = hr.shape[0] - 3
                        randce = hr.shape[0]

                for dp in range(randcs, randce, step):
                    # print('dp = ', dp)
                    utility.savecolorim(self.testsave + name + '-dfnoNormC%d.png' % dp, addnoise[dp] - hr[dp], norm=False)
                    utility.savecolorim(self.testsave + name + '-C%d.png' % dp, addnoise[dp])
                    # utility.savecolorim(self.testsave + name + '-HRC%d.png' % dp, hr[dp])
                    addnoisepatch255 = addnoise255[dp, :patchsize, :patchsize]
                    # hrpatch255 = hr255[dp, :patchsize, :patchsize]
                    lrpatch255 = lr255[dp, :patchsize, :patchsize]
                    HallAssess = mean_absolute_error(addnoisepatch255, lrpatch255)
                    cresultlst.append(HallAssess)
            elif 'Tribolium' in data_test:
                    if condition == 1:
                        randcs = 2
                        randce = hr.shape[0] - 2
                        step = (hr.shape[0] - 4) // 6
                    else:
                        randcs = hr.shape[0] // 2 - 1
                        randce = randcs + 3
                        step = 1
                    for randc in range(randcs, randce, step):
                        # hrpatch = normalize(hr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                        lrpatch = normalize(lr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                        addnoisepatchour = normalize(addnoise255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                        HallAssess = mean_absolute_error(addnoisepatchour, lrpatch)
                        cresultlst.append(HallAssess)
            resultlst.append(np.mean(np.array(cresultlst)))
        result = np.mean(np.array(resultlst))
        print(result)

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        
        return [_prepare(a) for a in args]


if __name__ == '__main__':
    task = 1  # 2  #
    inputF = 1
            
    if task == 1: # Assessment of SR results
        testset = 'CCPs'  # 'Microtubules'  # 'F-actin'  # 'ER'  #
        srdatapath = ''
        lrpath = ''
        lrpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/test/CCPs/LR/im1_LR.tif'
        srdatapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/test/CCPs/output_DFCAN-SISR/im1_LR.tif'
    elif task == 2: # Assessment of Denoising results
        condition = 1
        testset = 'Denoising_Planaria'  # 'Denoising_Tribolium'  #
        denoisedatapath = ''
        noisypath = ''
        denoisedatapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/Denoising_Planaria/test_data/SwinIRmto1/c1/EXP278_Smed_fixed_RedDot1_sub_5_N7_m0001.tif'
        noisypath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/Denoising_Planaria/test_data/condition_1/EXP278_Smed_fixed_RedDot1_sub_5_N7_m0001.tif'
    
    savename = 'Degrad-%s/' % testset
    unimodel = model.DegModel(tsk=task, inchannel=inputF)
    
    args = options()
    torch.manual_seed(args.seed)
    _model = model.Model(args, unimodel, rp=rp)
    if task == 1:        
        loader_test = [dataloader.DataLoader(
            Deg_SR(srpath=srdatapath, lrpath=lrpath),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)]
    elif task == 2:        
        loader_test = [dataloader.DataLoader(
            Deg_Flourescenedenoise(denoisepath=denoisedatapath, noisepath=noisypath),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0,
        )]
    
    t = Trainer(args, loader_test, args.data_test, _model)
    if task == 1:
        t.test()
    elif task == 2:
        t.test3Ddenoise(condition=condition, data_test=testset)