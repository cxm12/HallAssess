import os
import torch
import torch.nn as nn
from modelDegrad.Degmodel import DegModel 
gpu = torch.cuda.is_available()


class Model(nn.Module):
    def __init__(self, args, srmodel=None, rp='.'):
        super(Model, self).__init__()
        print('Making model...')
        self.scale = args.scale[0]
        self.chop = args.chop
        self.cpu = args.cpu
        self.device = torch.device('cpu' if (not gpu) else 'cuda')
        print(self.device, ' = self.device')
        
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        self.args = args
        self.outchannel = args.n_colors
        self.model = srmodel.to(self.device)
        
        # self.proj_updater = ProjectionUpdater(self.model, feature_redraw_interval=640)
        if args.precision == 'half':
            self.model.half()
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        self.dir = os.path.join(rp, 'experiment', args.save)
        
        print('self.dir = ', self.dir)
        self.load(self.dir, pre_train='', resume=-2, cpu=args.cpu)
        print(self.model)
        self.model = self.model.to(self.device)
        os.makedirs(self.dir, exist_ok=True)

    def forward(self, x, tsk):
        self.tsk = tsk
        # self.proj_updater.redraw_projections()
        if self.chop and not self.training:
            return self.forward_chop(x, tsk)
        else:
            return self.model(x, tsk)

    def get_model(self):
        return self.model

    def state_dict(self, **kwargs):
        target = self.model
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.model
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_best.pt')
            )
        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_{}.pt'.format(epoch)))
            
            torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model_latest.pt'))

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -2:
            m = os.path.join(apath, 'model_best.pt')
            print('Load Model from ', m)
            self.model.load_state_dict(torch.load(m, **kwargs), strict=True)
        else:
            print('!!!!!!!!  Not Load Model  !!!!!!')
            assert resume == 0 and pre_train == '.'
        
    def load_network(self, load_path, strict=True, param_key=None):  # 'params'params_ema
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(load_net, strict=strict)
        print(f'Loading {self.model.__class__.__name__} model from {load_path}.')

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            print('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print('warning', f'  {v}')
            print('warning', 'Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print('warning', f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print('warning', f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def forward_chop(self, x, tsk, shave=10, min_size=40000):
        n_GPUs = min(self.n_GPUs, 4)
        b, _, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        h_size, w_size = h_half + 16, w_half + 16
        h_size += 8
        w_size += 8
        h_size = h_size // 8 * 8
        w_size = w_size // 8 * 8

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch, tsk)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, tsk, shave=shave, min_size=min_size)\
                for patch in lr_list]
        
        output = x.new(b, self.outchannel, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output
