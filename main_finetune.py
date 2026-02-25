# import argparse
# import datetime
# import json
# import numpy as np
# import os
# import time
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter

# import timm

# #assert timm.__version__ == "0.3.2" # version check
# from timm.models.layers import trunc_normal_
# from timm.data.mixup import Mixup
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# import util.lr_decay as lrd
# import util.misc as misc
# from util.datasets import build_dataset
# from util.pos_embed import interpolate_pos_embed
# from util.misc import NativeScalerWithGradNormCount as NativeScaler
# from spike_quan_wrapper import myquan_replace, SNNWrapper

# import models_vit
# import wandb
# import matplotlib.pyplot as plt
# from engine_finetune import train_one_epoch, evaluate, unstruct_prune

# import warnings
# from functools import partial
# from copy import deepcopy

# warnings.filterwarnings("ignore", category=UserWarning)


# def get_args_parser():
#     parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
#     parser.add_argument('--batch_size', default=64, type=int,
#                         help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
#     parser.add_argument('--epochs', default=50, type=int)
#     parser.add_argument('--print_freq', default=1000, type=int,
#                         help='print_frequency')
#     parser.add_argument('--accum_iter', default=1, type=int,
#                         help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
#     parser.add_argument('--project_name', default='T-SNN', type=str, metavar='MODEL',
#                         help='Name of model to train')

#     # Model parameters
#     parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
#                         help='Name of model to train')

#     parser.add_argument('--input_size', default=224, type=int,
#                         help='images input size')
#     parser.add_argument('--encoding_type', default="analog", type=str,
#                         help='encoding type for snn')
#     parser.add_argument('--time_step', default=2000, type=int,
#                         help='time-step for snn')
#     parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
#                         help='Drop path rate (default: 0.1)')
#     parser.add_argument('--drop_rate', type=float, default=0.0, metavar='PCT',
#                         help='Dropout rate')

#     # Optimizer parameters
#     parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
#                         help='Clip gradient norm (default: None, no clipping)')
#     parser.add_argument('--weight_decay', type=float, default=0.05,
#                         help='weight decay (default: 0.05)')

#     parser.add_argument('--lr', type=float, default=None, metavar='LR',
#                         help='learning rate (absolute lr)')
#     parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
#                         help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
#     parser.add_argument('--layer_decay', type=float, default=0.75,
#                         help='layer-wise lr decay from ELECTRA/BEiT')
#     parser.add_argument('--act_layer', type=str, default="relu",
#                         help='Using ReLU or GELU as activation')

#     parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
#                         help='lower lr bound for cyclic schedulers that hit 0')

#     parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
#                         help='epochs to warmup LR')

#     # Augmentation parameters
#     parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
#                         help='Color jitter factor (enabled only when not using Auto/RandAug)')
#     parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
#                         help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
#     parser.add_argument('--smoothing', type=float, default=0.1,
#                         help='Label smoothing (default: 0.1)')

#     # * Random Erase params
#     parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
#                         help='Random erase prob (default: 0.25)')
#     parser.add_argument('--remode', type=str, default='pixel',
#                         help='Random erase mode (default: "pixel")')
#     parser.add_argument('--recount', type=int, default=1,
#                         help='Random erase count (default: 1)')
#     parser.add_argument('--resplit', action='store_true', default=False,
#                         help='Do not random erase first (clean) augmentation split')

#     # * Mixup params
#     parser.add_argument('--mixup', type=float, default=0,
#                         help='mixup alpha, mixup enabled if > 0.')
#     parser.add_argument('--cutmix', type=float, default=0,
#                         help='cutmix alpha, cutmix enabled if > 0.')
#     parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
#                         help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
#     parser.add_argument('--mixup_prob', type=float, default=1.0,
#                         help='Probability of performing mixup or cutmix when either/both is enabled')
#     parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
#                         help='Probability of switching to cutmix when both mixup and cutmix enabled')
#     parser.add_argument('--mixup_mode', type=str, default='batch',
#                         help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

#     # * Finetuning params
#     parser.add_argument('--finetune', default='',
#                         help='finetune from checkpoint')
#     parser.add_argument('--global_pool', action='store_true')
#     parser.set_defaults(global_pool=True)
#     parser.add_argument('--cls_token', action='store_false', dest='global_pool',
#                         help='Use class token instead of global pool for classification')

#     # Dataset parameters
#     parser.add_argument('--dataset', default='imagenet', type=str,
#                         help='dataset name')
#     parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
#                         help='dataset path')
#     parser.add_argument('--nb_classes', default=1000, type=int,
#                         help='number of the classification types')
#     parser.add_argument('--define_params', action='store_true')
#     parser.add_argument('--mean', nargs='+', type=float)
#     parser.add_argument('--std', nargs='+', type=float)

#     parser.add_argument('--output_dir', default='./output_dir',
#                         help='path where to save, empty for no saving')
#     parser.add_argument('--log_dir', default='./output_dir',
#                         help='path where to tensorboard log')
#     parser.add_argument('--device', default='cuda',
#                         help='device to use for training / testing')
#     parser.add_argument('--seed', default=0, type=int)
#     parser.add_argument('--resume', default='',
#                         help='resume from checkpoint')

#     parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
#                         help='start epoch')
#     parser.add_argument('--eval', action='store_true',
#                         help='Perform evaluation only')
#     parser.add_argument('--energy_eval', action='store_true',
#                         help='Perform evaluation with energy consumption')
#     parser.add_argument('--wandb', action='store_true',
#                         help='Using wandb or not')
#     parser.add_argument('--dist_eval', action='store_true', default=False,
#                         help='Enabling distributed evaluation (recommended during training for faster monitor')
#     parser.add_argument('--num_workers', default=32, type=int)
#     parser.add_argument('--pin_mem', action='store_true',
#                         help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
#     parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
#     parser.set_defaults(pin_mem=True)

#     # distributed training parameters
#     parser.add_argument('--world_size', default=1, type=int,
#                         help='number of distributed processes')
#     parser.add_argument('--local_rank', default=-1, type=int)
#     parser.add_argument('--dist_on_itp', action='store_true')
#     parser.add_argument('--dist_url', default='env://',
#                         help='url used to set up distributed training')
    
#     # training mode
#     parser.add_argument('--mode', default="ANN", type=str,
#                         help='the running mode of the script["ANN", "QANN_PTQ", "QANN_QAT", "SNN"]')
#     # prune
#     parser.add_argument('--ratio', default=0.0, type=float,
#                         help='the ratio of unstructure prune')
#     # LSQ quantization
#     parser.add_argument('--level', default=32, type=int,
#                         help='the quantization levels')
#     parser.add_argument('--weight_quantization_bit', default=32, type=int, help="the weight quantization bit")
#     parser.add_argument('--neuron_type', default="ST-BIF", type=str,
#                         help='neuron type["ST-BIF", "IF"]')
#     parser.add_argument('--remove_softmax', action='store_true',
#                         help='need softmax or not')
    # parser.add_argument('--gelu_path', default=r'/home/hyuntaek/STA/premodels/distilled_gelu_64.pth', type=str)
#     return parser


# spike_rates = {}
# class MeasureSpikeRateHook:
#     def __init__(self, name):
#         self.name = name
#         self.total_spikes = 0.0
#         self.total_elements = 0.0

#     def __call__(self, module, input, output):
#         if isinstance(output, torch.Tensor):
#             # 1. 스파이크 개수 누적 (+= 사용)
#             self.total_spikes += torch.count_nonzero(output).item()
            
#             # 2. 전체 요소 개수 누적 (+= 사용)
#             self.total_elements += output.numel()

#     # 나중에 결과를 계산하기 위한 함수
#     def compute_rate(self):
#         if self.total_elements > 0:
#             return self.total_spikes / self.total_elements
#         return 0.0
    

# class SaveOutputNpyHook:
#     def __init__(self, module_name, save_dir, max_batches=1):
#         self.module_name = module_name
#         self.save_dir = save_dir
#         self.max_batches = max_batches
#         self.call_count = 0

#     def __call__(self, module, input, output):
#         if self.call_count >= self.max_batches:
#             return

#         if isinstance(output, torch.Tensor):
#             # 1. CPU로 이동 및 Detach
#             tensor_val = output.detach().cpu()
            
#             # 2. 파일명 안전하게 변환 (특수문자 제거)
#             safe_name = self.module_name.replace("model.", "").replace(".linear", "").replace(".conv", "")
            
#             # 3. .npy 파일로 저장 (경로: save_dir/레이어이름.npy)
#             # SNN의 경우 TimeStep 차원이 있다면 평균을 내서 저장할지, 전체를 저장할지 결정해야 함
#             # 보통 비교를 위해 SNN은 TimeStep 차원에 대해 평균(Mean)을 내서 QANN과 차원을 맞춥니다.
            
#             if len(tensor_val.shape) == 4 and tensor_val.shape[0] != args.batch_size: 
#                  # SNN 짐작: [T, B, C, ...] 형태라면 T(0번축)에 대해 평균
#                  # 만약 [B, T, C, ...] 형태라면 1번축 평균. 모델 구조에 따라 확인 필요.
#                  # 여기서는 T가 0번 축이라고 가정 (SpikeZIP 보통 구조)
#                  save_val = tensor_val.mean(dim=0).numpy()
#             else:
#                  save_val = tensor_val.numpy()

#             file_path = os.path.join(self.save_dir, f"{safe_name}.npy")
#             np.save(file_path, save_val)
#             print(f"Saved {file_path} shape={save_val.shape}")
        
#         self.call_count += 1

# def set_sparsity_weight(model):
#     for name, m in model.named_modules():
#         if name.count("proj")>0 or name.count("fc2")>0:
#             if isinstance(m,torch.nn.Sequential) and isinstance(m[0],torch.nn.Linear):
#                 m[0].weight.data = m[0].weight_mask
#             elif isinstance(m,torch.nn.Linear):
#                 m.weight.data = m.weight_mask
    
# def cal_sparsity(model):
#     zero_number = 0
#     total_bumber = 0
#     for name, m in model.named_modules():
#         if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
#             zero_number = zero_number + torch.sum(m.weight==0)
#             total_bumber = total_bumber + m.weight.numel()

#     print("prune finish!!!!! global sparsity:",(zero_number/total_bumber)*100)

# def main(args):
#     misc.init_distributed_mode(args)

#     print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
#     print("{}".format(args).replace(', ', ',\n'))

#     device = torch.device(args.device)

#     # fix the seed for reproducibility
#     seed = args.seed + misc.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)

#     cudnn.benchmark = True

#     dataset_train = build_dataset(is_train=True, args=args)
#     dataset_val = build_dataset(is_train=False, args=args)

#     if True:  # args.distributed:
#         num_tasks = misc.get_world_size()
#         global_rank = misc.get_rank()
#         sampler_train = torch.utils.data.DistributedSampler(
#             dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
#         )
#         print("Sampler_train = %s" % str(sampler_train))
#         if args.dist_eval:
#             if len(dataset_val) % num_tasks != 0:
#                 print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
#                       'This will slightly alter validation results as extra duplicate entries are added to achieve '
#                       'equal num of samples per-process.')
#             sampler_val = torch.utils.data.DistributedSampler(
#                 dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)  # shuffle=True to reduce monitor bias
#         else:
#             sampler_val = torch.utils.data.SequentialSampler(dataset_val)
#     else:
#         sampler_train = torch.utils.data.RandomSampler(dataset_train)
#         sampler_val = torch.utils.data.SequentialSampler(dataset_val)

#     if global_rank == 0 and args.log_dir is not None:
#         os.makedirs(args.log_dir, exist_ok=True)
#         args.log_dir = os.path.join(args.log_dir,
#                                     "{}_{}_{}_{}_{}_act{}_weightbit{}".format(args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level,args.weight_quantization_bit))
#         os.makedirs(args.log_dir, exist_ok=True)
#         log_writer = SummaryWriter(log_dir=args.log_dir)
#         if args.wandb:
#             wandb.init(config=args, project=args.project_name,
#                        name="{}_{}_{}_{}_{}_act{}_weightbit{}".format(args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level,args.weight_quantization_bit),
#                        dir=args.output_dir)
#     else:
#         log_writer = None

#     data_loader_train = torch.utils.data.DataLoader(
#         dataset_train, sampler=sampler_train,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=True,
#     )

#     data_loader_val = torch.utils.data.DataLoader(
#         dataset_val, sampler=sampler_val,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=False
#     )

#     mixup_fn = None
#     mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
#     if mixup_active:
#         print("Mixup is activated!")
#         mixup_fn = Mixup(
#             mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
#             prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
#             label_smoothing=args.smoothing, num_classes=args.nb_classes)

#     if args.act_layer == "relu":
#         activation = nn.ReLU
#     elif args.act_layer == "gelu":
#         activation = nn.GELU
#     else:
#         raise NotImplementedError

#     print("args.drop_path",args.drop_path)
#     if "vit_small" in args.model:
#         model = models_vit.__dict__[args.model](
#             num_classes=args.nb_classes,
#             drop_path_rate=args.drop_path,
#             drop_rate = args.drop_rate,
#             global_pool=False,
#             act_layer=activation,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6),
#             # qkv_bias=True,
#         )
#     else:
#         model = models_vit.__dict__[args.model](
#             num_classes=args.nb_classes,
#             drop_path_rate=args.drop_path,
#             drop_rate = args.drop_rate,
#             global_pool=args.global_pool,
#             act_layer=activation,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6),
#             # qkv_bias=True,
#         )

#     if args.finetune and not args.eval and not (args.mode == "SNN") and not (args.mode == "QANN-QAT" and args.eval):
#         checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

#         print("Load pre-trained checkpoint from: %s" % args.finetune)
#         checkpoint_model = checkpoint['model']
#         state_dict = model.state_dict()
#         for k in ['head.weight', 'head.bias']:
#             if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#                 print(f"Removing key {k} from pretrained checkpoint")
#                 del checkpoint_model[k]

#         # interpolate position embedding
#         interpolate_pos_embed(model, checkpoint_model)

#         # load pre-trained model
#         msg = model.load_state_dict(checkpoint_model, strict=False)
#         print(msg)

#         # if args.global_pool:
#         #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
#         # else:
#         #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

#         # manually initialize fc layer
#         trunc_normal_(model.head.weight, std=2e-5)

#     if args.rank == 0:
#         print("======================== ANN model ========================")
#         f = open(f"{args.log_dir}/ann_model_arch.txt","w+")
#         f.write(str(model))
#         f.close()
#     if args.mode.count("QANN") > 0:
#         myquan_replace(model, args.level, args.weight_quantization_bit)
        
#         # [평가 모드일 때만 실행되는 블록]
#         if args.eval:
#             checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False) if not args.eval else torch.load(args.resume, map_location='cpu', weights_only=False)
#             print("Load pre-trained checkpoint from: %s" % args.finetune)
#             checkpoint_model = checkpoint['model']
#             state_dict = model.state_dict()
#             for k in ['head.weight', 'head.bias']:
#                 if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#                     print(f"Removing key {k} from pretrained checkpoint")
#                     del checkpoint_model[k]

#             # interpolate position embedding
#             interpolate_pos_embed(model, checkpoint_model)

#             # load pre-trained model
#             msg = model.load_state_dict(checkpoint_model, strict=False)
#             print(msg)

#             if args.rank == 0:
#                 save_dir = os.path.join(args.log_dir, "numpy_qann")
#                 os.makedirs(save_dir, exist_ok=True)
#                 print(f"Hook registered! Saving .npy files to {save_dir}")

#                 # [수정된 로직] 모든 모듈을 순회하며 대상 레이어 및 양자화 레이어 탐색
#                 from spike_quan_layer import MyQuan # 클래스 타입 체크용
#                 for name, module in model.named_modules():
#                     # (1) 일반 연산 레이어 저장
#                     if isinstance(module, (nn.Linear, nn.Conv2d)): 
#                         hook_instance = SaveOutputNpyHook(name, save_dir, max_batches=1)
#                         module.register_forward_hook(hook_instance)
                    
#                     # (2) 양자화 레이어(MyQuan) 저장 - 이름에 '_post' 추가
#                     elif isinstance(module, MyQuan):
#                         hook_instance = SaveOutputNpyHook(name + "_post", save_dir, max_batches=1)
#                         module.register_forward_hook(hook_instance)
#                     elif isinstance(module, (nn.GELU, nn.ReLU)):
#                             # 파일명을 직관적으로 (예: blocks.11.mlp.act_act.npy)
#                             hook_name = name.replace(".1", "") + "_act"
#                             hook_instance = SaveOutputNpyHook(hook_name, save_dir, max_batches=1)
#                             module.register_forward_hook(hook_instance)
#         # [공통] 모델 구조 저장 및 출력 (Eval/Train 모두 실행)
#         if args.rank == 0:
#             print("======================== QANN model =======================")
#             f = open(f"{args.log_dir}/qann_model_arch.txt","w+")
#             f.write(str(model))
#             f.close()

#     elif args.mode == "SNN":
#         # -------------------------------------------------------------------------
#         # 1. 구조 변경 및 가중치 이식 [모든 Rank 실행]
#         # -------------------------------------------------------------------------
#         myquan_replace(model, args.level, args.weight_quantization_bit)

#         ugo_path = args.gelu_path if hasattr(args, 'gelu_path') else '/home/hyuntaek/STA/premodels/distilled_gelu_64.pth'
#         # 모든 Rank가 각자 로컬에서 가중치를 로드해야 함
#         ugo_ckpt = torch.load(ugo_path, map_location='cpu')
        
#         model_state_dict = model.state_dict()
#         for i in range(12): 
#             base_key = f"blocks.{i}.mlp.act.approximator"
#             try:
#                 model_state_dict[f"{base_key}.0.weight"].copy_(ugo_ckpt['approximator.0.weight'])
#                 model_state_dict[f"{base_key}.0.bias"].copy_(ugo_ckpt['approximator.0.bias'])
#                 model_state_dict[f"{base_key}.2.weight"].copy_(ugo_ckpt['approximator.2.weight'])
#                 model_state_dict[f"{base_key}.2.bias"].copy_(ugo_ckpt['approximator.2.bias'])
#             except KeyError as e:
#                 pass # 에러 로그는 생략하거나 Rank 0만 출력

#         # -------------------------------------------------------------------------
#         # 2. 메인 체크포인트 로드 [모든 Rank 실행]
#         # -------------------------------------------------------------------------
#         checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False) if not args.eval else torch.load(args.resume, map_location='cpu', weights_only=False)
#         checkpoint_model = checkpoint['model']
        
#         # 불필요한 키 삭제 로직
#         state_dict = model.state_dict()
#         for k in ['head.weight', 'head.bias']:
#             if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#                 del checkpoint_model[k]

#         keys_to_del = [k for k in checkpoint_model.keys() if "mlp.act.0.s" in k]
#         for k in keys_to_del:
#             del checkpoint_model[k]

#         interpolate_pos_embed(model, checkpoint_model)
#         model.load_state_dict(checkpoint_model, strict=False)

#         # -------------------------------------------------------------------------
#         # 3. SNN 변환 [핵심: 모든 Rank에서 실행되어야 함!]
#         # 절대 if args.rank == 0: 안에 넣지 마세요.
#         # -------------------------------------------------------------------------
#         model = SNNWrapper(
#             ann_model=model, 
#             cfg=None, 
#             time_step=args.time_step, 
#             Encoding_type=args.encoding_type, 
#             level=args.level, 
#             neuron_type=args.neuron_type, 
#             model_name=args.model, 
#             is_softmax=not args.remove_softmax
#         )
        
#         if args.ratio > 0.0:
#             set_sparsity_weight(model)
#             cal_sparsity(model)

#         # GPU 이동 (모든 Rank 실행)
#     model.to(device)

# #  # ====================================================
# #         # [Rank 0 전용] 분석 및 NPY Hook 등록
# #         # ====================================================
# #     if args.rank == 0:
# #             print("\n[Analysis] Measuring Spike Firing Rates...")
# #             from spike_quan_layer import IFNeuron
# #             # from utils import MeasureSpikeRateHook, SaveOutputNpyHook

# #             hook_instances = [] 
# #             handle_list = []
            
# #             # 1. 분석용 훅 등록
# #             for name, module in model.named_modules():
# #                 if isinstance(module, IFNeuron):
# #                     hook = MeasureSpikeRateHook(name)
# #                     handle = module.register_forward_hook(hook)
# #                     hook_instances.append(hook)
# #                     handle_list.append(handle)
            
# #             # [수정 핵심] 메모리 초기화 및 평가 모드 전환
# #             torch.cuda.empty_cache()
# #             model.eval()
            
# #             # [수정 핵심] with torch.no_grad() 필수!
# #             # 이걸 안 하면 T=32일 때 12개 블록의 모든 미분 그래프가 쌓여서 OOM 발생
# #             with torch.no_grad():
# #                 for images, target in data_loader_val:
# #                     images = images.to(device, non_blocking=True)
# #                     print(f"Running Inference for Analysis... (Batch Size: {images.shape[0]})")
# #                     model(images) 
# #                     break # 1개 배치만 보면 되므로 break
            
# #             # 2. 결과 계산
# #             spike_rates = {} 
# #             for hook in hook_instances:
# #                 spike_rates[hook.name] = hook.compute_rate()

# #             # 3. 분석용 훅 제거 (메모리 해제)
# #             for h in handle_list:
# #                 h.remove()
            
# #             # 훅 인스턴스 참조 해제
# #             del hook_instances, handle_list
# #             torch.cuda.empty_cache() # 분석 끝났으니 메모리 청소

# #             # 4. 상태 리셋
# #             print("[Info] Resetting IFNeuron states...")
# #             for name, module in model.named_modules():
# #                 if isinstance(module, IFNeuron):
# #                     module.reset()

# #             # 5. 그래프 그리기
# #             if spike_rates:
# #                 layers = list(spike_rates.keys())
# #                 rates = list(spike_rates.values())

# #                 plt.figure(figsize=(20, 8))
# #                 plt.bar(layers, rates, color='skyblue', edgecolor='black')
# #                 plt.axhline(y=1.0, color='red', linestyle='--', label='Max (100%)')
# #                 plt.axhline(y=0.0, color='black', linestyle='-', label='Dead (0%)')
# #                 plt.axhline(y=0.1, color='green', linestyle='--', label='Target (~10%)')
# #                 plt.title(f"SNN Spike Firing Rates (Model: {args.model})", fontsize=16)
# #                 plt.xlabel("Layer Name", fontsize=12)
# #                 plt.ylabel("Firing Rate", fontsize=12)
# #                 plt.xticks(rotation=90, fontsize=8) 
# #                 plt.legend()
# #                 plt.tight_layout()
                
# #                 save_path = os.path.join(args.log_dir, "spike_rate_analysis.png")
# #                 plt.savefig(save_path)
# #                 print(f"[Done] Graph saved to: {save_path}")

# #             # 6. .npy 파일 저장용 Hook 등록 (분석 종료 후)
# #             save_dir = os.path.join(args.log_dir, "numpy_snn")
# #             os.makedirs(save_dir, exist_ok=True)
# #             print(f"SNN Hook registered! Saving .npy files to {save_dir}")
            
# #             for name, module in model.named_modules():
# #                 if isinstance(module, (nn.Linear, nn.Conv2d)): 
# #                     hook_instance = SaveOutputNpyHook(name, save_dir, max_batches=1)
# #                     module.register_forward_hook(hook_instance)
# #                 elif isinstance(module, IFNeuron):
# #                     hook_instance = SaveOutputNpyHook(name + "_post", save_dir, max_batches=1)
# #                     module.register_forward_hook(hook_instance)
#     model_without_ddp = model if args.mode != "SNN" else model.model
#     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     if args.rank == 0:
#         print("Model = %s" % str(model_without_ddp))
#         print('number of params (M): %.2f' % (n_parameters / 1.e6))

#     eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
#     if args.lr is None: 
#         args.lr = args.blr * eff_batch_size / 256

#     # Distributed Data Parallel 설정
#     if args.distributed:
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
#         # DDP로 감싸지면 .module을 통해 접근해야 함
#         model_without_ddp = model.module if args.mode != "SNN" else model.module.model

#     # Optimizer 설정
#     param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
#         no_weight_decay_list=model_without_ddp.no_weight_decay(),
#         layer_decay=args.layer_decay
#     )
#     optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
#     loss_scaler = NativeScaler()

#     if mixup_fn is not None:
#         # smoothing is handled with mixup label transform
#         criterion = SoftTargetCrossEntropy()
#     elif args.smoothing > 0.:
#         criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
#     else:
#         criterion = torch.nn.CrossEntropyLoss()

#     print("criterion = %s" % str(criterion))

#     if args.mode != "SNN":
#         misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

#     if args.eval:
#         if args.energy_eval:
#             from energy_consumption_calculation import get_model_complexity_info
#             ts1 = time.time()
#             Nops, Nparams = get_model_complexity_info(model, (3, 224, 224), data_loader_val,ost = open(f"{args.log_dir}/energy_info.txt","w+"), as_strings=True, print_per_layer_stat=True, verbose=True, syops_units='Mac', param_units=' ', output_precision=3)
#             print("Nops: ", Nops)
#             print("Nparams: ", Nparams)
#             t_cost = (time.time() - ts1) / 60
#             print(f"Time cost: {t_cost} min")
#         else:
#             test_stats = evaluate(data_loader_val, model, device, args)
#         # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
#         # for k, v in test_stats.items():
#         #     print(k, v)
#             if args.mode == "SNN" and misc.is_main_process():
#                 for k, v in test_stats.items():
#                     print(k, v)
#                 with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
#                     json.dump(test_stats, f)
#         exit(0)

#     print(f"Start training for {args.epochs} epochs")
#     start_time = time.time()
#     max_accuracy = 0.0
#     for epoch in range(args.start_epoch, args.epochs):
#         if args.distributed:
#             data_loader_train.sampler.set_epoch(epoch)
#         train_stats = train_one_epoch(
#             model, criterion, data_loader_train,
#             optimizer, device, epoch, loss_scaler,
#             args.clip_grad, mixup_fn,
#             log_writer=log_writer,
#             args=args
#         )
#         if args.output_dir:
#             misc.save_model(
#                 args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
#                 loss_scaler=loss_scaler, epoch=epoch)

#         test_stats = evaluate(data_loader_val, model, device, args)
#         print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
#         max_accuracy = max(max_accuracy, test_stats["acc1"])
#         print(f'Max accuracy: {max_accuracy:.2f}%')

#         if log_writer is not None:
#             log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
#             log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
#             log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
#             if args.wandb:
#                 epoch_1000x = int(((len(data_loader_train)-1) / len(data_loader_train) + epoch) * 1000)
#                 wandb.log({'test_acc1_curve': test_stats['acc1']}, step=epoch_1000x)
#                 wandb.log({'test_acc5_curve': test_stats['acc5']}, step=epoch_1000x)
#                 wandb.log({'test_loss_curve': test_stats['loss']}, step=epoch_1000x)
#                 if args.mode == "SNN":
#                     for t in range(model.max_T):
#                         wandb.log({'acc1@{}_curve'.format(t+1): test_stats['acc@{}'.format(t+1)]}, step=epoch_1000x)

#         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
#                         **{f'test_{k}': v for k, v in test_stats.items()},
#                         'epoch': epoch,
#                         'n_parameters': n_parameters}

#         if args.output_dir and misc.is_main_process():
#             if log_writer is not None:
#                 log_writer.flush()
#             with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
#                 f.write(json.dumps(log_stats) + "\n")

#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print('Training time {}'.format(total_time_str))


# if __name__ == '__main__':
#     args = get_args_parser()
#     args = args.parse_args()
#     if args.output_dir:
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#         args.output_dir = os.path.join(args.output_dir,
#                                        "{}_{}_{}_{}_{}_act{}_weightbit{}".format(args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level,args.weight_quantization_bit))
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#         print(args.output_dir)
#     main(args)
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from spike_quan_wrapper import myquan_replace, SNNWrapper, myquan_replace_QANN

import models_vit
import wandb
import matplotlib.pyplot as plt
from engine_finetune import train_one_epoch, evaluate, unstruct_prune

import warnings
from functools import partial
from copy import deepcopy

warnings.filterwarnings("ignore", category=UserWarning)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--print_freq', default=1000, type=int,
                        help='print_frequency')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--project_name', default='T-SNN', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--encoding_type', default="analog", type=str,
                        help='encoding type for snn')
    parser.add_argument('--time_step', default=2000, type=int,
                        help='time-step for snn')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--act_layer', type=str, default="relu",
                        help='Using ReLU or GELU as activation')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--dataset', default='imagenet', type=str,
                        help='dataset name')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--define_params', action='store_true')
    parser.add_argument('--mean', nargs='+', type=float)
    parser.add_argument('--std', nargs='+', type=float)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--energy_eval', action='store_true',
                        help='Perform evaluation with energy consumption')
    parser.add_argument('--wandb', action='store_true',
                        help='Using wandb or not')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # training mode
    parser.add_argument('--mode', default="ANN", type=str,
                        help='the running mode of the script["ANN", "QANN_PTQ", "QANN_QAT", "SNN"]')
    # prune
    parser.add_argument('--ratio', default=0.0, type=float,
                        help='the ratio of unstructure prune')
    # LSQ quantization
    parser.add_argument('--level', default=32, type=int,
                        help='the quantization levels')
    parser.add_argument('--weight_quantization_bit', default=32, type=int, help="the weight quantization bit")
    parser.add_argument('--neuron_type', default="ST-BIF", type=str, help='neuron type["ST-BIF", "IF"]')
    parser.add_argument('--remove_softmax', action='store_true', help='need softmax or not')
    parser.add_argument('--gelu_path', default=r'/home/hyuntaek/STA/premodels/distilled_gelu_64.pth', type=str)
    
    return parser


spike_rates = {}
class MeasureSpikeRateHook:
    def __init__(self, name):
        self.name = name
        self.total_spikes = 0.0
        self.total_elements = 0.0

    def __call__(self, module, input, output):
        if isinstance(output, torch.Tensor):
            # 1. 스파이크 개수 누적 (+= 사용)
            self.total_spikes += torch.count_nonzero(output).item()
            
            # 2. 전체 요소 개수 누적 (+= 사용)
            self.total_elements += output.numel()

    # 나중에 결과를 계산하기 위한 함수
    def compute_rate(self):
        if self.total_elements > 0:
            return self.total_spikes / self.total_elements
        return 0.0
    

class SaveOutputNpyHook:
    def __init__(self, module_name, save_dir, max_batches=1):
        self.module_name = module_name
        self.save_dir = save_dir
        self.max_batches = max_batches
        self.call_count = 0

    def __call__(self, module, input, output):
        if self.call_count >= self.max_batches:
            return

        if isinstance(output, torch.Tensor):
            # 1. CPU로 이동 및 Detach
            tensor_val = output.detach().cpu()
            
            # 2. 파일명 안전하게 변환 (특수문자 제거)
            safe_name = self.module_name.replace("model.", "").replace(".linear", "").replace(".conv", "")
            
            # 3. .npy 파일로 저장 (경로: save_dir/레이어이름.npy)
            # SNN의 경우 TimeStep 차원이 있다면 평균을 내서 저장할지, 전체를 저장할지 결정해야 함
            # 보통 비교를 위해 SNN은 TimeStep 차원에 대해 평균(Mean)을 내서 QANN과 차원을 맞춥니다.
            
            if len(tensor_val.shape) == 4 and tensor_val.shape[0] != args.batch_size: 
                 # SNN 짐작: [T, B, C, ...] 형태라면 T(0번축)에 대해 평균
                 # 만약 [B, T, C, ...] 형태라면 1번축 평균. 모델 구조에 따라 확인 필요.
                 # 여기서는 T가 0번 축이라고 가정 (SpikeZIP 보통 구조)
                 save_val = tensor_val.mean(dim=0).numpy()
            else:
                 save_val = tensor_val.numpy()

            file_path = os.path.join(self.save_dir, f"{safe_name}.npy")
            np.save(file_path, save_val)
            print(f"Saved {file_path} shape={save_val.shape}")
        
        self.call_count += 1

def set_sparsity_weight(model):
    for name, m in model.named_modules():
        if name.count("proj")>0 or name.count("fc2")>0:
            if isinstance(m,torch.nn.Sequential) and isinstance(m[0],torch.nn.Linear):
                m[0].weight.data = m[0].weight_mask
            elif isinstance(m,torch.nn.Linear):
                m.weight.data = m.weight_mask
    
def cal_sparsity(model):
    zero_number = 0
    total_bumber = 0
    for name, m in model.named_modules():
        if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
            zero_number = zero_number + torch.sum(m.weight==0)
            total_bumber = total_bumber + m.weight.numel()

    print("prune finish!!!!! global sparsity:",(zero_number/total_bumber)*100)

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 고정 시트 설정을 통한 재현성 확보
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed 환경 가정
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Distributed evaluation with dataset not divisible by process number.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 로깅 설정
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        args.log_dir = os.path.join(args.log_dir,
                                    "{}_{}_{}_{}_{}_act{}_weightbit{}".format(args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level,args.weight_quantization_bit))
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        if args.wandb:
            wandb.init(config=args, project=args.project_name,
                       name="{}_{}_{}_{}_{}_act{}_weightbit{}".format(args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level,args.weight_quantization_bit),
                       dir=args.output_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Mixup 설정
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    activation = nn.ReLU if args.act_layer == "relu" else nn.GELU

    # 모델 초기화 (Vision Transformer)
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        drop_rate = args.drop_rate,
        global_pool=args.global_pool if "vit_small" not in args.model else False,
        act_layer=activation,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    if args.finetune and not args.eval and not (args.mode == "SNN") and not (args.mode == "QANN-QAT" and args.eval):
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    if args.rank == 0:
        print("======================== ANN model ========================")
        f = open(f"{args.log_dir}/ann_model_arch.txt","w+")
        f.write(str(model))
        f.close()
    if args.mode.count("QANN") > 0:
        myquan_replace_QANN(model, args.level, args.weight_quantization_bit)
        
        # [평가 모드일 때만 실행되는 블록]
        if args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False) if not args.eval else torch.load(args.resume, map_location='cpu', weights_only=False)
            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            if args.rank == 0:
                save_dir = os.path.join(args.log_dir, "numpy_qann")
                os.makedirs(save_dir, exist_ok=True)
                print(f"Hook registered! Saving .npy files to {save_dir}")

                # [수정된 로직] 모든 모듈을 순회하며 대상 레이어 및 양자화 레이어 탐색
                from spike_quan_layer import MyQuan # 클래스 타입 체크용
                for name, module in model.named_modules():
                    # (1) 일반 연산 레이어 저장
                    if isinstance(module, (nn.Linear, nn.Conv2d)): 
                        hook_instance = SaveOutputNpyHook(name, save_dir, max_batches=1)
                        module.register_forward_hook(hook_instance)
                    
                    # (2) 양자화 레이어(MyQuan) 저장 - 이름에 '_post' 추가
                    elif isinstance(module, MyQuan):
                        hook_instance = SaveOutputNpyHook(name + "_post", save_dir, max_batches=1)
                        module.register_forward_hook(hook_instance)
                    elif isinstance(module, (nn.GELU, nn.ReLU)):
                            # 파일명을 직관적으로 (예: blocks.11.mlp.act_act.npy)
                            hook_name = name.replace(".1", "") + "_act"
                            hook_instance = SaveOutputNpyHook(hook_name, save_dir, max_batches=1)
                            module.register_forward_hook(hook_instance)
        # [공통] 모델 구조 저장 및 출력 (Eval/Train 모두 실행)
        if args.rank == 0:
            print("======================== QANN model =======================")
            f = open(f"{args.log_dir}/qann_model_arch.txt","w+")
            f.write(str(model))
            f.close()
    elif args.mode == "SNN":
        # 1. 모델 구조 변경 (ANN -> QANN 구조로 선변환)
        myquan_replace(model, args.level, args.weight_quantization_bit)

        # 2. GELU 근사(Approximator) 가중치 로드
        ugo_path = args.gelu_path if hasattr(args, 'gelu_path') else '/home/hyuntaek/STA/premodels/distilled_gelu_64.pth'
        if os.path.exists(ugo_path):
            ugo_ckpt = torch.load(ugo_path, map_location='cpu')
            model_state_dict = model.state_dict()
            for i in range(12): 
                # [수정 1] UGO가 Sequential의 1번 인덱스로 이동했으므로 경로에 '.1.' 추가
                base_key = f"blocks.{i}.mlp.act.1.approximator" 
                try:
                    model_state_dict[f"{base_key}.0.weight"].copy_(ugo_ckpt['approximator.0.weight'])
                    model_state_dict[f"{base_key}.0.bias"].copy_(ugo_ckpt['approximator.0.bias'])
                    model_state_dict[f"{base_key}.2.weight"].copy_(ugo_ckpt['approximator.2.weight'])
                    model_state_dict[f"{base_key}.2.bias"].copy_(ugo_ckpt['approximator.2.bias'])
                except KeyError as e:
                    pass
            print(f"==> [SNN] Loaded Approximator weights from {ugo_path}")

        # 3. 메인 체크포인트 로드 (Resume 또는 Finetune)
        checkpoint_path = args.resume if (args.resume and os.path.exists(args.resume)) else args.finetune
        
        if checkpoint_path:
            print(f"==> [SNN] Loading main checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint_model = checkpoint['model']

            # [수정 2] MyQuan이 복구되었으므로 양자화 스텝 파라미터(act.0.s) 삭제 코드를 제거합니다!
            # 기존의 keys_to_del 관련 for 루프는 통째로 지웁니다.

            # 위치 임베딩 보간 및 가중치 로드
            interpolate_pos_embed(model, checkpoint_model)
            
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(f"==> [SNN] Checkpoint load result: {msg}")

        # 4. SNN 래핑 (Time-step 차원 확장 및 뉴런 주입)
        model = SNNWrapper(
            ann_model=model, 
            cfg=None, 
            time_step=args.time_step, 
            Encoding_type=args.encoding_type, 
            level=args.level, 
            neuron_type=args.neuron_type, 
            model_name=args.model, 
            is_softmax=not args.remove_softmax
        )
        
        if args.ratio > 0.0:
            set_sparsity_weight(model)
            cal_sparsity(model)

    model.to(device)

    # DDP 및 옵티마이저 설정
    model_without_ddp = model if args.mode != "SNN" else model.model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module if args.mode != "SNN" else model.module.model

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # 손실 함수 설정
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.mode == "ANN": # 또는 일반적인 ANN 학습 모드일 때만 실행
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    else:
        print(f"==> [Info] {args.mode} mode detected. Skipping strict misc.load_model to prevent architecture mismatch.")
    # 평가 모드 실행
    if args.eval:
        if args.energy_eval:
            from energy_consumption_calculation import get_model_complexity_info
            Nops, Nparams = get_model_complexity_info(model, (3, 224, 224), data_loader_val, ost=open(f"{args.log_dir}/energy_info.txt","w+"), as_strings=True)
        else:
            test_stats = evaluate(data_loader_val, model, device, args)
            if args.mode == "SNN" and misc.is_main_process():
                print(test_stats)
                with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
                    json.dump(test_stats, f)
        exit(0)

    # 학습 루프
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, 
            loss_scaler, args.clip_grad, mixup_fn, log_writer=log_writer, args=args
        )
        
        if args.output_dir:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, 
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device, args)
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Accuracy: {test_stats['acc1']:.1f}%, Max accuracy: {max_accuracy:.2f}%")

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            if args.wandb:
                epoch_1000x = int(((len(data_loader_train)-1) / len(data_loader_train) + epoch) * 1000)
                wandb.log({'test_acc1_curve': test_stats['acc1']}, step=epoch_1000x)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Training time {}'.format(total_time))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.output_dir = os.path.join(args.output_dir,
                                       "{}_{}_{}_{}_{}_act{}_weightbit{}".format(args.project_name, args.model, args.dataset, args.act_layer, args.mode, args.level,args.weight_quantization_bit))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(args.output_dir)
    main(args)