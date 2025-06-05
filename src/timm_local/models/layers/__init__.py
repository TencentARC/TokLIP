# NOTE timm.models.layers is DEPRECATED, please use timm.layers, this is here to reduce breakages in transition
from timm_local.layers.activations import *
from timm_local.layers.adaptive_avgmax_pool import \
    adaptive_avgmax_pool2d, select_adaptive_pool2d, AdaptiveAvgMaxPool2d, SelectAdaptivePool2d
from timm_local.layers.attention_pool2d import AttentionPool2d, RotAttentionPool2d, RotaryEmbedding
from timm_local.layers.blur_pool import BlurPool2d
from timm_local.layers.classifier import ClassifierHead, create_classifier
from timm_local.layers.cond_conv2d import CondConv2d, get_condconv_initializer
from timm_local.layers.config import is_exportable, is_scriptable, is_no_jit, set_exportable, set_scriptable, set_no_jit,\
    set_layer_config
from timm_local.layers.conv2d_same import Conv2dSame, conv2d_same
from timm_local.layers.conv_bn_act import ConvNormAct, ConvNormActAa, ConvBnAct
from timm_local.layers.create_act import create_act_layer, get_act_layer, get_act_fn
from timm_local.layers.create_attn import get_attn, create_attn
from timm_local.layers.create_conv2d import create_conv2d
from timm_local.layers.create_norm import get_norm_layer, create_norm_layer
from timm_local.layers.create_norm_act import get_norm_act_layer, create_norm_act_layer, get_norm_act_layer
from timm_local.layers.drop import DropBlock2d, DropPath, drop_block_2d, drop_path
from timm_local.layers.eca import EcaModule, CecaModule, EfficientChannelAttn, CircularEfficientChannelAttn
from timm_local.layers.evo_norm import EvoNorm2dB0, EvoNorm2dB1, EvoNorm2dB2,\
    EvoNorm2dS0, EvoNorm2dS0a, EvoNorm2dS1, EvoNorm2dS1a, EvoNorm2dS2, EvoNorm2dS2a
from timm_local.layers.fast_norm import is_fast_norm, set_fast_norm, fast_group_norm, fast_layer_norm
from timm_local.layers.filter_response_norm import FilterResponseNormTlu2d, FilterResponseNormAct2d
from timm_local.layers.gather_excite import GatherExcite
from timm_local.layers.global_context import GlobalContext
from timm_local.layers.helpers import to_ntuple, to_2tuple, to_3tuple, to_4tuple, make_divisible, extend_tuple
from timm_local.layers.inplace_abn import InplaceAbn
from timm_local.layers.linear import Linear
from timm_local.layers.mixed_conv2d import MixedConv2d
from timm_local.layers.mlp import Mlp, GluMlp, GatedMlp, ConvMlp
from timm_local.layers.non_local_attn import NonLocalAttn, BatNonLocalAttn
from timm_local.layers.norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d
from timm_local.layers.norm_act import BatchNormAct2d, GroupNormAct, convert_sync_batchnorm
from timm_local.layers.padding import get_padding, get_same_padding, pad_same
from timm_local.layers.patch_embed import PatchEmbed
from timm_local.layers.pool2d_same import AvgPool2dSame, create_pool2d
from timm_local.layers.squeeze_excite import SEModule, SqueezeExcite, EffectiveSEModule, EffectiveSqueezeExcite
from timm_local.layers.selective_kernel import SelectiveKernel
from timm_local.layers.separable_conv import SeparableConv2d, SeparableConvNormAct
from timm_local.layers.split_attn import SplitAttn
from timm_local.layers.split_batchnorm import SplitBatchNorm2d, convert_splitbn_model
from timm_local.layers.std_conv import StdConv2d, StdConv2dSame, ScaledStdConv2d, ScaledStdConv2dSame
from timm_local.layers.test_time_pool import TestTimePoolHead, apply_test_time_pool
from timm_local.layers.trace_utils import _assert, _float_to_int
from timm_local.layers.weight_init import trunc_normal_, trunc_normal_tf_, variance_scaling_, lecun_normal_

import warnings
warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
