# import jittor as jt
# from jittor import nn
# from jittor import Module
# from jittor import init
import torch
import torch.nn as nn

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

class PreNormResidual0(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        #x = x.permute(0,3,1,2)
        x0 = x
        reshaped_input = x.reshape(x.size(0) * x.size(2) * x.size(3), -1)

        temp = self.norm(reshaped_input)
        temp_reshaped = temp.reshape(x.size(0), x.size(1), x.size(2), x.size(3))
        # norm_out = self.fn(self.norm(reshaped_input)) + reshaped_input
        norm_out = self.fn(temp_reshaped) + x

        return norm_out


class PreNormResidual1(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.GELU(),
            nn.Linear(dim * 3, dim),
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        #x = x.permute(0,3,1,2)
        x0 = x
        reshaped_input = x.reshape(x.size(0) * x.size(2) * x.size(3), -1)

        # temp = self.norm(reshaped_input)
        # temp1 = self.fn(temp)

        norm_out = self.fn(self.norm(reshaped_input)) + reshaped_input
        norm_reshaped = norm_out.reshape(x.size(0), x.size(1), x.size(2), x.size(3))

        return norm_reshaped


def spatial_shift1(x):
    b, w, h, c = x.size()
    # x_temp1 = x[:, :w - 1, :, :c // 4]
    # x_temp2 = x[:, 1:, :, :c // 4]
    # are_equal3 = torch.all(torch.eq(x_temp1, x_temp2))

    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        # (4,3,64,64,32)
        b, k, h, w, c = x_all.shape
        # reshape(b, k, -1, c) 操作会把 (4,3,64,64,32) 变成 (4,3,4096,32)
        x_all = x_all.reshape(b, k, -1, c)  # bs,k,n,c
        # torch.sum(x_all, 1) 会把 (4,3,4096,32) 按着第1维度，3个加起来得到 (4,4096,32)
        # sum_up = torch.sum(x_all, 1)
        a = torch.sum(torch.sum(x_all, 1), 1)  # bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # bs,kc
        hat_a = hat_a.reshape(b, self.k, c)  # bs,k,c
        bar_a = self.softmax(hat_a)  # bs,k,c --> (4,3,32)
        attention = bar_a.unsqueeze(-2)  # bs,k,1,c --> (4,3,1,32)
        out = attention * x_all  # bs,k,n,c --> (4,3,1,32)*(4,3,4096,32) = (4,3,4096,32) --> 相当于在 HW 这 4096 维上没有权重，其他位置有权重做 attention
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


# class S2Attention(nn.Module):
#     def __init__(self, channels=512):
#         super().__init__()
#         self.mlp1 = nn.Linear(channels, channels * 3)
#         self.mlp2 = nn.Linear(channels, channels)
#         self.split_attention = SplitAttention(channels)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         b, h, w, c = x.size()
#         x = x.reshape(b, h * w, c)
#         # 通过 mlp 故意把通道从 c 转成 3c 的
#         x = self.mlp1(x)
#         x = x.reshape(b, h, w, 3 * c)
#
#         # check 一下通道有没有分对
#         # x_check_stack = torch.cat([x[:, :, :, :c], x[:, :, :, c:c * 2], x[:, :, :, c * 2:]], dim=3)
#         # are_equal3 = torch.all(torch.eq(x, x_check_stack ))
#
#         x1 = spatial_shift1(x[:, :, :, :c])
#         x2 = spatial_shift2(x[:, :, :, c:c * 2])
#         x3 = x[:, :, :, c * 2:]
#         x_all = torch.stack([x1, x2, x3], 1)  # (4,3,64,64,32)
#         a = self.split_attention(x_all)  # (4,64,64,32)
#         x = self.mlp2(a)
#         x = x.permute(0, 3, 1, 2)
#         return x


class S2Attention(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        b, h, w, c = x.size()
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = x[:,:,:,c*2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)   # (4,9,9,192)
        # 不确定要不要调顺序
        x = x.permute(0, 3, 1, 2)
        nnn=1
        return x

class S2Block(nn.Module):
    def __init__(self, d_model, depth, expansion_factor = 4, dropout = 0.):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual0(d_model, S2Attention(d_model)),
                PreNormResidual1(d_model, nn.Sequential(
                    nn.Linear(d_model, d_model * expansion_factor),
                    nn.GELU(),
                    # nn.Dropout(dropout),
                    nn.Linear(d_model * expansion_factor, d_model),
                    # nn.Dropout(dropout)
                ))
            ) for _ in range(depth)]
        )

    def forward(self, x):
        #x = x.permute(0, 2, 3, 1)
        x = self.model(x)
        #x = x.permute(0, 3, 1, 2)
        return x

# class S2MLPv2(nn.Module):
#     def __init__(
#         self,
#         image_size=64,
#         patch_size=[4, 2],
#         in_channels=32,
#         num_classes=1000,
#         d_model=[32, 64],
#         depth=[4, 12],
#         expansion_factor = [3, 3],
#     ):
#         image_size = pair(image_size)
#         oldps = [1, 1]
#         for ps in patch_size:
#             ps = pair(ps)
#             assert (image_size[0] % (ps[0] * oldps[0])) == 0, 'image must be divisible by patch size'
#             assert (image_size[1] % (ps[1] * oldps[1])) == 0, 'image must be divisible by patch size'
#             oldps[0] = oldps[0] * ps[0]
#             oldps[1] = oldps[1] * ps[1]
#         assert (len(patch_size) == len(depth) == len(d_model) == len(expansion_factor)), 'patch_size/depth/d_model/expansion_factor must be a list'
#         super().__init__()
#
#         self.stage = len(patch_size)
#         self.stages = nn.Sequential(
#             *[nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
#                 S2Block(d_model[i], depth[i], expansion_factor[i], dropout = 0.)
#             ) for i in range(self.stage)]
#         )
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
#         self.mlp_head = nn.Sequential(
#             nn.Linear(d_model[-1], num_classes)
#         )
#         nnn=1
#
#     def forward(self, x):
#         embedding = self.stages(x)
#         embedding = self.avgpool(embedding)
#         embedding = torch.flatten(embedding, 1)
#         out = self.mlp_head(embedding)
#
#         return out

class S2MLPv2(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=[3],
        in_channels=3,
        num_classes=1000,
        d_model=[32, 64],
        depth=[4, 14],
        expansion_factor = [3, 3],
    ):
        # image_size = pair(image_size)
        # oldps = [1, 1]
        # for ps in patch_size:
        #     ps = pair(ps)
        #     assert (image_size[0] % (ps[0] * oldps[0])) == 0, 'image must be divisible by patch size'
        #     assert (image_size[1] % (ps[1] * oldps[1])) == 0, 'image must be divisible by patch size'
        #     oldps[0] = oldps[0] * ps[0]
        #     oldps[1] = oldps[1] * ps[1]
        # assert (len(patch_size) == len(depth) == len(d_model) == len(expansion_factor)), 'patch_size/depth/d_model/expansion_factor must be a list'
        super().__init__()

        self.stage = len(patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else d_model[i - 1], d_model[i], kernel_size=3, stride=1, padding=1),
                S2Block(d_model[i], depth[i], expansion_factor[i], dropout = 0.)
            ) for i in range(self.stage)]
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(d_model[-1], num_classes)
        # )

        # # 测试网络
        # self.test1 = nn.Conv2d(32, 192, kernel_size=7, stride=7, padding=0, bias=False)
        # self.test2 = S2Attention(channels = 192)
        # self.test3 = S2Block(d_model=192, depth=4)
        # self.test4 = nn.LayerNorm(192)
        #
        # d_model = 192
        # expansion_factor = 3
        # dropout = 0.
        # self.test5 = PreNormResidual1(d_model, nn.Sequential(
        #     nn.Linear(d_model, d_model * expansion_factor),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model * expansion_factor, d_model),
        #     nn.Dropout(dropout)
        # ))

    def forward(self, x):

        # # 测试网络
        # x_test1 = self.test1(x)   # x (4,32,64,64)
        # # x_test2 = self.test2(x_test1)   # x_test1 (4,192,9,9)
        # x_test3 = self.test3(x_test1)
        # reshaped_input = x_test1.view(x_test1.size(0)*x_test1.size(2)*x_test1.size(3), -1)
        # x_test4 = self.test4(reshaped_input)
        # result_tensor = x_test4.view(x_test1.size(0), x_test1.size(1), x_test1.size(2), x_test1.size(3))

        # 直接输出？
        embedding = self.stages(x)   # x (4,32,128,128) --> x (4,32,128,128)

        # embedding = self.avgpool(embedding)   # (4,384,4,4) --> (4,384,1,1)
        # embedding = torch.flatten(embedding, 1)   # (4,384,1,1) --> (4,384)
        # out = self.mlp_head(embedding)   # (4,384) --> (4,1000)

        return embedding