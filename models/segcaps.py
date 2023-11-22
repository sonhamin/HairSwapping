'''
Credit to: https://github.com/ImMrMa/SegCaps.pytorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_same(input, weight, bias=None, stride=[1, 1], dilation=(1, 1), groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)
    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.conv2d(input, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)


def max_pool2d_same(input, kernel_size, stride=1, dilation=1, ceil_mode=False, return_indices=False):
    input_rows = input.size(2)
    out_rows = (input_rows + stride - 1) // stride
    padding_rows = max(0, (out_rows - 1) * stride +
                       (kernel_size - 1) * dilation + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    cols_odd = (padding_rows % 2 != 0)
    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding_rows // 2, dilation=dilation,
                        ceil_mode=ceil_mode, return_indices=return_indices)




class CapsuleLayer(nn.Module):
    def __init__(self, t_0,z_0, op, k, s, t_1, z_1, routing):
        super().__init__()
        self.t_1 = t_1
        self.z_1 = z_1
        self.op = op
        self.k = k
        self.s = s
        self.routing = routing
        self.convs = nn.ModuleList()
        self.t_0=t_0
        for _ in range(t_0):
            if self.op=='conv':
                self.convs.append(nn.Conv2d(z_0, self.t_1*self.z_1, self.k, self.s,padding=1,bias=False))
            else:
                self.convs.append(nn.ConvTranspose2d(z_0, self.t_1 * self.z_1, self.k, self.s,padding=1,output_padding=1))

    def forward(self, u):  # input [N,CAPS,C,H,W]
                
        if u.shape[1]!=self.t_0:
            print('------------')
            print(u.shape)
            print(self.t_0)
            print('------------')            
            raise ValueError("Wrong type of operation for capsule")
        op = self.op
        k = self.k
        s = self.s
        t_1 = self.t_1
        z_1 = self.z_1
        routing = self.routing
        N = u.shape[0]
        H_1=u.shape[3]
        W_1=u.shape[4]
        t_0 = self.t_0

        u_t_list = [u_t.squeeze(1) for u_t in u.split(1, 1)]  # 将cap分别取出来

        u_hat_t_list = []

        for i, u_t in zip(range(self.t_0), u_t_list):  # u_t: [N,C,H,W]
            if op == "conv":
                u_hat_t = self.convs[i](u_t)  # 卷积方式
            elif op == "deconv":
                u_hat_t = self.convs[i](u_t) #u_hat_t: [N,t_1*z_1,H,W]
            else:
                raise ValueError("Wrong type of operation for capsule")
            H_1 = u_hat_t.shape[2]
            W_1 = u_hat_t.shape[3]
            u_hat_t = u_hat_t.reshape(N, t_1,z_1,H_1, W_1).transpose_(1,3).transpose_(2,4)
            u_hat_t_list.append(u_hat_t)    #[N,H_1,W_1,t_1,z_1]
        v=self.update_routing(u_hat_t_list,k,N,H_1,W_1,t_0,t_1,routing)
        return v
    def update_routing(self,u_hat_t_list, k, N, H_1, W_1, t_0, t_1, routing):
        one_kernel = torch.ones(1, t_1, k, k).cuda()  # 不需要学习
        b = torch.zeros(N, H_1, W_1, t_0, t_1).cuda()  # 不需要学习
        b_t_list = [b_t.squeeze(3) for b_t in b.split(1, 3)]
        u_hat_t_list_sg = []
        for u_hat_t in u_hat_t_list:
            u_hat_t_sg=u_hat_t.detach()
            u_hat_t_list_sg.append(u_hat_t_sg)

        for d in range(routing):
            if d < routing - 1:
                u_hat_t_list_ = u_hat_t_list_sg
            else:
                u_hat_t_list_ = u_hat_t_list

            r_t_mul_u_hat_t_list = []
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                # routing softmax (N,H_1,W_1,t_1)
                b_t.transpose_(1, 3).transpose_(2, 3)  #[N,t_1,H_1, W_1]
                b_t_max = torch.nn.functional.max_pool2d(b_t,k,1,padding=1)
                b_t_max = b_t_max.max(1, True)[0]
                c_t = torch.exp(b_t - b_t_max)
                sum_c_t = conv2d_same(c_t, one_kernel, stride=(1, 1))  # [... , 1]
                r_t = c_t / sum_c_t  # [N,t_1, H_1, W_1]
                r_t = r_t.transpose(1, 3).transpose(1, 2)  # [N, H_1, W_1,t_1]
                r_t = r_t.unsqueeze(4)  # [N, H_1, W_1,t_1, 1]
                r_t_mul_u_hat_t_list.append(r_t * u_hat_t)  # [N, H_1, W_1, t_1, z_1]
            p = sum(r_t_mul_u_hat_t_list)  # [N, H_1, W_1, t_1, z_1]
            v = squash(p)
            if d < routing - 1:
                b_t_list_ = []
                for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                    # b_t     : [N, t_1,H_1, W_1]
                    # u_hat_t : [N, H_1, W_1, t_1, z_1]
                    # v       : [N, H_1, W_1, t_1, z_1]
                    # [N,H_1,W_1,t_1]
                    b_t.transpose_(1,3).transpose_(2,1)
                    b_t_list_.append(b_t + (u_hat_t * v).sum(4))
        v.transpose_(1, 3).transpose_(2, 4)
        # print(v.grad)
        return v
    def squash(self, p):
        p_norm_sq = (p * p).sum(-1, True)
        p_norm = (p_norm_sq + 1e-9).sqrt()
        v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
        return v


def update_routing(u_hat_t_list,k,N,H_1,W_1,t_0,t_1,routing):
    one_kernel = torch.ones(1, t_1, k, k).cuda()#不需要学习
    b = torch.zeros(N, H_1, W_1, t_0, t_1 ).cuda()#不需要学习
    b_t_list = [b_t.squeeze(3) for b_t in b.split(1, 3)]
    u_hat_t_list_sg = []
    for u_hat_t in u_hat_t_list:
        u_hat_t_sg = u_hat_t.clone()
        u_hat_t_sg.detach_()
        u_hat_t_list_sg.append(u_hat_t_sg)

    for d in range(routing):
        if d < routing - 1:
            u_hat_t_list_ = u_hat_t_list_sg
        else:
            u_hat_t_list_ = u_hat_t_list

        r_t_mul_u_hat_t_list = []
        for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
            # routing softmax (N,H_1,W_1,t_1)
            b_t.transpose_(1, 3).transpose_(2, 3)
            torch.nn.functional.max_pool2d(b_t,k,)
            b_t_max = max_pool2d_same(b_t, k, 1)
            b_t_max = b_t_max.max(1, True)[0]
            c_t = torch.exp(b_t - b_t_max)
            sum_c_t = conv2d_same(c_t, one_kernel, stride=(1, 1))  # [... , 1]
            r_t = c_t / sum_c_t  # [N,t_1, H_1, W_1]
            r_t = r_t.transpose(1, 3).transpose(1, 2)  # [N, H_1, W_1,t_1]
            r_t = r_t.unsqueeze(4)  # [N, H_1, W_1,t_1, 1]
            r_t_mul_u_hat_t_list.append(r_t * u_hat_t)  # [N, H_1, W_1, t_1, z_1]

        p = sum(r_t_mul_u_hat_t_list)  # [N, H_1, W_1, t_1, z_1]
        v = squash(p)
        if d < routing - 1:
            b_t_list_ = []
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                # b_t     : [N, t_1,H_1, W_1]
                # u_hat_t : [N, H_1, W_1, t_1, z_1]
                # v       : [N, H_1, W_1, t_1, z_1]
                b_t = b_t.transpose(1, 3).transpose(1, 2)  # [N,H_1,W_1,t_1]
                b_t_list_.append(b_t + (u_hat_t * v).sum(4))
            b_t_list = b_t_list_
        v.transpose_(1,3).transpose_(2,4)
    #print(v.grad)
    return v

def squash( p):
    p_norm_sq = (p * p).sum(-1, True)
    p_norm = (p_norm_sq + 1e-9).sqrt()
    v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
    return v











class SegCaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, padding=1,bias=False),
            #CapsuleLayer(1, 3, "conv", k=3, s=1, t_1=3, z_1=16, routing=1),
        )
        self.step_1 = nn.Sequential(  # 1/2
            CapsuleLayer(1, 16, "conv", k=3, s=2, t_1=2, z_1=16, routing=1),
            CapsuleLayer(2, 16, "conv", k=3, s=1, t_1=4, z_1=16, routing=3),
        )
        self.step_2 = nn.Sequential(  # 1/4
            CapsuleLayer(4, 16, "conv", k=3, s=2, t_1=4, z_1=32, routing=3),
            CapsuleLayer(4, 32, "conv", k=3, s=1, t_1=8, z_1=32, routing=3)
        )
        self.step_3 = nn.Sequential(  # 1/8
            CapsuleLayer(8, 32, "conv", k=3, s=2, t_1=8, z_1=64, routing=3),
            CapsuleLayer(8, 64, "conv", k=3, s=1, t_1=8, z_1=32, routing=3)
        )
        
        
        

        
        self.conv_1_hair = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, padding=1,bias=False),
            #CapsuleLayer(1, 3, "conv", k=3, s=1, t_1=3, z_1=16, routing=1),
            

        )
        self.step_1_hair = nn.Sequential(  # 1/2
            CapsuleLayer(1, 16, "conv", k=3, s=2, t_1=2, z_1=16, routing=1),
            CapsuleLayer(2, 16, "conv", k=3, s=1, t_1=4, z_1=16, routing=3),
        )
        self.step_2_hair = nn.Sequential(  # 1/4
            CapsuleLayer(4, 16, "conv", k=3, s=2, t_1=4, z_1=32, routing=3),
            CapsuleLayer(4, 32, "conv", k=3, s=1, t_1=8, z_1=32, routing=3)
        )
        self.step_3_hair = nn.Sequential(  # 1/8
            CapsuleLayer(8, 32, "conv", k=3, s=2, t_1=8, z_1=64, routing=3),
            CapsuleLayer(8, 64, "conv", k=3, s=1, t_1=8, z_1=32, routing=3)
        )        
        
        
        
        
        self.step_4 = CapsuleLayer(16, 32, "deconv", k=3, s=2, t_1=8, z_1=32, routing=3)
        self.step_5 = CapsuleLayer(24, 32, "conv", k=3, s=1, t_1=8, z_1=32, routing=3)

        self.step_6 = CapsuleLayer(8, 32, "deconv", k=3, s=2, t_1=8, z_1=16, routing=3)
        self.step_7 = CapsuleLayer(16, 16, "conv", k=3, s=1, t_1=4, z_1=8, routing=3)
        
        self.step_8 = CapsuleLayer(4, 8, "deconv", k=3, s=2, t_1=4, z_1=16, routing=3)
        self.step_10 = CapsuleLayer(6, 16, "conv", k=3, s=1, t_1=1, z_1=16, routing=3)

        
        self.Conv_1x1 = nn.Conv2d(16,3,kernel_size=1,stride=1,padding=0)
        self.relu = nn.ReLU()
        
    def forward(self, x, hair):
        x = self.conv_1(x)
        x.unsqueeze_(1)
        #print(x.shape)
        #x = self.conv_1(x)

        skip_1 = x  # [N,1,16,H,W]
        x = self.step_1(x)

        skip_2 = x  # [N,4,16,H/2,W/2]
        x = self.step_2(x)

        skip_3 = x  # [N,8,32,H/4,W/4]
        x = self.step_3(x)  # [N,8,32,H/8,W/8]
        #print(x.shape)

        hair = self.conv_1_hair(hair)
        hair.unsqueeze_(1)
        #hair = self.conv_1_hair(hair)

        skip_1_hair = hair  # [N,1,16,H,W]
        hair = self.step_1_hair(hair)

        skip_2_hair = hair  # [N,4,16,H/2,W/2]
        hair = self.step_2_hair(hair)

        skip_3_hair = hair  # [N,8,32,H/4,W/4]
        hair = self.step_3_hair(hair)  # [N,8,32,H/8,W/8]
        #print(hair.shape)
        
        comb = torch.cat((x, hair), 1)
        #print(comb.shape)
        

        x = self.step_4(comb)  # [N,8,32,H/4,W/4]
        
        #print(x.shape)
        #print(skip_3.shape)
        #print(skip_3_hair.shape)
        
        comb_skip_3 = torch.cat((skip_3, skip_3_hair), 1)
        x = torch.cat((x, comb_skip_3), 1)  # [N,16,32,H/4,W/4]
        x = self.step_5(x)  # [N,4,32,H/4,W/4]

        x = self.step_6(x)  # [N,4,16,H/2,W/2]

        comb_skip_2 = torch.cat((skip_2, skip_2_hair), 1)
        x = torch.cat((x, comb_skip_2), 1)   # [N,8,16,H/2,W/2]
        x = self.step_7(x)  # [N,4,16,H/2,W/2]
        
        #print(x.shape)
        x = self.step_8(x)  # [N,2,16,H,W]
        
        comb_skip_1 = torch.cat((skip_1, skip_1_hair), 1)
  
        x=torch.cat((x, comb_skip_1),1)
        x=self.step_10(x)
        #print(x.shape)
        x = x.squeeze(1)
        x = self.Conv_1x1(x)
        x = self.relu(x)

        return x
        
        #v_lens = self.compute_vector_length(x)
        #v_lens = self.tan(v_lens)
        #return v_lens
    def compute_vector_length(self, x):
        #print('s')
        #print(x.shape)
        
        out1 = x[:, 0:1, :, :, :]
        out1.squeeze_(1)
        
        out2 = x[:, 1:2, :, :, :]
        out2.squeeze_(1)
        
        out3 = x[:, 2:3, :, :, :]
        out3.squeeze_(1)
        
        out1 = (out1.pow(2)).sum(1, True)+1e-9
        out1=out1.sqrt()
        
        out2 = (out2.pow(2)).sum(1, True)+1e-9
        out2=out2.sqrt()
        
        out3 = (out3.pow(2)).sum(1, True)+1e-9
        out3=out3.sqrt()
        
        
        out = torch.cat((out1, out2, out3), 1)
        
        return out