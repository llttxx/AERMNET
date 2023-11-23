import torch
import torch.nn as nn
import torchvision
from utils_semantic_contrast import *
import math
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Attention  传入初始化为一组图像特征的 3个矩阵和全连接层维度    返回注意力上的注意图像特征



class X_Linear_block(nn.Module):
    def __init__(self,N=196,D_q=1024,D_k=1024,D_v=1024,D_c=512,D_B=1024):
        super(X_Linear_block, self).__init__()
        # Q=torch.zeros(batch, D_q)
        # K=torch.zeros(batch, N, D_k)
        # V=torch.zeros(batch, N, D_v)

        self.N=N
        self.D_q = D_q
        self.D_k = D_k
        self.D_v = D_v
        self.D_c = D_c
        self.D_B = D_B

        self.ELU=nn.ELU()
        self.RELU=nn.ReLU()
        self.softMax = nn.Softmax(dim=1)
        self.sigMoid=nn.Sigmoid()
        self.laynorm = nn.LayerNorm(D_B)
        self.W_k=nn.Linear(D_k,D_B)
        self.W_q_k=nn.Linear(D_q,D_B)

        self.W_k_B=nn.Linear(D_B,D_c)
        self.W_b=nn.Linear(D_c,1)
        self.W_e=nn.Linear(D_c,D_B)
        self.W_v=nn.Linear(D_v,D_B)
        self.W_v_q=nn.Linear(D_q,D_B)
        self.W_k_m = nn.Linear(D_v + D_k, D_k)
        self.W_v_m = nn.Linear(D_v + D_v, D_v)


    def forward(self, Q,K,V):
        batch=Q.size(0)
        B_k=torch.zeros(batch, self.N, self.D_B).to(device) 
        B_k_pie=torch.zeros(batch, self.N, self.D_c).to(device) 
        b_s=torch.zeros(batch,self.N).to(device) 
        B_v=torch.zeros(batch, self.N, self.D_B).to(device) 
        

#         for i in range(self.N):
#             B_k[:,i,:]=torch.mul(self.ELU(self.W_k(K[:,i,:])),self.ELU(self.W_q_k(Q)))
        B_k=torch.mul(self.ELU(self.W_k(K)),self.ELU(self.W_q_k(Q).unsqueeze(1)))
    
        B_k_pie=self.RELU(self.W_k_B(B_k))
        b_s=self.W_b(B_k_pie).squeeze(2)
        

        
        beita_s=self.softMax(b_s)  # (batch,N)
        B_gang=torch.mean(B_k_pie,1)     # (batch,D_c)
        beita_c=self.sigMoid(self.W_e(B_gang))   # (batch,D_B)
         
        B_v=torch.mul(self.W_v(V),self.W_v_q(Q).unsqueeze(1))   #  这里用什么激活函数比较好？  （batch，N，D_B）

        v_MAO=torch.mul(torch.mean(torch.mul(beita_s.unsqueeze(2),B_v),dim=1 ) ,beita_c)  #  （batch，D_B）  若是出问题 ，可能就是这里，X_Linear最后一个环节

        v_MAO_m=torch.zeros_like(K)
        v_MAO_m=v_MAO_m+v_MAO.unsqueeze(1)

        # k(m)_i 更新
        K_m=self.laynorm(self.RELU(self.W_k_m(torch.cat((v_MAO_m,K),dim=2),))+K)
        V_m=self.laynorm(self.RELU(self.W_v_m(torch.cat((v_MAO_m,V),dim=2),))+V)

        return v_MAO , K_m ,V_m


class AOAttention(nn.Module):
    def __init__(self,  D):  # (k,D)
        super(AOAttention, self).__init__()
        self.W_q = nn.Linear(2 * D, D, bias=True)
        self.W_v = nn.Linear(2 * D, D, bias=True)
        self.sigmoid = nn.Sigmoid()

    # # 根据Q,K,计算一批图像 的特征相似性分数  (batch,k)
    def f_similarity(self,target1, behaviored1):  # (batch,1,d)   (batch,k,d)
        scores = torch.zeros(target1.size(0), behaviored1.size(1))
        for j in range(target1.size(0)):
            target = target1[j]
            behaviored = behaviored1[j]
            attention_distribution = []
            for i in range(behaviored.size(0)):
                attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
                attention_distribution.append(attention_score)
            attention_distribution = torch.Tensor(attention_distribution)
            score = attention_distribution / torch.sum(attention_distribution, 0)
            scores[j] = score
        return scores  # 标准化

    # 根据Q,K,V,生成V_
    def Att(self,Q, K, V):  # (b,k,d)  (b,k,d)  (b,k,d)
        V_ = torch.zeros(Q.size(0), Q.size(1), V.size(2))
        V_=V_.to(device) 
        soft = nn.Softmax(dim=1)
        for i in range((Q.size(1))):
            alpha_i_j = soft(self.f_similarity(Q[:, i, :].unsqueeze(1), K))  # (b,k)
            alpha_i_j = alpha_i_j.unsqueeze(2)  # (b,k,1)
            alpha_i_j = alpha_i_j.expand_as(V)  # (b,k,d)
            alpha_i_j=alpha_i_j.to(device)
            V_[:, i, :] = torch.sum(alpha_i_j.mul(V), dim=1)  # (b,d)
        return V_  # (b,k,d)

    def forward(self,Q,K,V):

        V_=self.Att(Q,K,V)

        s = torch.cat((V_, Q), 2)  # 第3个维度concat  b, k,2*D

        image_feats_on_att= self.sigmoid(self.W_v(s)).mul(self.W_q(s))

        return image_feats_on_att


# 输入A ,返回A'
class multi_head_att(nn.Module):
    def __init__(self, D):
        super(multi_head_att, self).__init__()
        self.W_Q = nn.Linear(D, D)
        self.W_K = nn.Linear(D, D)
        self.W_V = nn.Linear(D, D)
        self.softmax = nn.Softmax(dim=2)  #(b,k,d)最后一个维度 特征的维度上进行softmax
        self.layernorm=nn.LayerNorm(D,eps=1e-05, elementwise_affine=True)
        self.AOAttention=AOAttention(D)

    # 多头 8个
    def slice(self, Q, K, V):  # (b,K,1024)
        d = Q.size(2) // 8
        list_Q = []
        list_k = []
        list_v = []

        for i in range(8):
            tempQ=Q[:,:,d * i:d * (i + 1)]
            tempK=K[:,:,d * i:d * (i + 1)]
            tempV=V[:,:,d * i:d * (i + 1)]
            list_Q.append(tempQ)
            list_k.append(tempK)
            list_v.append(tempV)

        return list_Q,list_k,list_v

    def forward(self,B,A):
        list_Q,list_k,list_v=self.slice(self.W_Q(B),self.W_K(A),self.W_V(A))
        head=[]
        for i in range(len(list_v)):

            f_dot_att=self.softmax(list_Q[i].matmul(list_k[i].permute(0,2,1))/8)
            f_dot_att=f_dot_att.matmul(list_v[i])
            head.append(f_dot_att)
        f_mh_att=torch.cat((head[0],head[1],head[2],head[3],head[4],head[5],head[6],head[7]),dim=2)   #多头  （ b，k,D）
        #AOAttention = self.AOAttention(f_mh_att, self.W_Q(B))

        AOAttention = self.AOAttention( self.W_Q(B) ,f_mh_att ,f_mh_att)

        return  self.layernorm(AOAttention+A)





'''
图中的图像EMbed是经过Linear,dropout,加激活函数
K0=V
V0=V
Q0=torch.mean(V,1)


K1=K_···
V1=V_
Q1=Q_

一共M+1 个Q 组成图像级特征， 最后一个时刻的V_是增强的区域级特征  att_feats

M+1个图像级特征concat起来经过Linear  就是全局图像特征 GV_FEATS
        
增强的区域特征 att_feats 在每个时间步 作为 LSTM后的X_Linear   的K,V，Q是LSTM的h       
全局图像特征经过Linear后与上下文语音和词嵌入concat起来

'''

class MogrifierLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mogrify_steps,k=5):
        super(MogrifierLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        
#         self.mogrifier_list = nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(hidden_size, k, bias=False),
#                                                                  torch.nn.Linear(k, input_size,
#                                                                                  bias=True))])  # start with q
#         for i in range(1, mogrify_steps):
#             if i % 2 == 0:
#                 self.mogrifier_list.extend([torch.nn.Sequential(torch.nn.Linear(hidden_size, k, bias=False),
#                                                                 torch.nn.Linear(k, input_size, bias=True))])  # q
#             else:
#                 self.mogrifier_list.extend([torch.nn.Sequential(torch.nn.Linear(input_size, k, bias=False),
#                                                                 torch.nn.Linear(k, hidden_size, bias=True))])  # r
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])  # q
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])  # r

    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i + 1) % 2 == 0:
                h = (2 * torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2 * torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct
    
    
    
    
class Encoder(nn.Module):

    def __init__(self, encoded_image_size=7):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=False)  # pretrained ImageNet ResNet-101
        pretrain_path = '/zengxh_fix/starstar/wzq/My_models/resnet101-5d3b4d8f.pth'
        resnet.load_state_dict(torch.load(pretrain_path))
        # 移除线性层和池层（因为我们没有进行分类）
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # 将图像调整为固定大小以允许输入可变大小的图像
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        #自适应平均池化，输出的特征是(encoded_image_size, encoded_image_size)
        
        # !!!!!!!!!!!!!!!!!
        self.avgpool = nn.AvgPool2d(7)
        # !!!!!!!!!!!!!!!!!
        
        self.adaptive_pool_contrast_feature = nn.AdaptiveAvgPool2d((1, 1))
        self.fine_tune()
        self.adjust = nn.Linear(2048, 512)

    def forward(self, images):
        """
        Forward propagation.

        :param images: 图像张量(batch_size, 3, image_size, image_size)
        :return: 编码后的图像
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)  (batch_size, 2048, 8,8)
        # !!!!!!!!!
        batch_size = out.shape[0]
        global_features = self.avgpool(out).view(batch_size, -1)
        # !!!!!!!!!
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)  14
        
        contrast_features = self.adaptive_pool_contrast_feature(out)
        
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        
        contrast_features = self.adjust(torch.squeeze(contrast_features))
        
        return out, global_features, contrast_features

    def fine_tune(self, fine_tune=True):
        """
        允许或防止编码器的卷积块2到4的梯度计算。.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = fine_tune
        # 如果微调，仅微调卷积块2到4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    注意网络
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: 图像编码的特征大小
        :param decoder_dim: RNN的解码大小
        :param attention_dim: 注意力网络的维度
        """
        super(Attention, self).__init__()
        #self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # 对编码图像进行变换的线性层
        self.encoder_att = nn.Linear(decoder_dim, attention_dim)  # 对编码图像进行变换的线性层
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # 转换解码器输出的线性层
        self.full_att = nn.Linear(attention_dim, 1)  # 用于计算被softmax的值的线性层
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: 图像编码 (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: 上一个解码器输出, (batch_size, decoder_dim)
        :return: 注意权重编码, 权重
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)  num_pixes这里是1
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class MLC(nn.Module):
    def __init__(self,
                 classes=11,
                 sementic_features_dim=1024,
                 fc_in_features=2048,
                 k=10):
        # K = 1 ?,修改成10
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = nn.Softmax()
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
#         print('avg_features.shape',avg_features.shape)
        tags = self.softmax(self.classifier(avg_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
#         print('tags.shape',tags.shape)
#         print('semantic_features.shape',semantic_features.shape)
        # return tags, semantic_features
        return semantic_features


# !!!!!  RM Modual  !!!!!!!!!!!!!!
class RelationMemory(nn.Module):
    def __init__(self, D):
        super(RelationMemory, self).__init__()
        self.W_Q = nn.Linear(D, D)
#         self.W_K = nn.Linear(D*2, D)
#         self.W_V = nn.Linear(D*2, D)
        self.W_K = nn.Linear(D, D)
        self.W_V = nn.Linear(D, D)
        self.softmax = nn.Softmax(dim=2)  # (b,k,d)最后一个维度 特征的维度上进行softmax
        self.layernorm = nn.LayerNorm(D, eps=1e-05, elementwise_affine=True)
        self.AOAttention = AOAttention(D)
        # !!!!!!!!!!!!!!!!!!!!
        self.mlp = nn.Sequential(nn.Linear(1024, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU())

        self.W = nn.Linear(1024, 1024 * 2)
        self.U = nn.Linear(1024, 1024 * 2)

    # 多头 8个
    def slice(self, Q, K, V):  # (b,K,1024)
        # !!!!!!!!!!!!!
#         print("Q:",Q.size())
#         print("k:",K.size())
#         print("V:",V.size())
#         print("Q:",Q.size(1))
        d = Q.size(2) // 8
        # !!!!!!!!!!!!!!!!
#         d = Q.size(2) // 8
        list_Q = []
        list_k = []
        list_v = []

        for i in range(8):
            tempQ = Q[:, :, d * i:d * (i + 1)]
            tempK = K[:, :, d * i:d * (i + 1)]
            tempV = V[:, :, d * i:d * (i + 1)]
            list_Q.append(tempQ)
            list_k.append(tempK)
            list_v.append(tempV)

        return list_Q, list_k, list_v

    def forward(self, B, A):
        # B是memory  A是input_rm
        list_Q, list_k, list_v = self.slice(self.W_Q(B), self.W_K(A), self.W_V(A))
        head = []
        for i in range(len(list_v)):
#             print("list_Q[i]", list_Q[i].size())
#             print("list_k[i]", list_k[i].size())
#             print("list_v[i]", list_v[i].size())
            f_dot_att = self.softmax(list_Q[i].matmul(list_k[i].permute(0, 2, 1)) / 8)
            f_dot_att = f_dot_att.matmul(list_v[i])
            head.append(f_dot_att)
        f_mh_att = torch.cat((head[0], head[1], head[2], head[3], head[4], head[5], head[6], head[7]),
                             dim=2)  # 多头  （ b，k,D）
#         print("f_mh_att:", f_mh_att.size())  [32,1,1024]
        # AOAttention = self.AOAttention(f_mh_att, self.W_Q(B))
        # 多头注意力添加了AOA模块
        AOA = self.AOAttention(self.W_Q(B), f_mh_att, f_mh_att)
#         print("AOA:",AOA.size())  [32,1,1024]
        Z = self.layernorm(AOA + A)  # [32,1,1024]
        next_memory = B + Z
#         print("Z:", Z.size())
#         print("next_memory:", next_memory.size())  #[32,1,1024]

        next_memory = next_memory + self.mlp(next_memory)
#         gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(B))
        gates = self.W(B) + self.U(torch.tanh(A))
        gates = torch.split(gates, split_size_or_sections=1024, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * B
#         print("next_memory:", next_memory.size())  #[32,1,1024]
#         next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)
        #修改维度
        next_memory = next_memory.squeeze(1)
#         print("next_memory:", next_memory.size()) [32, 1024]

        return next_memory
# !!!!!!!!!!!!!!!!!!!!!


class DecoderWithAttention(nn.Module):
    """
    解码器
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, word2idx, encoder_dim=2048, dropout=0.5,
                 max_seq_len=52):
        """
        embed_dim  decoder_dim  设置为1024
        :param attention_dim: 注意网络大小
        :param embed_dim: 嵌入大小
        :param decoder_dim: RNN解码器大小
        :param vocab_size: 词表大小
        :param encoder_dim: 编码图像的特征尺寸
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()
        
        self.AOA=AOAttention(1024)
        self.multi_head=multi_head_att(1024)
        self.X_Linear=X_Linear_block()
        self.feat_embed=nn.Linear(encoder_dim,1024)
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout_num = dropout
        self.ctx_dim = 1024
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.W_G=nn.Linear(1024*2,1024)

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 嵌入层
        self.dropout = nn.Dropout(p=self.dropout_num)
        self.decode_step = MogrifierLSTMCell(embed_dim + decoder_dim, decoder_dim, 4)  # LSTMCell
        #self.decode_step = nn.LSTMCell(embed_dim + decoder_dim, decoder_dim)  # LSTMCell
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        #self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # 线性层来创建一个sigmoid激活门
        self.f_beta = nn.Linear(decoder_dim, decoder_dim)  # 线性层来创建一个sigmoid激活门
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
#         self.fc1 = nn.Linear(decoder_dim*2, decoder_dim)      # 两个上下文本向量进行concat后进行全连接层进行降维
        # self.fc1 = nn.Linear(decoder_dim * 3, decoder_dim)  # 三个上下文本向量进行concat后进行全连接层进行降维
        self.fc1 = nn.Linear(decoder_dim * 4, decoder_dim)  # 四个上下文本向量进行concat后进行全连接层进行降维
  
        self.init_weights()
        # !!!!!!!!!!!!!
        self.rm = RelationMemory(1024)
        self.input_embed = nn.Linear(encoder_dim, 1024)
        
        # !!!!!!!!添加MLC
        self.mlc = MLC()
        # !!!!!!!!

    def init_weights(self):
        """
        均匀分布初始化
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        加载预训练好的词嵌入

        :param embeddings: 预训练好的词嵌入
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        是否微调嵌入层（只有在不使用预训练的嵌入层时使用）。

        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        初始化h,c
        像素数是不是14*14呢，后面编码是2048过一个线性层的
        :param encoder_out: 图像编码,  (batch_size, num_pixels, encoder_dim)
        :return: h,c
        """
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch,encoder_dim)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def _forward_step(self, it, t, encoder_out, h, c, embeddings, alphas):

        attention_weighted_encoding, alpha = self.attention(encoder_out, h)
        gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
        attention_weighted_encoding = gate * attention_weighted_encoding

        h, c = self.decode_step(
            torch.cat([embeddings(it), attention_weighted_encoding], dim=1), (h, c))  # (batch_size, decoder_dim)
        preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
        alphas[:, t, :] = alpha
        logprobs = F.log_softmax(preds, dim=1)

        return logprobs, alphas, h, c

    #     def _forward_step(self, it,t,encoder_out,h,embeddings,alphas):

    #         attention_weighted_encoding, alpha = self.attention(encoder_out,h)
    #         gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
    #         attention_weighted_encoding = gate * attention_weighted_encoding

    #         h= self.decode_step(
    #             torch.cat([embeddings(it), attention_weighted_encoding], dim=1),h)  # (batch_size, decoder_dim)
    #         preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
    #         alphas[:, t, :] = alpha
    #         logprobs = F.log_softmax(preds, dim=1)

    #         return logprobs ,alphas ,h

    '''
    图中的图像EMbed是经过Linear,dropout,加激活函数
    K0=V
    V0=V
    Q0=torch.mean(V,1)


    K1=K_···
    V1=V_
    Q1=Q_

    一共M+1 个Q 组成图像级特征， 最后一个时刻的V_是增强的区域级特征  att_feats

    M+1个图像级特征concat起来经过Linear  就是全局图像特征 GV_FEATS

    增强的区域特征 att_feats 在每个时间步 作为 LSTM后的X_Linear   的K,V，Q是LSTM的h       
    全局图像特征经过Linear后与上下文语义和词嵌入concat起来

    '''
    def forward(self, all_feats, global_features, encoded_captions, caption_lengths):  # 第二个位置是目标检测特征  第三个位置是卷积特征
        """
        Forward propagation.

        :param encoded_captions: 字幕编码,  (batch_size, max_caption_length)
        :param caption_lengths: 字幕长度,  (batch_size, 1)
        :return: scores for vocabulary, 已排序的字幕编码, 解码长度, weights, sort indices
        """


        if 1:
            batch_size = all_feats.size(0)
            encoder_dim = all_feats.size(-1)
            vocab_size = self.vocab_size
            

            # 展平图像
            all_feats = all_feats.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim) 1？
            CNN_feats= self.dropout(self.feat_embed(all_feats))  # (batch,49,1024)

     
            num_pixels = all_feats.size(1)  #
      
      		# !!!!!!!
            semantic_feats = F.relu(self.mlc(global_features))
        	# !!!!!!!

            # 按长度递减对输入数据进行排序; 图像，图像对应字幕，图像对应字幕长度
            caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)  # （batch,）

            encoded_captions = encoded_captions[sort_ind]
            CNN_feats = CNN_feats[sort_ind]


            all_feats=all_feats[sort_ind]
            
            # !!!!!!!
            semantic_feats = semantic_feats[sort_ind]
            # !!!!!!!


            # Embedding
            embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
            # 长度不足max_caption_length的，后面填充0

            # 初始化LSTM 状态  ，0初始化还是下面这种都可以，反正只是学习参数
            
            h, c = self.init_hidden_state(all_feats)  # (batch_size, decoder_dim)
            h1, c1 = self.init_hidden_state(all_feats)   # 增加新的一层LSTM初始化

            # 我们不会在<end>位置解码，因为只要生成<end>我们就停止生成
            # 所以，解码长度是实际长度-1
            decode_lengths = (caption_lengths - 1).tolist()

            # 创建张量来保存单词预测分数和alphas
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)  # 与真实句子一样长，求交叉熵训练吗？
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)  # 每个字都对应一个alpha

            a_mean=torch.mean(CNN_feats,dim=1)
            ctx_=torch.zeros_like(a_mean)
            
            memory = torch.zeros_like(a_mean)
            
            # 在每个时间步，根据解码器先前的隐藏状态输出，通过注意加权编码器的输出进行解码，然后在解码器中使用前一个单词和注意力加权编码生成一个新词
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])  # 这一个batch的句子，一旦有一个句子生成结束，batch就减去1，直到全部生成

                Input = torch.cat((a_mean[:batch_size_t]+ctx_[:batch_size_t], embeddings[:batch_size_t, t, :]), dim=1)
                
                # !!!!!!!!RM!!!!!!!!! 
                input_rm = torch.cat((memory[:batch_size_t], embeddings[:batch_size_t, t, :]), dim=1)
                input_rm = self.input_embed(input_rm)   # [32,1024]
                memory = self.rm(memory[:batch_size_t].unsqueeze(1), input_rm.unsqueeze(1))
                # !!!!!!!!!!!!!!!!!!!
      
                attention_weighted_encoding, alpha = self.attention(CNN_feats[:batch_size_t], h[:batch_size_t])
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding  # (batch_size_t,2048)
                # print(ctx.shape,attention_weighted_encoding.shape)
                
                h, c = self.decode_step(Input,(h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                a_mao=self.multi_head(h.unsqueeze(1),CNN_feats[:batch_size_t])
                ctx_=self.AOA(h.unsqueeze(1),a_mao,a_mao).squeeze(1)
                
                # !!!!!!!! semantic features !!!!!!!
                a_mao_semantic = self.multi_head(h.unsqueeze(1), semantic_feats[:batch_size_t])
                ctx_semantic = self.AOA(h.unsqueeze(1), a_mao_semantic, a_mao_semantic).squeeze(1)
                # !!!!!!!!
                
                # add LSTM2 增加一个LSTM结果，生成新的上下文本向量
                # h = h.unsqueeze.expand(h.size(0), 49, h.size(1))    # [32, 49, 1024]  错误
                a_mao = (a_mao * alpha.unsqueeze(2)).sum(dim=1)              # [32, 1024]
                # print('a_mao2:', a_mao.size())
                input2 = torch.cat((a_mao, h), dim=1)    # [32, 2048]
                h1, c1 = self.decode_step(input2, (h1[:batch_size_t], c1[:batch_size_t]))
                ctx1 = h1
                
                # AOA生成的上下文本向量和LSTM2生成的上下文本向量进行concat
                ctx_ = torch.cat([ctx1, ctx_], dim=1)
                # print("ctx_2:", ctx_.size.size())   # [ 32, 2048]
                
                # !!!!!!!RM  Memory!!!!!!!!!
                ctx_ = torch.cat([ctx_, memory], dim=1)
                # !!!!!!!!!!!!!!!!
                
                # !!!!!!!!!!! semantic feature
                ctx_ = torch.cat([ctx_, ctx_semantic], dim=1)
                # !!!!!!!!!!!
                
                ctx_ = self.fc1(ctx_)    # [ 32, 1024]
                

                preds = self.fc(self.dropout(ctx_))  # (batch_size_t, vocab_size)

                predictions[:batch_size_t, t, :] = preds
                alphas[:batch_size_t, t, :] = alpha

            return predictions, encoded_captions, decode_lengths, alphas, sort_ind

