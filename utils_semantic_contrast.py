import os
import numpy as np
import h5py
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import random
from self_critical.cider.pyciderevalcap.ciderD.ciderD import CiderD
from self_critical.bleu.bleu import Bleu
import torch.nn as nn
import cv2
import torch.nn.functional as F
import argparse

#创建图像数据文件夹、保存，建立词映射、保存，以及一些工具类

def create_input_files(output_folder='/zengxh_phd/wzq/image_captioning_second_point_Xray/use_openi',captions_per_image=1,
                       max_len=50):
    """
    创建输入文件 for training, validation, and test data.

    :param captions_per_image: 每个图像对应多少个句子
    :param output_folder: 保存文件的文件夹
    :param max_len: caption的最大长度
    """





    #将对应数据放到对应文件

    tem_cap = []
    for i, j in enumerate(['/zengxh_phd/wzq/image_captioning_second_point_Xray/use_openi/train.txt',
                           '/zengxh_phd/wzq/image_captioning_second_point_Xray/use_openi/val.txt']):
        data_name = []
        cap1 = []
        with open(j, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                cap1.append(line[3].split(' '))
        tem_cap.append(cap1)

    train_image_captions = tem_cap[0]
    val_image_captions = tem_cap[1]




    # 建立词映射
    # 分好词后的文本数据
    targetTxt = '/zengxh_phd/wzq/image_captioning_second_point_Xray/use_openi/all.txt'
    word = []
    with open(targetTxt, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')  #
            a=line[3].strip().split(' ')
            word.extend(a)
    word_dict = Counter(word)

    words = [w for w in word_dict.keys() if word_dict[w] > 0]

    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 2
    word_map['<start>'] = len(word_map)
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0


    # 保存词和数字的映射到 JSON   word_map
    with open(os.path.join(output_folder, 'WORDMAP.json'), 'w',encoding='utf-8') as j:
        json.dump(word_map, j,ensure_ascii=False)


    # 采样每个图像的字幕，将图像保存，并将字幕及其长度保存到JSON文件
    seed(123)
    for  imcaps, split in [( train_image_captions, 'TRAIN'),
                                   (val_image_captions, 'VAL'),
                        ]:
        enc_captions = []
        enc=[]
        caplens = []
        for j, c in enumerate(imcaps):
            # 编码字幕
            enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
            enc_cap = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                word_map['<end>']] 
            # 字母长度加首尾标志
            c_len = len(c) + 2
            caplens.append(c_len)
            enc_captions.append(enc_c)
            enc.append(enc_cap)


        # 保存编码字幕和长度到 JSON files
        with open(os.path.join(output_folder, split + '_CAPTIONS_'  + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(output_folder, split + '_CAPLENS_' + '.json'), 'w') as j:
            json.dump(caplens, j)

            
        with open(os.path.join(output_folder, split + '_CAPTIONS_no_pad'  + '.json'), 'w') as j:
            json.dump(enc, j)

#embeddings (len(vocab), emb_dim)
def init_embedding(embeddings):
    """
    均匀分布填充嵌入张量
    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)



def load_embeddings(emb_file, word_map): #  emb_file是啥格式？ 里面是什么字符串
    # emb_file 是txt文件   猜测里面的内容是每一行是一个word 后面是他的嵌入表示比如  我 1 5 9 0 ... 5 3 5   这种，且用空格隔开
    # 有个疑问，这个一一对应的嵌入表示是怎么来的？
    #难道是测试或者验证的时候使用吗？ 但是没看到训练的时候保存到文件了啊
    """
    为指定的词映射创建嵌入张量，以便加载到模型中。

    :param emb_file:包含嵌入的文件（以glow格式存储）
    :param word_map: 词映射
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    返回 ：嵌入词在单词映射中的顺序，嵌入的维数
    """

    # 寻找嵌入维度
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())  #单词集合

    #创建张量来保存词嵌入，初始化
    embeddings = torch.FloatTensor(len(vocab), emb_dim)  #不是可以随便设置？ emb_dim是否有必要
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):  #line是文件每一行
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
        #filter(lambda n: n and not n.isspace(), line[1:])  筛选 出 line[1:]这列表存在且不全为空格组成的字符串
        #然后将这些字符串化成浮点数  装入list


        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


#梯度截断
def clip_gradient(optimizer, grad_clip):
    """
   截断在反向传播期间计算的梯度，以避免梯度爆炸。

    :param optimizer: 具有要截断的梯度的优化器
    :param grad_clip: 截断阈值
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement,  mlc,mlc_optimizer,encoder,decoder, encoder_optimizer,decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset  已处理数据集的基名称
    :param epoch: epoch number
    :param epochs_since_improvement: BLEU-4评分自上次提升以来的epoch数
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: 优化程序更新编码器的权重，如果微调的话
    :param decoder_optimizer: 优化程序更新解码器的权重
    :param bleu4: 此epoch的验证BLEU-4分数
    :param is_best: checkpoint 到目前为止是否最好？
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'mlc':mlc,
             'mlc_optimizer':mlc_optimizer,
             'encoder':encoder,
             'decoder': decoder,
             'encoder_optimizer':encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = data_name +'_checkpoint' + '.pth.tar'
    torch.save(state, filename)
    # 如果这个检查点是目前为止最好的检查点，请存储一个副本，这样它就不会被更糟糕的检查点覆盖
    if is_best:
        torch.save(state, data_name+'BEST_' +'_checkpoint' + '.pth.tar' )


#管理一些变量的更新  在初始化的时候就调用的重置方法reset。当调用该类对象的update方法的时候就会进行变
# 量更新，当要读取某个变量的时候，可以通过对象.属性的方式来读取，比如在train函数中的top1.val读取top1准确率。
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    跟踪度量的最新、平均值、总和和计数。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    按指定的因子收缩学习率。

    :param optimizer: 学习率必须缩小的优化器.
    :param shrink_factor: 将学习率乘以区间（0，1）的因子。
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor


#topk精度  batch_size个样本中  有多少个是正确的   只要在前k个精度  出现了类别l，就算是正确
def accuracy(scores, targets, k):
    """
    根据预测的和真实的标签计算top-k精度。

    :param scores: 模型分数
    :param targets: 真实标签   （batch,1）
    :param k: top-k的k
    :return: top-k 精度
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 只有一个数字的tensor
    return correct_total.item() * (100.0 / batch_size)


# 解决曝光误差 
def scheduled_sampling(p,emd,pre):  #输入 (batch，emd)    (batch,emd)
    # 以概率p输入生成的词
    EMD=torch.zeros_like(emd)
    for batch in range(emd.size(0)): 
        np.random.seed(batch)
        P=random.random()  #生成0到1 的浮点数
        if P>p:
            EMD[batch]=emd[batch]
        else :
            EMD[batch]=pre[batch]
    
    return EMD
            
    
    


def _array_to_str(arr, sos_token, eos_token):
    arr = list(arr)
    if arr[0] == sos_token:
        arr = arr[1:]
    out = ''
    for i in range(len(arr)):
        if arr[i] == eos_token:
            break
        out += str(arr[i]) + ' '
    out += str(eos_token)
    return out.strip()



def flip_image(img):
    '''
        The method of flipping the picture horizontally.

        flipCode
        0   Vertical flipping of the image
        >0  Horizontal flipping of the image
        <0  Simultaneous horizontal and vertical flipping of the image
    '''
    return cv2.flip(img,flipCode = 1)

def gaussian_blur(img):
    '''
        The method of adding random Gaussian noise to the image.
    '''
    kernel_size = (5, 5)
    sigma = 1.5
    return cv2.GaussianBlur(img, kernel_size, sigma)

def rotate_img(img):
    '''
        The method of rotating the picture 90 degrees clockwise.

        rotateCode
        cv2.ROTATE_90_CLOCKWISE  :  Rotate by 90 degrees clockwise
        cv2.ROTATE_180           :  Rotate by 180 degrees clockwise
        cv2.ROTATE_90_COUNTERCLOCKWISE : Rotate by 270 degrees clockwise
    '''
    return cv2.rotate(img,rotateCode=cv2.ROTATE_90_CLOCKWISE)

def contrast_trans(rgb_img, threshold=0.5):
    '''
        The method of changing the color contrast of the image.
    '''
    img = rgb_img * 1.0
    img_out = img

    img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - threshold * 255.0)
    img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - threshold * 255.0)
    img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - threshold * 255.0)
    img_out = img_out / 255.0
    return img_out

def center_crop(img):
    '''
        The method of cropping the center of the image.
    '''

    width,length= img.shape[:2]
    window_size = 80
    width = int(width/2)
    length = int(length/2)
    min_width = width - window_size
    max_width = width + window_size
    min_length = length - window_size
    max_length = length + window_size
    crop = img[min_width:max_width,min_length:max_length]
    img_out = cv2.resize(crop,(224,224))
    return img_out


def calc_euli_distance(repre1, repre2):  # (batch,feature_size)
    repre1 = F.normalize(repre1)
    repre2 = F.normalize(repre2)
    sim_matrix = torch.matmul(repre1,repre2.T) + torch.ones([repre1.shape[0],repre2.shape[0]]).cuda()
    #sim_matrix = torch.matmul(repre1, repre2.T) + torch.ones([repre1.shape[0], repre2.shape[0]])
    return sim_matrix

def contrast_loss(repre):  # (batch,dim)
    b=repre.shape[0]//2
    label = torch.arange(b)
    labels = torch.cat([label for i in range(2)], dim=0)  # 复制2份concat   (2*batch,dim)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # 正匹配为1 负匹配为0
    print(labels.shape)
    # 相似性矩阵
    similarity_matrix = calc_euli_distance(repre, repre)
    print(similarity_matrix.shape)
    mask = torch.eye(labels.shape[0], dtype=torch.bool)  # [64,64]  主对角线全为 True

    # 去除自身的标签配对矩阵
    labels = labels[~mask].view(labels.shape[0], -1)  # [64，63]  ~mask 好像是对mask进行取反  true变false

    # 去除自身的相似矩阵
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].sum()
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].sum()
    closs =negatives/(positives+1e-6)

    return closs


def info_nce_loss(features):
    bt=features.shape[0]//2
    labels = torch.cat([torch.arange(bt) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    features = F.normalize(features, dim=1)
#     print("features:", features.shape)  #[64, 512]
    # similarity_matrix = torch.matmul(features, features.T)
    similarity_matrix = torch.matmul(features, features.t())
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.uint8)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape
    # select and combine multiple positives
    positives = similarity_matrix[labels.byte()].view(labels.shape[0], -1)
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.byte()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    logits = logits / 0.07
    return logits, labels



def get_ciderd_scorer(text1,text2, word_map):
    print('====> get_ciderd_scorer begin')
    refs_idxs = []
    for i in text1:
        refs_idxs.append(_array_to_str(i, word_map['<start>'], word_map['<end>']))
    for i in text2:
        refs_idxs.append(_array_to_str(i, word_map['<start>'], word_map['<end>']))

    scorer = CiderD(refs=refs_idxs)
    print('====> get_ciderd_scorer end')
    return scorer


def get_self_critical_reward(sample_captions, greedy_captions, fns, ground_truth,
                             sos_token, eos_token, scorer):
    batch_size = len(fns)
    sample_captions = sample_captions.cpu().numpy()
    greedy_captions = greedy_captions.cpu().numpy()
    assert sample_captions.shape[0] == greedy_captions.shape[0] == batch_size
    max_seq_len = sample_captions.shape[1] + 1
    sample_result = []
    greedy_result = []
    gts = {}
    for i, fn in enumerate(fns):
        sample_result.append({'image_id': fn, 'caption': [_array_to_str(sample_captions[i], sos_token, eos_token)]})
        greedy_result.append({'image_id': fn, 'caption': [_array_to_str(greedy_captions[i], sos_token, eos_token)]})
        caps = []
        for cap in ground_truth[fn]:
            caps.append(_array_to_str(cap[:max_seq_len], sos_token, eos_token))
        gts[fn] = caps
    all_result = sample_result + greedy_result
    if isinstance(scorer, CiderD):
        _, scores = scorer.compute_score(gts, all_result)
    elif isinstance(scorer, Bleu):
        _, scores = scorer.compute_score(gts, all_result)
        scores = np.array(scores[3])
    else:
        raise Exception('do not support this scorer: %s' % type(scorer))

    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], sample_captions.shape[1], 1)
    return rewards


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq_logprobs, seq_masks, reward):
        output = - seq_logprobs * seq_masks * reward
        output = torch.sum(output) / torch.sum(seq_masks)

        return output

    
