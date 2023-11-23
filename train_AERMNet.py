import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from datasets_semantic_contrast import *
from utils_semantic_contrast import *
from nltk.translate.bleu_score import corpus_bleu
from AERMNet_semantic_contrast_Model import *



# 数据参数
data_folder_new = '/zengxh_fix/starstar/wzq/image_captioning_second_point_Xray/wzq_checkpoint' 
data_folder = '/zengxh_fix/starstar/wzq/image_captioning_second_point_Xray/use_openi'  # 包含create_input_File.py保存的数据文件的文件夹
data_name = 'AERMNet_semantic_contrast_epoch50_xray_wzq_a_0_0_0_1_'  # 数据文件共享的基名称


# 模型参数
emb_dim = 1024
attention_dim = 1024
decoder_dim = 1024
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # 只有当模型的输入是固定大小时才设置为true；否则会有大量的计算开销
#如果网络的输入数据在每次 迭代都变化的话，会导致 cuDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。

# 训练参数

start_epoch = 0
epochs =50
epochs_since_improvement = 0  # 从验证BLEU有了改进之后，保持记录epochs的数量
batch_size = 32
workers = 2  # 原本是1
# encoder_lr =1e-3  # 编码器的学习率（如果微调）
# decoder_lr = 1e-3  # 解码器的学习率
encoder_lr =2e-4  # 编码器的学习率（如果微调）
decoder_lr = 2e-4  # 解码器的学习率
mlc_lr=2e-4
grad_clip = 5.  #截断梯度的绝对值
alpha_c = 1.  # “双随机注意”的正则化参数
best_bleu4 = 0.  # BLEU-4 score
print_freq = 50  # 每__批打印训练和验证信息
checkpoint = None  # 检查点的路径，如果没有，则为None
fine_tune_encoder = True 


def main():
    """
    训练和验证
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch,  data_name, word_map
    
    rl_criterion = RewardCriterion()   #词向量的损失。
    
    # 读取词映射
    word_map_file = os.path.join(data_folder, 'WORDMAP.json')  #取得词映射的文件
    with open(word_map_file, 'r',encoding='utf-8') as j:
        word_map = json.load(j)

    # 初始化/加载 checkpoint
    #checkpoint是一个字典
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       word2idx=word_map,
                                       dropout=dropout)
        #筛选出requires_grad为true的参数，并用优化器优化
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        mlc=MLC()
        mlc_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, mlc.parameters()),
                                             lr=mlc_lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        encoder=checkpoint['encoder']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        mlc=checkpoint['mlc']
        encoder_optimizer = checkpoint['encoder_optimizer']
        mlc_optimizer = checkpoint['mlc_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
    # 转到GPU
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    mlc=mlc.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss().to(device)
    mse=nn.MSELoss().to(device)
    
    
    # 自定义 dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset( data_name, 'TRAIN', transform=transforms.Compose([transforms.ToTensor(),normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset( data_name, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
    LOSS_LIST=[]
    # Epochs
    for epoch in range(start_epoch, epochs):
        if epoch<100:
            # 如果连续8个epoch没有改善，则降低学习率，20个epochs后终止训练
#             if epochs_since_improvement == 100:
#                 break
            if epochs_since_improvement > 0 and epochs_since_improvement % 10 == 0:
                adjust_learning_rate(decoder_optimizer, 0.8)


            # 每个epoch的训练
            loss_mean=train(train_loader=train_loader,
                  encoder=encoder,
                  decoder=decoder,
                  criterion=criterion,
                  mse=mse,
                  mlc=mlc,
                  mlc_optimizer=mlc_optimizer,
                  encoder_optimizer=encoder_optimizer,
                  decoder_optimizer=decoder_optimizer,
                  epoch=epoch)

            # 一个批次的验证
            recent_bleu4 = validate(val_loader=val_loader,
                                    encoder=encoder,
                                    decoder=decoder,
                                    criterion=criterion,
                                    mse=mse,
                                    mlc=mlc,
                                    epoch=epoch)

            # 检查是否有提升
            LOSS_LIST.append(loss_mean)
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            save_checkpoint(os.path.join(data_folder_new,data_name), epoch, epochs_since_improvement,mlc,mlc_optimizer, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)
    LOSS_=[w.cpu().item() for w in LOSS_LIST]
    with open('/zengxh_fix/starstar/wzq/image_captioning_second_point_Xray/wzq_json/AERMNet_semantic_contrast_epoch50_loss_wzq_a_0_0_0_1.json', 'w',encoding='utf-8') as j:
        json.dump(LOSS_, j,ensure_ascii=False)    
        
            
def train(train_loader, encoder, decoder, criterion,mse,mlc,mlc_optimizer, encoder_optimizer, decoder_optimizer, epoch):
    """
    一个epoch的训练.

    :param train_loader: 训练数据的DataLoader
    :param decoder: decoder model
    :param criterion: loss函数
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    num_classes=11
    label_smooth=0.1
    decoder.train()  # train mode (如果有 dropout and batchnorm 就用)
    encoder.train()
    mlc.train()


    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    loss_mean=0
    # !!! add all_feats_aug
    for i, (all_feats, all_feats_aug, fn, caps, caplens) in enumerate(train_loader):  #CaptionDataset方法
        data_time.update(time.time() - start)
       # print('all_feats, all_feats_aug, fn, caps, caplens',all_feats, all_feats_aug, fn, caps, caplens)
        print('caplens',caplens.shape)
        print('caplens',caplens)
        all_feats = torch.cat([all_feats, all_feats_aug], dim=0)
        
        fn = fn * 2
        caps = torch.cat([caps, caps], dim=0)
        caplens = torch.cat([caplens, caplens], dim=0)

        all_feats=all_feats.to(device)

        caps = caps.to(device)
        caplens = caplens.to(device)

        # forward
        if encoder is not None:
            # !!!!!!!!!!!!
            all_feats, global_features, contrast_features = encoder(all_feats)  # (batch，7,7，2048)

        #decode_lengths 是每个解码句子的长度，是句子长度+1,有个end   (batch,)
        
        # !!!!!!!!!!!!
        logits, labels = info_nce_loss(contrast_features)
        logits = logits.to(device)
        labels = labels.to(device)

        b=all_feats.size(0)
        e=all_feats.size(3)
        mean=torch.mean(all_feats.view(b,-1,e),dim=1)
        mean=mean.to(device)  #(32,2048)


        #decode_lengths 是每个解码句子的长度，是句子长度+1,有个end   (batch,)
        score, caps_sorted, decode_lengths, alphas, sort_ind = decoder(all_feats, global_features, caps, caplens)

        # 为我们从<start>开始解码，所以目标都是<start>之后的单词，直到<end>
        target = caps_sorted[:, 1:]   #真实字幕

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        score = pack_padded_sequence(score, decode_lengths, batch_first=True)
        target = pack_padded_sequence(target, decode_lengths, batch_first=True)
        scores = score.data
        targets = target.data
        
        # print("scores:", scores.shape)
        # print("targets:", targets.shape)
        

        loss = criterion(scores, targets)

        # 加双随机注意正则化
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        
        # 加对比损失
        # a = 0.01 0.05 0.1 0.5 1 5 10
        loss += 0.001 * criterion(logits, labels)
        
        
        loss_mean+=loss.clone().detach()
        # BP
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        mlc_optimizer.zero_grad()


        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)

        # 只有用了optimizer.step()，模型才会更新
        encoder_optimizer.step()
        mlc_optimizer.step()
        decoder_optimizer.step()


        # 跟踪度量
        top5 = accuracy(scores, targets, 5)
        # 这里为什么要加上sum(decode_lengths)？？？？？？？？？？？？？？
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # 输出状态
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    return loss_mean 

def validate(val_loader, encoder, decoder, criterion,mse,mlc,epoch):
    """
    执行一个epoch的验证

    :param val_loader: DataLoader for validation data.

    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    #如果使用METEOR 会不会更好一些呢
    encoder.eval()


    decoder.eval()  # eval mode (no dropout or batchnorm)


    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # 计算BLEU-4分数的真实字幕
    hypotheses = list()  # 预测字幕

    # 显式禁用梯度计算以避免CUDA内存错误
    with torch.no_grad():
        # Batches

        for i, (all_feats,fn,caps, caplens, allcaps)  in enumerate(val_loader):

            all_feats=all_feats.to(device)

            caps = caps.to(device)
            caplens = caplens.to(device)
            # Forward
            if encoder is not None:
                # 多返回一个参数
                all_feats, global_features, _ = encoder(all_feats)
            

            b=all_feats.size(0)
            e=all_feats.size(3)
            mean=torch.mean(all_feats.view(b,-1,e),dim=1)
            mean=mean.to(device)
        

            
            score, caps_sorted, decode_lengths, alphas, sort_ind = decoder(all_feats, global_features, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            target = caps_sorted[:, 1:]
            scores_copy = score.clone()
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            score = pack_padded_sequence(score, decode_lengths, batch_first=True)
            target = pack_padded_sequence(target, decode_lengths, batch_first=True)
            scores = score.data
            targets = target.data

            # Calculate loss
            loss = criterion(scores, targets)

            # 正则化
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            


            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,loss=losses, top5=top5accs))

            # 存储每个图像的引用（真实标题）和假设（预测）
            # 如果对于n个图像，我们有n个预测，且a,b,c对应每个图像
            # we need references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # # because images were sorted in the decoder
            allcaps = allcaps[sort_ind]  # allcaps (batch,shlf.cpi, max_length)
            for j in range(allcaps.shape[0]):  #batch
                img_caps = allcaps[j].tolist()  #(self.cpi,max_length)
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['Start'], word_map['Pad']}],
                        img_caps))  # 删除start和pad
                references.append(img_captions)
            

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)   #(batch,length)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads 和end
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)
        
        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)
        print(len(references))
        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4 




def train_rl(ciderd_scorer,decoder,train_loader,decoder_optimizer,mode,rl_criterion = RewardCriterion(),training=True):
    decoder.train(training)
    loss_val = 0.0
    reward_val = 0.0
    # 每一个batch
    for fns, imgs, caps_tensor, lengths, ground_truth in tqdm.tqdm(train_loader, ncols=100):
    #ground_truth的格式[['a',','],['c'],...]
        caps_tensor = caps_tensor.to(device) 
        imgs = imgs.to(device)
        lengths =lengths.to(device)
        if training and mode == 'rl':
 
            sample_captions, sample_logprobs, seq_masks = decoder(imgs, caps_tensor, lengths,sample_max=0,mode=mode)
            decoder.eval()
            with torch.no_grad():
                greedy_captions, _, _ = decoder(imgs, caps_tensor, lengths,mode=mode)
            decoder.train(training)
            reward = get_self_critical_reward(
                        sample_captions, greedy_captions, fns, ground_truth,
                        word_map['Start'], word_map['End'], ciderd_scorer)
            loss = rl_criterion(sample_logprobs, seq_masks, torch.from_numpy(reward).float().to(device))
            reward_val += float(np.mean(reward[:, 0]))

                
        loss_val += float(loss)
        if training:
            decoder_optimizer.zero_grad()
            loss.backward()
            clip_gradient(decoder_optimizer, grad_clip)
            decoder_optimizer.step()
    return loss_val / len(train_loader), reward_val / len(train_loader)




if __name__ == '__main__':
    main()
