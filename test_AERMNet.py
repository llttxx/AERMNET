import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets_semantic_contrast import *
from utils_semantic_contrast import *
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
import torch.nn.functional as F
from tqdm import tqdm

# 参数
data_folder = '/zengxh_fix/starstar/wzq/image_captioning_second_point_Xray/use_openi'  # 包含create_input_File.py保存的数据文件的文件夹 '/zengxh_phd/wzq/image_captioning_second_point_Xray/use_openi'
data_name = '_' + str(1) + 'MAMNet_RM_Xray_wzq' 

checkpoint = '/zengxh_fix/starstar/wzq/image_captioning_second_point_Xray/wzq_checkpoint/AERMNet_semantic_contrast_epoch50_xray_wzq_a_0_0_0_1_BEST__checkpoint.pth.tar'

# 词映射文件
word_map_file = '/zengxh_fix/starstar/wzq/image_captioning_second_point_Xray/use_openi/WORDMAP.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Load model
checkpoint = torch.load(checkpoint)
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
mlc = checkpoint['mlc']
mlc = mlc.to(device)
mlc.eval()


decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r', encoding='utf-8') as j:
    word_map = json.load(j)

# 词映射的逆映射   之前是词-数字   现在数字-词
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: 生成句子用beam search 评估
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset( data_name, 'TEST',data_folder),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # TODO: 批量beam search
    # 因此，不要使用大于1的批处理大小-重要！

    # 为每个图像存储引用（真实字幕）和假设（预测）的列表
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # 对于每个图像
    for i, (all_feats,fn,caps, caplens, allcaps) in enumerate(
        tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        all_feats = all_feats.to(device)
        all_feats, global_features,_ = encoder(all_feats)  # (1,7,7,2048)
        

        b = all_feats.size(0)
        e = all_feats.size(3)
        mean_feats = torch.mean(all_feats.view(b, -1, e), dim=1)
        mean_feats = mean_feats.to(device)
        # pre_tag, semantic_features = mlc(mean_feats)
        semantic_features = mlc(mean_feats)
        semantic_features = semantic_features.to(device)

        k = beam_size

        # Encode
        encoder_dim = all_feats.size(3)
        encoder_out = all_feats.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # !!!!!!! semantic features !!!!!!!
        semantic_feats = F.relu(decoder.mlc(global_features))
        # !!!!!!!!!!!!!!

        # 我们将把这个问题当作批量大小为k的问题
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['Start']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()



        # 展平图像
        all_feats = all_feats.view(1, -1, encoder_dim)  # (K, num_pixels, encoder_dim) 1？
        # CNN_feats= decoder.dropout(decoder.feat_embed(all_feats))
        CNN_feats = decoder.feat_embed(all_feats)  #  (1,49,1024)
        Q = torch.mean(CNN_feats, dim=1)  # (1,1024)

         # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(all_feats)
        h=h.expand(k,h.size(1))
        c=c.expand(k,h.size(1))
        
        # add第二个LSTM初始化
        h1, c1 = decoder.init_hidden_state(all_feats)
        h1=h1.expand(k,h.size(1))
        c1=c1.expand(k,h.size(1))
        
        ctx_ = torch.zeros_like(h)  #(1, 1024)
        ctx_=ctx_.expand(k,  1024)
        
        # add LSTM2产生的ctx1
        ctx1 = torch.zeros_like(h1)
        ctx1 = ctx1.expand(k, 1024)


        #print(semantic_features.shape,ctx_.shape)
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        a_mean=torch.mean(CNN_feats,dim=1)
        a_mean=a_mean.expand(k,  1024)
        
        # !!!!!!!!!  add RM  !!!!!!!!!
        memory = torch.zeros_like(h1)
        memory = memory.expand(k, 1024)
        
        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            Input = torch.cat((a_mean + ctx_, embeddings), dim=1)
            
            # add rm
            input_rm = torch.cat((memory, embeddings), dim=1)
            input_rm = decoder.input_embed(input_rm)
            
            memory = decoder.rm(memory.unsqueeze(1), input_rm.unsqueeze(1))
            # !!!!!!!!!
            
            # add alpha,获得alpha
            attention_weighted_encoding, alpha = decoder.attention(CNN_feats, h)


            h, c = decoder.decode_step(Input, (h, c))  # (s, decoder_dim)
            a_mao = decoder.multi_head(h.unsqueeze(1), CNN_feats)
            ctx_ = decoder.AOA(h.unsqueeze(1), a_mao, a_mao).squeeze(1)
            
            # !!!!!!!!!!!!!!! semantic features !!!!!!!!!!
            a_mao_semantic = decoder.multi_head(h.unsqueeze(1), semantic_feats)
            ctx_semantic = decoder.AOA(h.unsqueeze(1), a_mao_semantic, a_mao_semantic).squeeze(1)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # LSTM2生成的新的上下文本向量
            a_mao = (a_mao * alpha.unsqueeze(2)).sum(dim=1)    # alpha怎么获取？
            input2 = torch.cat((a_mao, h), dim=1)    
            h1, c1 = decoder.decode_step(input2, (h1, c1))
            ctx1 = h1
            
            # AOA生成的上下文本向量和LSTM2生成的上下文本向量进行concat
            ctx_ = torch.cat([ctx1, ctx_], dim=1)
            
            # RM模块产生的
            ctx_ = torch.cat([ctx_, memory], dim=1)
            # !!!!!!!!
            
            # !!!!! semantic !!!!!!!!
            ctx_ = torch.cat([ctx_, ctx_semantic], dim=1)
            # !!!!!!!!!!!!!!!!!!!!!!!
            
            ctx_ = decoder.fc1(ctx_)
              
            scores = decoder.fc(ctx_)
            # scores = decoder.fc(ctx_)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)    # '/'报错，修改
            # prev_word_inds = top_k_words // vocab_size  # (s)  # 去掉了小数点后的数字
            # prev_word_inds = torch.floor_divide(top_k_words, vocab_size)  # (s)    # 
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['End']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            
            # add h1,h2
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            
            # semantic_features=semantic_features[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            ctx_=ctx_[prev_word_inds[incomplete_inds]]
            a_mean=a_mean[prev_word_inds[incomplete_inds]]
            # !!!!!! semantic !!!!!
            semantic_feats = semantic_feats[prev_word_inds[incomplete_inds]]
            # !!!!!  RM !!!!!!
            memory=memory[prev_word_inds[incomplete_inds]]
            
            # Break if things have been going on too long
            if step > 200:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References

        img_caps = caps.tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['Start'], word_map['End'], word_map['Pad']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['Start'], word_map['End'], word_map['Pad']}])

        # 单个句子间的bleu分数
        pre_caps = [w for w in seq if w not in {word_map['Start'], word_map['End'], word_map['Pad']}]
        assert len(references) == len(hypotheses)
        bleu_4 = sentence_bleu(img_captions, pre_caps, weights=(1, 0, 0, 0))
        # print(bleu_4)   # 不打印每一次的结果，直接最后打印

        
        pre_caps = [rev_word_map[w] for w in seq if w not in {word_map['Start'], word_map['End'], word_map['Pad']}]
        pre_caps_num=[w for w in seq if w not in {word_map['Start'], word_map['End'], word_map['Pad']}]
        str_res = ''
        seq_res=''
        for i in range(len(pre_caps)):
            str_res += pre_caps[i] + ' '
            seq_res+= str(pre_caps_num[i])+' '
            
        img_cap = [rev_word_map[w] for w in img_captions[0]]
        img_cap_num=[w for w in img_captions[0]]
        str_gts = ''
        str_gts_num=''
        for i in range(len(img_cap)):
            str_gts += img_cap[i] + ' '
            str_gts_num += str(img_cap_num[i])+ ' '
            
        with open('/zengxh_fix/starstar/wzq/image_captioning_second_point_Xray/wzq_TXT/wzq_pre_AERMNet_sematic_contrast_epoch50_xray_a_0_0_0_1.txt', 'a+', encoding='utf-8') as f:

                f.write(str(fn) + '\t' + str_res.strip(' ') + '\n')

        with open('/zengxh_fix/starstar/wzq/image_captioning_second_point_Xray/wzq_TXT/wzq_truth_AERMNet_semantic_contrast_epoch50_xray_a_0_0_0_1.txt', 'a+', encoding='utf-8') as f:

                f.write(str(fn) + '\t' + str_gts.strip(' ') + '\n')

        
        
        
        # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':
    beam_size = 1
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
