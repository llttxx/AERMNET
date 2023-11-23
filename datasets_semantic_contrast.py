import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np
from PIL import Image
from torchvision import transforms


def create_collate_fn(word2idx):
    def collate_fn(dataset):
        ground_truth = {}
        tmp = []
        for fn, img, caption, caplen in dataset:  # caps是一个图像对应的多个文本 格式是[['a',','],['c'],...]
            
            tensor_caption=caption.unsqueeze(0)
            caption = caption.numpy().tolist()
            caption = [w for w in caption if w != word2idx['<pad>']]
            ground_truth[fn] = [caption[:]]
            for cap in [caption]:
                tmp.append([fn, img.unsqueeze(0),tensor_caption, cap, caplen])
        
        dataset = tmp  # dataset此时是一个list [[fn, cap[0], fc_feat, att_feat],[fn, caps[1], fc_feat, att_feat],[fn, caps[2], fc_feat, att_feat]]

        dataset.sort(key=lambda p: len(p[3]),
                     reverse=True)  # 上面的dataset按dataset第二个元素的长度为索引，从大到小排列，这里是按文本长度从大到小排列   
        fns, imgs, tensor_captions,caps, caplens = zip(*dataset)
        imgs = torch.cat((imgs), dim=0)

        lengths = [min(len(c), 52) for c in caps]
        caps_tensor = torch.cat((tensor_captions),dim=0)  # (batch,52)
        for i, c in enumerate(caps):
            end_cap = lengths[i]
            caps_tensor[i, :end_cap] = torch.LongTensor(c[:end_cap])

        lengths = torch.LongTensor(lengths)
        lengths = lengths.unsqueeze(1)
        return fns, imgs, caps_tensor, lengths, ground_truth  # ground_truth的格式{'filename':[['a',','],['c'],...]}
    return collate_fn


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_name, split, data_folder='/zengxh_fix/starstar/wzq/image_captioning_second_point_Xray/use_openi',
                 transform=None):
        """
        :param data_folder: 存放数据的文件夹
        :param data_name: 已处理数据集的基名称
        :param split: split, one of 'TRAIN', 'VAL'or 'TEST'
        :param transform:图像转换
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        
        self.transform_RandomRotation = transforms.Compose([transforms.ToPILImage(),
                                                            transforms.RandomRotation(360),
                                                            transforms.ToTensor()])
        
        self.transform_RandomHorizontalFlip = transforms.Compose([transforms.ToPILImage(),
                                                                  transforms.RandomHorizontalFlip(0.5),
                                                                  transforms.ToTensor()])
        
        
        self.transform_normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])

        if self.split in {'VAL', 'TEST'}:

            
            with open(os.path.join(data_folder,'test_image_all_feats.npy'),'rb')as f :
                val_all_feat=np.load(f)
            
            self.all_feats=val_all_feat
            # 一个图像对应几个字幕
            self.cpi = 1
            # Load encoded captions (completely into memory)
            with open(os.path.join(data_folder, 'test_caps.json'), 'r') as j:
                self.captions = json.load(j)

            # Load caption lengths (completely into memory)
            with open(os.path.join(data_folder, 'test_caplens.json'), 'r') as j:
                self.caplens = json.load(j)

            file_name_val = []
            with open(os.path.join(data_folder, 'test.txt'), 'r', encoding='utf-8') as j:
                lines = j.readlines()
                for line in lines:
                    line = line.strip().split('\t')
                    file_name_val.append(line[0])
            self.fns = file_name_val
        else:
            
            
            with open(os.path.join(data_folder,'train_image_all_feats.npy'),'rb')as f :
                train_all_feat=np.load(f)       
            self.all_feats=train_all_feat
            self.cpi = 1
            with open(os.path.join(data_folder,  'train_caps.json'), 'r') as j:
                captions = json.load(j)
                self.captions=torch.LongTensor(captions)

                
            # Load caption lengths (completely into memory)
            with open(os.path.join(data_folder, 'train_caplens.json'), 'r') as j:
                caplens = json.load(j)
                self.caplens = torch.LongTensor(caplens)

            file_name_train = []
            with open(os.path.join(data_folder, 'train.txt'), 'r', encoding='utf-8') as j:
                lines = j.readlines()
                for line in lines:
                    line = line.strip().split('\t')
                    file_name_train.append(line[0])
            self.fns = file_name_train
    
        # PyTorch transformation
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    # 图像和文本数据的简单处理
    def __getitem__(self, i):
        # 请记住，第N个标题对应于第个图像（N//captions_per_image）  从第0个开始的
        all_feats =torch.FloatTensor(self.all_feats[i]/ 255.)

#         if self.transform is not None:
#             img = self.transform(img)
        fn = self.fns[i]

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        if self.split is 'TRAIN':
            all_feats_aug1=self.transform_RandomHorizontalFlip(all_feats)
            # print("all_feats_aug1", all_feats_aug1.shape)  # [3, 256, 256]
            all_feats_aug2=self.transform_RandomRotation(all_feats)
            return all_feats_aug1, all_feats_aug2, fn, caption, caplen
        else:

            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])  # list切片索引左闭右开
            return all_feats,fn, caption, caplen, all_captions
    def __len__(self):
        return self.dataset_size


def get_dataloader(data_name, split,workers, batch_size,word2idx,folder):
    dataset = CaptionDataset(data_name, split,data_folder=folder)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True,\
            collate_fn = create_collate_fn(word2idx))
    return dataloader
