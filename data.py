#---------------------------需要用到的库-----------------------------------------
import json
from collections import Counter
import random
import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import string
from random import randrange
from PIL import Image

# -------------------------- 全局特殊标记（关键：统一作用域）--------------------------
__PAD_TOKEN = 'PAD'  # 填充标记
__UNK_TOKEN = 'UNK'  # 未知词标记
__EOS_TOKEN = 'EOS'  # 句尾标记
__normalize = {'mean': [0.485, 0.456, 0.406],
               'std': [0.229, 0.224, 0.225]}    #来自ImageNet数据集的统计结果

__FLICKR8K_IMG_PATH = "flickr8k/images/split_images/"
__FLICKR8K_ANN_PATH = "flickr8k/split_annotations/"
__TRAIN_PATH = {'root': os.path.join(__FLICKR8K_IMG_PATH, 'train'),
                'annFile': os.path.join(__FLICKR8K_ANN_PATH, 'train_annotations.json')
                }
__VAL_PATH = {'root': os.path.join(__FLICKR8K_IMG_PATH, 'val'),
              'annFile': os.path.join(__FLICKR8K_ANN_PATH, 'val_annotations.json')
              }

#--------------------------构建flickr8k的类------------------------
class Flickr8k:
    def __init__(self,ann_filepath='flickr8k/dataset_flickr8k.json'):
        with open(ann_filepath) as f:
            self.data = json.load(f)
            self.images=self.data['images']

    #------------------------得到所有token组成的列表-------------------
    def get_tokens_by_split(self, split='train', use_tokens=True):
        tokens=[]
        for img_info in self.images:
            if img_info['split'] != split:
                continue
            for sentence in img_info['sentences']:
                if use_tokens:
                    tokens.append(sentence['tokens'])
                else:
                    tokens.append(sentence['raw'])
        all_tokens = [word for sentence in tokens for word in sentence]
        return all_tokens   #得到了所有的token组成一个列表，可能有重复的

    #-----------------------得到每个图片对应的token列表-----------------------
    def img_to_caption(self, split='train'):
        img_captions={}
        for img_info in self.images:
            if img_info['split'] != split:
                continue
            img_id=img_info['imgid']
            captions = [sentence['tokens'] for sentence in img_info['sentences']]
            img_captions[img_id] = captions
        return img_captions

#-------------------------构建Flickr8k数据集类------------------------
'''
主要用来处理训练集/验证集的数据，输出为（图像张量，标题张量）
'''
class Flickr8kDataset(Dataset):
    def __init__(self,root,annFile,transform=None,target_transform=None):
        super().__init__()
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.target_transform = target_transform
        self.images_info=json.load(open(annFile))['images']
        self.filename_to_tokens=self._build_filename_tokens_map()

    def _build_filename_tokens_map(self):
        filename_map = {}
        for img_info in self.images_info:
            filename = img_info['filename']
            captions_tokens = [sent['tokens'] for sent in img_info['sentences']]
            filename_map[filename] = captions_tokens
        return filename_map

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx): #这里的idx由__len__产生
        img_info=self.images_info[idx]
        filename=img_info['filename']

        img_path=os.path.join(self.root,filename)
        image=Image.open(img_path).convert('RGB')

        captions_tokens=self.filename_to_tokens[filename]

        if self.transform is not None:
            image=self.transform(image)

        if self.target_transform is not None:
            target=self.target_transform(captions_tokens)
        else:
            target=captions_tokens

        return image, target

#--------------------------构建词汇表-------------------------------
def build_vocab(num_words=10000, split='train'):
    flickr8k=Flickr8k()
    all_tokens=flickr8k.get_tokens_by_split(split=split, use_tokens=True)
    raw_vocab=Counter(all_tokens).most_common(num_words)
    final_vocab=[word for word, count in raw_vocab]
    #加上全局标记
    vocab=final_vocab = [__PAD_TOKEN] + final_vocab+ [__UNK_TOKEN, __EOS_TOKEN]
    return vocab  #完全变成了列表的形式


#------------------------将文字转换为向量-----------------------------
'''
这部分的代码主要是构造了一个word2idx，将分词对应数字
同时，随机选择某个对应标题，并将其转换为张量形式
需要注意的是，转换的时候需要对应添加unk_idx和eos_idx
'''
def turn_to_tensor(vocab,rnd_captions=True):
    word2idx={ word:idx for idx , word in enumerate(vocab)}
    unk_idx=word2idx[__UNK_TOKEN]
    eos_idx=word2idx[__EOS_TOKEN]
    def get_caption_tensor(caption): #注意，这里应该是对应单个图片的五个标题[[],[],[],[],[]]的形式
        if rnd_captions:
            selected_idx=random.randrange(len(caption))
        else:
            selected_idx=0
        selected_caption=caption[selected_idx]
        #转换为索引的形式，注意这里是列表到列表（数字）
        target_caption_list=[word2idx.get(word,unk_idx) for word in selected_caption]
        target_caption_list.append(eos_idx)
        return torch.tensor(target_caption_list, dtype=torch.long)
    return get_caption_tensor

#------------------------解决字幕序列长度不一致（批次处理）-------------------------
'''
这一步主要是对上一步的函数进行处理，因为上一步的函数是对单个图片进行处理
所以这一步需要统一长度和输入的批次问题
'''
def align_cap(vocab,max_len=50):
    pad_idx=vocab.index(__PAD_TOKEN)
    eos_idx=vocab.index(__EOS_TOKEN)

    def collate(img_cap):#输入是列表形式[(,),(,)],其中这里的元组是图像与标题对，注意输入的都是张量
        img_cap.sort(key=lambda x : len(x[1]) , reverse=True)
        imgs, caps=zip(*img_cap)    #这一步是解包元组并返回一系列可迭代的对象
        imgs=torch.cat([img.unsqueeze(0) for img in imgs],dim=0)    #这一步是在拼接向量
        lengths=[min(len(cap)+1,max_len) for cap in caps]
        batch_max_length=max(lengths)   #计算有效标题张量长度
        cap_tensor=torch.LongTensor(batch_max_length,len(caps)).fill_(pad_idx)
        for i,cap in enumerate(caps):
            end_cap=lengths[i]-1
            if end_cap<batch_max_length:
                cap_tensor[end_cap,i]=eos_idx

            cap_tensor[:end_cap,i].copy_(cap[:end_cap])#每一列是一个标题对应的张量
        return (imgs,(cap_tensor,lengths))
    return collate

#-----------------------Flickr8k数据加载函数-------------------------------
def get_flickr_data(vocab,train=True,img_size=224,scale_size=256,normalize=__normalize):
    if train:
        root,annFile=__TRAIN_PATH['root'],__TRAIN_PATH['annFile']
        img_transform=transforms.Compose(
            [
                transforms.Resize(scale_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize['mean'],normalize['std'])
            ]
        )
    else:
        root,annFile=__VAL_PATH['root'],__VAL_PATH['annFile']
        img_transform=transforms.Compose(
            [
                transforms.Resize(scale_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize['mean'], normalize['std'])
            ]
        )
    data=(Flickr8kDataset(
        root=root,
        annFile=annFile,
        transform=img_transform,
        target_transform=turn_to_tensor(vocab,rnd_captions=True)
    ),vocab)
    return data


def get_iterator(data,batch_size=32,max_length=30,shuffle=True,num_workers=4,pin_menmory=True):
    dataset,vocab=data
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_menmory,
        collate_fn=align_cap(vocab,max_len=max_length)
    )

#---ceshi----
'''
vocab=build_vocab(num_words=10000,split='train')
data=get_flickr_data(vocab)
train_iterator=get_iterator(data)
for idx,(images,captions) in enumerate(train_iterator):
    if idx==0:
        print(f'{idx}:{images}:{captions}')
    else:
        break
'''
