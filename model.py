import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from beam_search import CaptionGenerator
#-------------------------------模型类----------------------------------------
class CaptionModel(nn.Module):

    def __init__(self,cnn,vocab,embedding_size=256,rnn_size=256,num_layers=2,share_embedding_weights=False):
        super(CaptionModel, self).__init__()
        self.vocab=vocab
        self.cnn=cnn
        self.cnn.fc=nn.Linear(self.cnn.fc.in_features,embedding_size) #修改了线性层，保证特征输入lstm的时候能一致
        self.rnn=nn.LSTM(embedding_size,rnn_size,num_layers=num_layers)
        self.classifer=nn.Linear(rnn_size,len(vocab))
        self.embedder=nn.Embedding(len(self.vocab),embedding_size)
        if share_embedding_weights:
            self.embedder.weight=self.classifer.weight

    def forward(self,imgs,captions,lengths):
        embeddings=self.embedder(captions)
        img_feats=self.cnn(imgs).unsqueeze(0)
        embeddings = torch.cat([img_feats, embeddings], dim=0)
        packed_embeddings=pack_padded_sequence(embeddings,lengths)
        feats,state=self.rnn(packed_embeddings)
        pred=self.classifer(feats[0])
        return pred,state
    '''
    pred：所有有效时间步的词预测分数，用于计算训练损失（如交叉熵损失）。
    state：RNN 的最终状态，可用于后续解码（如生成完整句子时的初始状态）
    补充：state：LSTM 为 (h_n, c_n)（最终隐藏状态和细胞状态），形状 (num_layers, batch_size, hidden_dim)；GRU 为 h_n（仅隐藏状态）。
    '''
#-----------------------这部分是做推理任务的函数------------------------------
    def generate(self,img,scale_size=256,crop_size=224,
        eos_token='EOS',beam_size=3,
        max_caption_length=20,
        length_normalization_factor=0.0):

        preproc=[transforms.Resize(scale_size),
                 transforms.CenterCrop(crop_size),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )]

        if torch.is_tensor(img):
            preproc=[transforms.ToPILImage()]+preproc

        img_transform=transforms.Compose(preproc) #打包为整个操作函数
        cap_gen=CaptionGenerator(embedder=self.embedder,rnn=self.rnn,classifier=self.classifer,
                                 eos_id=self.vocab.index(eos_token),
                                 beam_size=beam_size,
                                 max_caption_length=max_caption_length,
                                 length_normalization_factor=length_normalization_factor)
        img=img_transform(img)
        if next(self.parameters()).is_cuda:
            img=img.cuda()
        with torch.no_grad():
            img = img.unsqueeze(0)
        img_feats = self.cnn(img).unsqueeze(0)
        sentences, score = cap_gen.beam_search(img_feats)
        sentences = [' '.join([self.vocab[idx] for idx in sent])
                     for sent in sentences]
        return sentences

    def save_checkpoint(self,filename):
        torch.save({'embedder':self.embedder.state_dict(),
                    'rnn_dict':self.rnn.state_dict(),
                    'cnn_dict':self.cnn.state_dict(),
                    'classifier_dict':self.classifer.state_dict(),
                    'vocab':self.vocab,
                    'model':self},
                   filename)

    def load_checkpoint(self,filename):
        cpnt=torch.load(filename)
        if 'cnn_dict' in cpnt:
            self.cnn.load_state_dict(cpnt['cnn_dict'])
        self.embedder.load_state_dict(cpnt['embedder_dict'])
        self.rnn.load_state_dict(cpnt['rnn_dict'])

        self.classifer.load_state_dict(cpnt['classifier_dict'])

    def finetune_cnn(self,allow=True):
        for p in self.cnn.parameters():
            p.requires_grad=allow
        for p in self.cnn.fc.parameters():
            p.requires_grad=True

