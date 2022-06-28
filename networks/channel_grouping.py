import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def load_cls_model(class_num, pretrained=False):
    #  return a resnet-based model for classification
    
    cls_model = models.resnet101(pretrained = pretrained)
    cls_model.fc = nn.Linear(in_features=2048, out_features=class_num, bias=True)
    
    return cls_model


class Identity(nn.Module):
    # replace some layers to ignore them
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    

    # ToDO: add map_location parameter in torch.load
def load_backbone_model(model_path):
    # ignore the average pooling and dense layer to get the feature map directly
    # the output 2048*14*14 is flattened, and need to be reshaped before convolution
    
    model = models.resnet101()
    model.avgpool = nn.Sequential()
    model.fc = nn.Sequential()
    
    model_dict = model.state_dict()
    
    pretrained_model_path = model_path
    pretrained_dict = torch.load(pretrained_model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
        
    return model



class channel_grouping_layer(nn.Module):
    
    '''
    adopt fc layers for clusterring:
    input: 2048 * 14 * 14
    output: [cluster_result, attention_mask]

    cluster_result: part_nums * channel_num
    weighted_feature: part_nums * 14 * 14
    '''

    def __init__(self, part_num, channel_num):
        super(channel_grouping_layer, self).__init__()
        self.part_num = part_num
        self.channel_num = channel_num
        
        hidden_states = int(self.channel_num * self.part_num/2)

        self.fc1 = nn.Linear(self.channel_num, hidden_states)
        self.fc2 = nn.Linear(hidden_states, self.channel_num * self.part_num)
        
#         self.fc1 = nn.Linear(self.channel_num,self.part_num)
#         self.fc2 = nn.Linear(self.part_num,self.channel_num)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool3d((196,1,1))

    def forward(self,x):
        # the input feature map is the output of layer4 in resnet101 with the dimension of 2048*14*14
        conv_matrix = torch.clone(x)
        conv_matrix = conv_matrix.reshape(conv_matrix.size(0), self.channel_num, 1, 196)

        '''
        get the weights for each channel
        input: feature maps of 2048 * 14 * 14, part_num n
        output: channel weights of part_num * 2048

        '''
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        grouping_result = x.unfold(-1, self.channel_num, self.channel_num)
        #  calculate the loss here to supervise pre-training

        '''
        get the weighted featuremap of the regions
        '''
        conv_matrix = conv_matrix.unsqueeze(1)
        x = grouping_result.unsqueeze(-1).unsqueeze(-1)
        x = x * conv_matrix
        x = x.transpose(2,4)
        x = self.avgpool2(x)    #avgpool over the channels
#         x = x * 0.1  
#         x = F.softmax(x, dim=2)
#         x = torch.exp(x)
#         x = torch.log(1 + x)
#         x = x * 4
#         x = x.squeeze(-1).squeeze(-1)
        x = x.reshape(x.size(0), self.part_num, 14, 14)


        return grouping_result, x 