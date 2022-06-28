import torch
import torch.nn as nn
from torch_geometric.nn import DenseGCNConv
from networks.self_attention import MultiHeadedAttention

class dgcn_cls(nn.Module):
    
    def __init__(self, part_num):
        super(dgcn_cls, self).__init__()
        self.part_num = part_num
        
        self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Conv2d(1, 1, kernel_size=(1,3), stride=1, padding=(0,1))

        self.fc1 = nn.Linear(196, 64)
        self.fc2 = nn.Linear(196, 64)
        self.fc3 = nn.Linear(196, self.part_num)
        self.fc_cls = nn.Linear(self.part_num+1, 2)

        self.conv1 = DenseGCNConv(196, 196)
        self.conv2 = DenseGCNConv(196, 196)
        
        self.mha = MultiHeadedAttention(1, self.part_num)

    def forward(self,feature_map, weighted_feature):
        # the inputs are the feature outputed by layer4 of resnet101 and the attention_mask outputed by the channel_grouping_layer
        # the dimension is 2048 * 14 *14 and part_num * 14 * 14, respectively

        region_features = weighted_feature.reshape(-1, self.part_num, 196)
        
        # get the dynamic correlation matrix of the graph for the image using self-attention
        q = self.fc1(region_features)
        k = self.fc2(region_features)
        v = self.fc3(region_features)
        corr_matrix = self.mha(q,k,v)

        # propogate the features by GCN
        node_feature = torch.relu(self.conv1(region_features, corr_matrix))
        node_feature = nn.Dropout()(node_feature)
        node_feature = torch.relu(self.conv2(node_feature, corr_matrix))
        node_feature = nn.Dropout()(node_feature)
        
        node_feature = node_feature.reshape(-1, self.part_num, 14, 14)
        
        # concatenate global feature and the integrated local features for classification
        
        feature_map = feature_map.reshape(-1, 2048, 1, 196)
        feature_map = feature_map.transpose(1,3)
        compressed_feature = self.global_avgpool(feature_map)
        compressed_feature = compressed_feature.reshape(-1, 1, 14, 14)
        
        final_feature = torch.cat((compressed_feature, node_feature),dim=1)
        
        res = self.global_avgpool(final_feature)
        res = torch.flatten(res, 1)
        res = self.fc_cls(res)
 
        return res