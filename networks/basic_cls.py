import torch
import torch.nn as nn

class cls(nn.Module):
    
    def __init__(self, part_num):
        super(cls, self).__init__()
        self.part_num = part_num
        self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_cls = nn.Linear(self.part_num+1, 2)

        
    def forward(self,feature_map, weighted_feature):
        # the inputs are the feature outputed by layer4 of resnet101 and the attention_mask outputed by the channel_grouping_layer
        # the dimension is 2048 * 14 *14 and part_num * 14 * 14, respectively
    
        # concatenate global feature and the region features for classification
        
        feature_map = feature_map.reshape(-1, 2048, 1, 196)
        feature_map = feature_map.transpose(1,3)
        compressed_feature = self.global_avgpool(feature_map)
        compressed_feature = compressed_feature.reshape(-1, 1, 14, 14)
        
        final_feature = torch.cat((compressed_feature, weighted_feature),dim=1)
        
        res = self.global_avgpool(final_feature)
        res = torch.flatten(res, 1)
        res = self.fc_cls(res)
 
        return res