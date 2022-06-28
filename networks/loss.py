import torch
import torch.nn as nn

class channel_grouping_loss(nn.Module):
    def __init__(self):
        super(channel_grouping_loss, self).__init__()
        
    def forward(self, feature):
        device = feature.device
        
        dis_loss = torch.zeros(1).to(device)
        div_loss = torch.zeros(1).to(device)

        mgr = feature.mean()

        for sample in feature:
            max_indexes = channel_grouping_loss.get_max_index(sample)

            for region_index, region in enumerate(sample):        
                max_x, max_y = max_indexes[region_index]

                for i in range(region.shape[0]):
                    for j in range(region.shape[1]):
                        dis_loss += (region[i,j] * region[i,j]) * ((max_x - i) * (max_x - i) + (max_y - j) * (max_y - j))

                        if region_index == 0:
                            max_others = max(sample[(region_index+1):, i, j])
                        else:
                            max_others = max(torch.cat((sample[:region_index, i, j], sample[(region_index+1):, i, j]), dim=0))

                        div_loss += (region[i,j] * region[i,j]) * ((max_others - mgr) * (max_others - mgr))
        
        
        shape = feature.shape
        sample_num, region_num, weight, height = shape[0], shape[1], shape[2], shape[3]               
        sum_num = sample_num * region_num * weight * height

        return dis_loss/sum_num, div_loss/sum_num
    
    @staticmethod
    def get_max_index(tensor):
        shape = tensor.shape
        indexes = []
        for i in range(shape[0]):
            mx = tensor[i, 0, 0]
            x, y = 0, 0
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if tensor[i, j, k] > mx:
                        mx = tensor[i, j, k]
                        x, y = j, k
            indexes.append([x, y])
        return indexes