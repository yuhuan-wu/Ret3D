# ------------------------------------------------------------------------------
# Portions of this code are from
# OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Licensed under the Apache License.
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch

from .graph_head_template import GraphHeadTemplate
from .basic_blocks import EdgeConvBlock, gnn_conv_single, reduce_nodes
import torch_geometric.transforms as T


# TODO: add registry
class MiXHead(GraphHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, code_size=9, add_rois_and_scores=False, test_cfg=None):
        super(MiXHead, self).__init__(num_class=num_class, model_cfg=model_cfg)
        print('MiXHead')
        self.model_cfg = model_cfg
        self.test_cfg = test_cfg
        self.code_size = code_size
        self.edge_radius = model_cfg.EDGE_RADIUS
        self.use_knn = model_cfg.USE_KNN
        self.add_rois_and_scores = model_cfg.get('ADD_ROIS_SCORES', False)

        pre_channel = input_channels

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=code_size,
            fc_list=self.model_cfg.REG_FC
        )

        self.cls_layers = self.cls_layers[:-1]
        self.reg_layers = self.reg_layers[:-1]

        if self.add_rois_and_scores:
            for p in self.parameters():
                p.requires_grad = False
        
        self.gconv_cls = EdgeConvBlock(self.model_cfg.CLS_FC[-1] + 7, model_cfg.GRAPH_OUT_DIM, num_updates=model_cfg.NUM_UPDATES)
        self.gconv_reg = EdgeConvBlock(self.model_cfg.REG_FC[-1] + 7, model_cfg.GRAPH_OUT_DIM, num_updates=model_cfg.NUM_UPDATES)

        self.cls_f = nn.Conv1d(model_cfg.GRAPH_OUT_DIM, num_class, kernel_size=1, bias=True)
        self.reg_f = nn.Conv1d(model_cfg.GRAPH_OUT_DIM, code_size, kernel_size=1, bias=True)

        self.init_weights()
    
    
    def init_weights(self, weight_init='xavier'):
        if weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        else:
            raise NotImplementedError
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_f.weight, mean=0, std=0.001)

    def forward(self, batch_dict, training):
        if self.use_knn:
            batch_dict['graph_data'] = T.KNNGraph(k=int(self.edge_radius))(batch_dict['graph_data'])
        else:
            batch_dict['graph_data'] = T.RadiusGraph(r=self.edge_radius)(batch_dict['graph_data'])
        batch_dict['batch_size'] = len(batch_dict['frame_id'])
        if training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_features'] = targets_dict['roi_features']
        else:
            batch_dict['rois'] = torch.cat([batch_dict['graph_data'].pos, batch_dict['graph_data'].box], dim=-1)
            batch_dict['roi_features'] = batch_dict['graph_data'].x

        # IMPORTANT, add rois and roi_scores to 
        if self.add_rois_and_scores:
            batch_dict['roi_features'] = torch.cat([batch_dict['roi_features'], batch_dict['rois'], batch_dict['graph_data'].roi_scores.unsqueeze(-1)], dim=1)
        
        pooled_features = batch_dict['roi_features'].reshape(-1, 1, batch_dict['roi_features'].shape[-1]).contiguous()

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).contiguous()  # (BxN, C, 1)
        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))  # 128, 256, 1

        rcnn_cls_feats = self.cls_layers(shared_features).contiguous().squeeze(-1)
        rcnn_reg_feats = self.reg_layers(shared_features).contiguous().squeeze(-1)

        rcnn_cls_feats = gnn_conv_single(batch_dict, rcnn_cls_feats, self.gconv_cls)
        rcnn_reg_feats = gnn_conv_single(batch_dict, rcnn_reg_feats, self.gconv_reg)

        rcnn_cls = self.cls_f(rcnn_cls_feats.unsqueeze(-1)).contiguous().squeeze(dim=-1)
        rcnn_reg = self.reg_f(rcnn_reg_feats.unsqueeze(-1)).contiguous().squeeze(dim=-1)

        if self.model_cfg.REDUCE_NODE and self.model_cfg.GRAPH_TYPE == 't':
            if training:
                input_list = [targets_dict['rcnn_cls_labels'], targets_dict['reg_valid_mask'],
                              targets_dict['gt_of_rois'], rcnn_cls, rcnn_reg, batch_dict['graph_data'].batch]
                cl = batch_dict['graph_data'].cl
                jl = batch_dict['graph_data'].jl
                targets_dict['rcnn_cls_labels'], targets_dict['reg_valid_mask'], targets_dict['gt_of_rois'], rcnn_cls, rcnn_reg, batch_dict['graph_data'].batch = reduce_nodes(input_list, cl, jl)
            else:
                input_list = [batch_dict['rois'], rcnn_cls, rcnn_reg,
                              batch_dict['graph_data'].batch, batch_dict['graph_data'].roi_labels,
                              batch_dict['graph_data'].roi_scores]
                cl = batch_dict['graph_data'].cl
                jl = batch_dict['graph_data'].jl
                batch_dict['rois'], rcnn_cls, rcnn_reg, batch_dict['graph_data'].batch, batch_dict['graph_data'].roi_labels,batch_dict['graph_data'].roi_scores = reduce_nodes(input_list, cl, jl)

        if not training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict




