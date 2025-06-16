# ------------------------------------------------------------------------------
# Portions of this code are from
# OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Licensed under the Apache License.
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch

from .graph_head_template import GraphHeadTemplate
from .basic_blocks import EdgeConvBlock, gnn_conv, reduce_nodes
import torch_geometric.transforms as T


# TODO: add registry
class GraphHead(GraphHeadTemplate):
    def __init__(self, input_channels, model_cfg, code_size=9, test_cfg=None):
        print('GraphHead')
        self.model_cfg = model_cfg
        if self.model_cfg.TARGET_CONFIG.CLS_SCORE_TYPE == 'recls':
            num_class = 3
        elif self.model_cfg.TARGET_CONFIG.CLS_SCORE_TYPE == 'roi_iou':
            num_class = 1
        super(GraphHead, self).__init__(num_class=num_class, model_cfg=model_cfg)
        self.test_cfg = test_cfg
        self.code_size = code_size
        self.num_class = num_class
        self.edge_radius = model_cfg.EDGE_RADIUS

        pre_channel = input_channels + code_size

        self.gconv_cls = EdgeConvBlock(pre_channel, model_cfg.GRAPH_OUT_DIM, num_updates=model_cfg.NUM_UPDATES)
        self.gconv_reg = EdgeConvBlock(pre_channel, model_cfg.GRAPH_OUT_DIM, num_updates=model_cfg.NUM_UPDATES)

        self.cls_layers = self.make_fc_layers(
            input_channels=self.gconv_cls.F_out,
            output_channels=self.num_class,
            fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=self.gconv_reg.F_out,
            output_channels=code_size,
            fc_list=self.model_cfg.REG_FC
        )

        self.init_weights()

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        else:
            raise NotImplementedError
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def forward(self, batch_dict, training):
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
        cls_feat, reg_feat = gnn_conv(batch_dict, self.gconv_cls, self.gconv_reg)

        cls_feat = cls_feat.unsqueeze(-1).contiguous()  # (BxN, C, 1)
        reg_feat = reg_feat.unsqueeze(-1).contiguous()  # (BxN, C, 1)

        rcnn_cls = self.cls_layers(cls_feat).squeeze(dim=-1).contiguous()  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(reg_feat).squeeze(dim=-1).contiguous()  # (B, C)

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
            targets_dict['batch_mask'] = batch_dict['graph_data'].batch

            self.forward_ret_dict = targets_dict

        return batch_dict




