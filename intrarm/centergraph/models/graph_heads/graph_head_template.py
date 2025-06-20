# ------------------------------------------------------------------------------
# Portions of this code are from
# OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Licensed under the Apache License.
# ------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from centergraph.core.bbox import box_torch_ops
from .target_assigner.proposal_target_layer import ProposalTargetLayer
from centergraph.utils import SigmoidFocalClassificationLoss


def limit_period(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period


class GraphHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)

        self.forward_ret_dict = None

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C), [3778, 9]
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1), [3778, 10]
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        roi_ry = limit_period(rois[:, 6], offset=0.5, period=np.pi*2)

        gt_of_rois[:, :6] = gt_of_rois[:, :6] - rois[:, :6]
        gt_of_rois[:, 6] = gt_of_rois[:, 6] - roi_ry

        gt_of_rois = box_torch_ops.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(-1, gt_of_rois.shape[-1])

        if rois.shape[-1] == 9:
            # rotate velocity
            gt_of_rois[:, 7:-1] = gt_of_rois[:, 7:-1] - rois[:, 7:]

            """
            roi_vel = gt_of_rois[:, :, 7:-1]
            roi_vel = torch.cat([roi_vel, torch.zeros([roi_vel.shape[0], roi_vel.shape[1], 1]).to(roi_vel)], dim=-1)

            gt_of_rois[:, :, 7:-1] = box_torch_ops.rotate_points_along_z(
                points=roi_vel.view(-1, 1, 3), angle=-roi_ry.view(-1)
            ).view(batch_size, -1, 3)[..., :2]
            """

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, 6] = heading_label

        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = forward_ret_dict['rcnn_reg'].shape[-1]
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'L1':
            reg_targets = gt_boxes3d_ct.view(rcnn_batch_size, -1)
            rcnn_loss_reg = F.l1_loss(
                rcnn_reg.view(rcnn_batch_size, -1),
                reg_targets,
                reduction='none'
            )  # [B, M, 7]

            tb_dict['rcnn_loss_dict'] = (rcnn_loss_reg.detach().view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum(0) / max(
                fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * rcnn_loss_reg.new_tensor(loss_cfgs.LOSS_WEIGHTS['code_weights'])
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(
                fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.detach().item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)

        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            #batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(),
            #                                        reduction='none')
            batch_loss_cls = F.binary_cross_entropy_with_logits(rcnn_cls_flat, rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.detach().item()}
        return rcnn_loss_cls, tb_dict

    def get_box_cls_layer_loss_reclassify(self, forward_ret_dict):
        batch_mask = forward_ret_dict['batch_mask'].view(-1)
        batch_size = int(batch_mask[-1] + 1)
        rcnn_cls = forward_ret_dict['rcnn_cls'].view(-1)
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)

        cared = rcnn_cls_labels >= 0
        positives = rcnn_cls_labels > 0
        negatives = rcnn_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()

        pos_normalizer = torch.zeros(cls_weights.shape[0], device=cls_weights.device)
        for i in range(batch_size):
            cur_batch_mask = batch_mask == i
            cur_sum = torch.sum(positives[cur_batch_mask])
            pos_normalizer[cur_batch_mask] = cur_sum.float()

        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = rcnn_cls_labels * cared.type_as(rcnn_cls_labels)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=rcnn_cls.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = rcnn_cls.view(one_hot_targets.shape[0], self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]

        loss_func = SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        cls_loss_src = loss_func(cls_preds, one_hot_targets, weights=cls_weights)
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': cls_loss.detach().item()}
        return cls_loss, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        if self.model_cfg.TARGET_CONFIG.CLS_SCORE_TYPE == 'recls':
            rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss_reclassify(self.forward_ret_dict)
        elif self.model_cfg.TARGET_CONFIG.CLS_SCORE_TYPE == 'roi_iou':
            rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, rois, cls_preds, box_preds):
        """
        :param batch_mask: (1465)
        :param rois: [1465, 9]
        :param cls_preds: [1465, 1]
        :param box_preds: [1465, 9]
        :return:
        """

        code_size = box_preds.shape[-1]
        # batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        # batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, 6].view(-1)
        roi_xyz = rois[:, 0:3].view(-1, 3)

        local_rois = rois.clone().detach()
        local_rois[:, 0:3] = 0

        batch_box_preds = (box_preds + local_rois).view(-1, code_size)
        batch_box_preds = box_torch_ops.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)

        batch_box_preds[:, 0:3] += roi_xyz
        # batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)

        return cls_preds, batch_box_preds