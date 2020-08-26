import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import generate_all_anchors

class OCR_Loss(nn.Module):
    def __init__(self, noobject_scale, coord_scale, class_scale):
        super(OCR_Loss, self).__init__()
        self.noobject_scale  = noobject_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale
        
        
    def forward(self, pred, target):
        '''
        yolo loss:
        pred: (loc_pred, conf_pred, class_score)
        target: (iou_target, iou_mask, box_target, box_mask, class_target, class_mask)
        
        loc_pred: (b, h*w*num_anchor, 4) -> sigma(t_x), sigma(t_y), sigma(t_w), sigma(t_h)
        conf_pred: (b, h*w*num_anchor, 1) -> sigma(t_c) iou loss ?
        class_score: (b, h*w*num_anchor, num_classes) -> predicted class scores
        
        iou_target: (b, h*w*num_anchor, 1)
        iou_mask: (b, h*w*num_anchor, 1)
        box_target: (b, h*w*num_anchor, 4)
        box_mask: (b, h*w*num_anchor, 1)
        class_target: (b, h*w*num_anchor, 1)
        class_mask: (b, h*w*num_anchor, 1)
        '''
        
        loc_pred, conf_pred, class_score = pred
        iou_target, iou_mask, box_target, \
        box_mask, class_target, class_mask = target
        
        batch_size, _, num_classes = class_score.size()
        class_score  = class_score.view(-1, num_classes)
        class_target = class_target.view(-1)
        class_mask   = class_mask.view(-1)
        
        # ignore non-object target
        class_keep = class_mask.nonzero(as_tuple=False).squeeze(1)
        class_score_keep  = class_score[class_keep, :]
        class_target_keep = class_target[class_keep]
        
        # calculate loss
        box_loss   = self.coord_scale * F.mse_loss(loc_pred * box_mask, box_target * box_mask, reduction='sum') / batch_size
        iou_loss   = self.noobject_scale * F.mse_loss(conf_pred * iou_mask, iou_target * iou_mask, reduction='sum') / batch_size
        class_loss = self.class_scale * F.cross_entropy(class_score_keep, class_target_keep) / batch_size
        
        return box_loss, iou_loss, class_loss

def build_target(pred, target, height, width, anchors):
    '''
    pred: (loc_pred, conf_pred, class_score)
    gt:   (gt_boxes, gt_classes, num_objs)
    '''
    loc_pred, conf_pred, class_score = pred
    gt_boxes, gt_classes, num_objs   = target
    
    batch_size = pred.size(0)
    
    # init target variable
    iou_target = loc_pred.new_zeros((batch_size, height * width, num_anchors, 1))
    iou_mask   = loc_pred.new_ones((batch_size, height * width, num_anchors, 1))
    
    box_target = loc_pred.new_zeros((batch_size, height * width, num_anchors, 4))
    box_mask   = loc_pred.new_ones((batch_size, height * width, num_anchors, 1))
    
    class_target = conf_pred.new_zeros((batch_size, height * width, num_anchors, 1))
    class_mask   = conf_pred.new_zeros((batch_size, height * width, num_anchors, 1))
    
    # 
    num_anchors = len(anchors)
    anchors = torch.tensor(anchors, dtype=torch.float)
    
    
    
if __name__ == '__main__':
    loc_pred = torch.randn(1, 2*2*5, 4)
    conf_pred = torch.randn(1, 2*2*5, 1)
    class_score = torch.randn(1, 2*2*5, 3)
    
    iou_target = torch.randn(1, 2*2*5, 1)
    iou_mask = iou_target > 0.5
    
    box_target = torch.randn(1, 2*2*5, 4)
    box_mask = iou_mask.clone()
    
    class_target = torch.randint(0, 3, (1, 2*2*5, 1))
    class_mask = iou_target.clone()
    
    ocr_loss = OCR_Loss(1, 1, 1)
    pred = (loc_pred, conf_pred, class_score)
    
    target = (iou_target, iou_mask, box_target, \
              box_mask, class_target, class_mask)
    
    loss = ocr_loss(pred, target)
    
    print(loss)
    anchors = torch.randn(5,2)
    generate_all_anchors(anchors,3,3)
    