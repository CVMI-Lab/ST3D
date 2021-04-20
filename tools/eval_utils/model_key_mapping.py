import os, sys
import torch

src = torch.load(sys.argv[1])  # source

src = src['model_state']


def second_key_mapping():
    src = torch.load(sys.argv[1])
    src = src['model_state']

    key_mapping = {
        'rpn_net.conv': 'backbone_3d.conv',
        'rpn_head.deblocks.': 'backbone_2d.deblocks.',
        'rpn_head.conv_cls.': 'dense_head.conv_cls.',
        'rpn_head.conv_box.': 'dense_head.conv_box.',
        'rpn_head.conv_dir_cls.': 'dense_head.conv_dir_cls.',
        'rpn_head.blocks.': 'backbone_2d.blocks.',

        # For PartA2
        'rpn_net.inv': 'backbone_3d.inv',
        'rpn_net.seg_cls_layer': 'point_head.cls_layers.0',
        'rpn_net.seg_reg_layer': 'point_head.part_reg_layers.0',

        'rcnn_net.shared_fc_layer.0.conv': 'roi_head.shared_fc_layer.0',
        'rcnn_net.shared_fc_layer.0.bn.bn': 'roi_head.shared_fc_layer.1',
        'rcnn_net.shared_fc_layer.2.conv': 'roi_head.shared_fc_layer.4',
        'rcnn_net.shared_fc_layer.2.bn.bn': 'roi_head.shared_fc_layer.5',
        'rcnn_net.shared_fc_layer.4.conv': 'roi_head.shared_fc_layer.8',
        'rcnn_net.shared_fc_layer.4.bn.bn': 'roi_head.shared_fc_layer.9',

        'rcnn_net.cls_layer.0.conv': 'roi_head.cls_layers.0',
        'rcnn_net.cls_layer.0.bn.bn': 'roi_head.cls_layers.1',
        'rcnn_net.cls_layer.2.conv': 'roi_head.cls_layers.4',
        'rcnn_net.cls_layer.2.bn.bn': 'roi_head.cls_layers.5',
        'rcnn_net.cls_layer.3.conv': 'roi_head.cls_layers.7',

        'rcnn_net.reg_layer.0.conv': 'roi_head.reg_layers.0',
        'rcnn_net.reg_layer.0.bn.bn': 'roi_head.reg_layers.1',
        'rcnn_net.reg_layer.2.conv': 'roi_head.reg_layers.4',
        'rcnn_net.reg_layer.2.bn.bn': 'roi_head.reg_layers.5',
        'rcnn_net.reg_layer.3.conv': 'roi_head.reg_layers.7',

        'rcnn_net.conv_': 'roi_head.conv_',
    }

    # for idx in [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17]:
    #     key_mapping['rpn_head.blocks.0.%d.' % idx] = 'backbone_2d.blocks.0.%d.' % (idx - 1)
    #     key_mapping['rpn_head.blocks.1.%d.' % idx] = 'backbone_2d.blocks.1.%d.' % (idx - 1)

    src_key_list = list(src.keys())
    num_replaced = 0
    for src_key in src_key_list:
        found = False
        for key, val in key_mapping.items():
            if key in src_key:
                new_key = src_key.replace(key, val)
                src[new_key] = src.pop(src_key)
                num_replaced += 1
                print('Replace: %s => %s' % (src_key, new_key))
                found = True
        if not found:
            print('Skip: %s' % src_key)

    print('Total replaced keys: %d / %d' % (num_replaced, len(src)))

    ans = {'model_state': src}
    new_path = os.path.join(os.path.dirname(sys.argv[1]), 'new_mapped_model.pth')
    torch.save(ans, new_path)


if __name__ == '__main__':
    second_key_mapping()
