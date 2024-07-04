import jittor as jt


if __name__ == '__main__':

    old_ckp_file = 'libs/swintransformer/checkpoints/cascade_mask_rcnn_swin_tiny_patch4_window7.pth'

    old_state_dict = jt.load(old_ckp_file)['state_dict']

    from collections import OrderedDict

    new_state_dict = OrderedDict()

    # 只保留backbone和neck的值
    for key, value in old_state_dict.items():

        print(key)
        if key[0:8] == 'backbone':
            new_state_dict[key] = value
        if key[0:4] == 'neck':
            new_state_dict[key] = value


    # open in torch 1.4.0
    jt.save(new_state_dict,
               'pretrained_weights/cascade_mask_rcnn_swin_tiny.pth.tar',
               _use_new_zipfile_serialization=False)

