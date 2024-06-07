import torch
import sys
sys.path.append("/data01/xjy/code/modal")
import ltr
def convert_and_save_weights(pretrained_path, save_path):
    # 使用PyTorch的torch.load来加载权重
    state_dict = torch.load(pretrained_path)

    # 将权重转换到Jittor中（此处假设是转换键名，去除'module.'前缀）
    jittor_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 将转换后的权重字典保存为.pth文件
    torch.save(jittor_state_dict, save_path)


# 调用函数，传入预训练权重路径和要保存的新权重路径
#convert_and_save_weights('ir.pth', 'ir_jt.pth')
convert_and_save_weights('Modal_FPN_ep0044.pth','Modal_FPN_ep0044_jt.pth')
# convert_and_save_weights('resnet50-19c8e357.pth', 'resnet50-19c8e357_jt.pth')

# convert_and_save_weights('vis.pth', 'vis_jt.pth')