import json
import os
import shutil
import os.path as osp
import mmcv
from tqdm import tqdm
import glob

image_id = 0
annotation_id = 0

mode='val'

adenomatous_json_dir = '/data3/publicData/Anti-UAV410/Anti-UAV/'+mode
image_root = ''
dataset_root = ''

out_path='annotationsnew/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
out_json = out_path+mode+'.json'

merged_data = {
                "licenses": [{"name": "", "id": 0, "url": ""}],
                "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
                "categories": [{"id": 1, "name": "uav", "supercategory": ""}],
                "images": [],
                "annotations": []
}
img_id_test = set()
imgs_num = 0
anno_num = 0


myid=0

while True:

    finishedflag=True

    for idx, f in tqdm(enumerate(os.listdir(adenomatous_json_dir))):

        json_dir = os.path.join(adenomatous_json_dir, f, 'IR_label.json')

        with open(json_dir) as json_file:

            data = json.load(json_file)
            exist = data['exist']
            annos = data['gt_rect']
            images = glob.glob(os.path.join(adenomatous_json_dir, f, '*.jpg'))
            assert len(images) == len(annos),"image length %d not equal to anno length %d: %s" %(len(images), len(annos), f)
            # if idx == 0:
            #     merged_data["licenses"] = data["licenses"]
            #     merged_data["info"] = data["info"]
            #     merged_data["categories"] = data["categories"]

            id_list = set()

            if myid>len(annos)-1:
                break
            else:
                finishedflag=False
                img = {}            
                img['id'] = len(merged_data["images"]) + 1
                img['file_name'] = osp.join(f, str(myid + 1).zfill(6) + '.jpg')
                img_ori = mmcv.imread(osp.join(adenomatous_json_dir, img['file_name']))
                img['height'] = img_ori.shape[0]
                img['width'] = img_ori.shape[1]
                img['license'] = 0
                img['flickr_url'] = ''
                img['coco_url'] = ''
                img['date_captured'] = 0
                merged_data["images"].append(img)
                
                if exist[myid] == 1:
                    if len(annos[myid]):
                        anno = {}
                        anno['id'] = len(merged_data["annotations"]) + 1
                        anno['category_id'] = 1
                        anno['image_id'] = img['id']
                        anno['bbox'] = annos[myid]
                        anno['segmentation'] = []
                        anno['area'] = anno['bbox'][2] * anno['bbox'][3]
                        anno['iscrowd'] = 0
                        anno['attributes'] = {"occluded": False}
                        merged_data["annotations"].append(anno)
                
    myid=myid+5
    if finishedflag:
        break

print('images %d, annos %d'%(len(merged_data["images"]), len(merged_data["annotations"])))

with open(out_json, 'w') as out_file:
    json.dump(merged_data, out_file)
