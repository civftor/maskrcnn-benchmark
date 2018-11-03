import os
import json
from PIL import Image
import torch
import torchvision
import torch.utils.data as data
from maskrcnn_benchmark.structures.bounding_box import BoxList

class SpireDataset(data.Dataset):
    def __init__(self, save_path, img_dir, ann_dir, transforms=None, ):
        # as you would do normally
        ann_path = os.path.join(save_path, ann_dir)
        img_path = os.path.join(save_path, img_dir)
        anns = os.listdir(ann_path)
        self.root = save_path
        self.img_path = img_path
        self.anns = []
        self.categories = []
        for ann in anns:
            _, ext = os.path.splitext(ann)
            if ext == '.json':
                with open(os.path.join(ann_path, ann), 'r') as load_f:
                    load_dict = json.load(load_f)
                    file_name = load_dict['file_name']
                    height = load_dict['height']
                    width = load_dict['width']
                    # print("{}, {}, {}".format(file_name, height, width))
                    ann_list = load_dict['annos']
                    # print(len(ann_list))
                    if len(ann_list) > 0:
                        self.anns.append(load_dict)
                        for each_ann in ann_list:
                            if each_ann['category_name'] not in self.categories:
                                self.categories.append(each_ann['category_name'])
                            each_ann['category_id'] = self.categories.index(each_ann['category_name'])

        self.transforms = transforms

    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, idx):
        img_name = self.anns[idx]['file_name']
        # load the image as a PIL Image
        img = Image.open(os.path.join(self.img_path, img_name)).convert('RGB')
        anno = self.anns[idx]['annos']

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        # classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno[0].has_key("segmentation") and len(anno[0]["segmentation"]) > 0:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size)
            target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": self.anns[idx]['height'], "width": self.anns[idx]['width']}
