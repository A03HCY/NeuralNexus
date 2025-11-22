import kagglehub
from pycocotools.coco import COCO
from .coco_data import COCOSegmentationDataset
import torchvision.transforms as transforms

path = kagglehub.dataset_download("pkdarabi/brain-tumor-image-dataset-semantic-segmentation")

train_path = path + '/train'
test_path = path + '/test'
valid_path = path + '/valid'

train_info = train_path + '/_annotations.coco.json'
test_info = test_path + '/_annotations.coco.json'
valid_info = valid_path + '/_annotations.coco.json'

train_coco = COCO(train_info)
test_coco = COCO(test_info)
valid_coco = COCO(valid_info)

norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = COCOSegmentationDataset(train_coco, train_path, transform=norm)
valid_dataset = COCOSegmentationDataset(valid_coco, valid_path, transform=norm)
test_dataset = COCOSegmentationDataset(test_coco, test_path, transform=norm)