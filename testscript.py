import numpy as np
from tqdm import tqdm

from ffcv_coco.load_coco import load_coco

# path = '/mnt/nfs/home/branhung/src/datasets/coco-box/val2017.txt'
# imgsz = 640
# batch_size = 16
# stride = 32

# loader, dataset = create_dataloader(path, imgsz, batch_size, stride)
# custom_dataset = CocoBoundingBox(path, imgsz, False)

# allclose = lambda i: np.allclose(dataset[i][1], custom_dataset[i][1])
# print('All image sizes == 640x640?' + str(all([custom_dataset[i][0].size==(640,640) for i in range(len(custom_dataset))])))
# print('\n')
# print('All bounding box labels matching?' + str(all([allclose(i) for i in range(len(dataset))])))
# print('\n')
# print('All shapes matching?' + str(all([dataset[i][3] == custom_dataset[i][3] for i in range(len(dataset))])))

loaders = load_coco()
for excerpt in tqdm(loaders['val']):
    print(excerpt)