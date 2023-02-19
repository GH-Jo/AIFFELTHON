import os
import mask_rcnn.myeval
from PIL import Image
from time import time

data_folder = './dataset1/dent/test/images'
test_img_flist = [ fname for fname in os.listdir(data_folder) if fname.endswith('jpg')or fname.endswith('jpeg') ]

pil_img = Image.open(os.path.join(data_folder,test_img_flist[0]))
mask_rcnn.myeval.model_init()
start_tm =time()
img, result = mask_rcnn.myeval.model_predict(pil_img)
end_tm = time()
print(f'mask rcnn {result},{end_tm-start_tm:.3f}sec')
Image.fromarray(img).save('1.png')

from unets import func

func.model_init()
from resnet50_trans1 import resnet50_trans
resnet50_trans.model_init()

start_tm = time()
result = resnet50_trans.predict(pil_img)
end_tm = time()
print(f'CNN {end_tm-start_tm:.3f}sec')

if result:
    print('파손')
else: 
    print('정상')
    end_tm = time()
    print(f'끝 {end_tm-start_tm:.3f}sec')

    exit(0)

print(os.getcwd())
#pil_img = Image.open(os.path.join(data_folder,test_img_flist[0]))
start_tm = time()
img, result = func.predict(pil_img)
end_tm = time()
print(f'UNet {result},{end_tm-start_tm:.3f}sec')
Image.fromarray(img).save('2.png')



mask_rcnn.myeval.model_init()
start_tm =time()
img, result = mask_rcnn.myeval.model_predict(pil_img)
end_tm = time()
print(f'mask rcnn {result},{end_tm-start_tm:.3f}sec')
Image.fromarray(img).save('1.png')
