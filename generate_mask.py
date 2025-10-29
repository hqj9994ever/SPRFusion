import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
 
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


dir = "inference/prompt"
output_dir = "SAM_SICE/prompt"
sam_checkpoint = "model_zoo/ckpt/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
os.makedirs(output_dir, exist_ok=True)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

for filename in os.listdir(dir):
    if filename.endswith(".png"):
        image = cv2.imread(os.path.join(dir, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(image)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show()


        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
        plt.close()