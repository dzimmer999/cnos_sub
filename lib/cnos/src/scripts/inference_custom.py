import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from src.utils.bbox_utils import CropResizePad
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
from src.model.utils import Detections, convert_npz_to_json
from src.model.loss import Similarity
from src.utils.inout import save_json_bop23
import cv2
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

np.set_printoptions(threshold=np.inf)

def visualize(rgb, detection_array, save_path="./tmp/tmp.png"):

    print(f"Visualizing with {len(detection_array)} objects.")
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # img = (255*img).astype(np.uint8)

    colors = distinctipy.get_colors(len(detection_array))
    alpha = 0.33
    
    for rgb_idx, detections in enumerate(detection_array):

        for mask_idx, det in enumerate(detections):
            mask = rle_to_mask(det["segmentation"])

            edge = canny(mask)
            edge = binary_dilation(edge, np.ones((2, 2)))

            r = int(255*colors[rgb_idx][0])
            g = int(255*colors[rgb_idx][1])
            b = int(255*colors[rgb_idx][2])
      
            img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
            img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
            img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
            img[edge, :] = 255

    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    prediction.show()
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat
        
def run_inference(template_dir, rgb_path, num_max_dets, conf_threshold, stability_score_thresh):
    start_time = time.time()

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name='run_inference.yaml')
    cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    metric = Similarity()
    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
        
    logging.info("Initializing template")


    template_list = os.listdir(template_dir)
    png_list = [template_img for template_img in template_list if ".png" in template_img]
    n = max([int(template_img.split("_")[1].split(".")[0]) for template_img in png_list])
    
    detection_array = []
    for i in range(n):
        i+=1

        print(f"Generating template (n={i}/{n}, t = {round(time.time() - start_time, 2)}s)")
        template_paths = glob.glob(f"{template_dir}/*_{i}.png")
        boxes, templates = [], []
        for path in template_paths:
            image = Image.open(path)
            boxes.append(image.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            templates.append(image)
    
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        
        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        proposal_processor = CropResizePad(processing_config.image_size)
        #templates = proposal_processor(images=templates, boxes=boxes).cuda()
        templates = proposal_processor(images=templates, boxes=boxes)
        
        save_image(templates, f"{template_dir}/cnos_results/templates_{i}.png", nrow=7)
        ref_feats = model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    )
        logging.info(f"Ref feats: {ref_feats.shape}")
        
        print(f"Running inference (n={i}/{n}, t = {round(time.time() - start_time, 2)}s)")
        # run inference
        rgb = Image.open(rgb_path).convert("RGB")
        #rgb = rgb_path
        detections = model.segmentor_model.generate_masks(np.array(rgb))
        detections = Detections(detections)
        decriptors = model.descriptor_model.forward(np.array(rgb), detections)
        
        # get scores per proposal
        scores = metric(decriptors[:, None, :], ref_feats[None, :, :])
        score_per_detection = torch.topk(scores, k=5, dim=-1)[0]
        score_per_detection = torch.mean(
            score_per_detection, dim=-1
        )
        
        # get top-k detections
        scores, index = torch.topk(score_per_detection, k=num_max_dets, dim=-1)
        detections.filter(index)
        
        # keep only detections with score > conf_threshold
        detections.filter(scores>conf_threshold)
        detections.add_attribute("scores", scores)
        detections.add_attribute("object_ids", torch.zeros_like(scores))
            
        detections.to_numpy()
        save_path = f"{template_dir}/cnos_results/detection"
        detections.save_to_file(0, 0, 0, save_path, "custom", return_results=False)
        detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
        save_json_bop23(save_path+".json", detections)
        detection_array.append(detections)
        #print(f"{detection_array=}")

    vis_img = visualize(rgb, detection_array)
    vis_img.save(f"{template_dir}/cnos_results/vis.png")
    print(f"Done after {time.time() - start_time} seconds.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", nargs="?", help="Path to root directory of the template")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--num_max_dets", nargs="?", default=3, type=int, help="Number of max detections")
    parser.add_argument("--confg_threshold", nargs="?", default=0.5, type=float, help="Confidence threshold")
    parser.add_argument("--stability_score_thresh", nargs="?", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()

    os.makedirs(f"{args.template_dir}/cnos_results", exist_ok=True)
    run_inference(args.template_dir, args.rgb_path, num_max_dets=args.num_max_dets, conf_threshold=args.confg_threshold, stability_score_thresh=args.stability_score_thresh)