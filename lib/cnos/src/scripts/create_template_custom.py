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

def run_inference(template_dir, stability_score_thresh, return_model=False):

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
    ref_feats_list = []

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
        
        save_image(templates, f"{template_dir}/cnos_results/templates_test_{i}.png", nrow=7)

        ref_feats = model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken")
        ref_feats_list.append(ref_feats)

        logging.info(f"Ref feats: {ref_feats.shape}")
        
        torch.save(templates, f"{template_dir}/cnos_results/templates_test_{i}.pt")
        
    if return_model is True:
        return model, ref_feats_list

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", nargs="?", help="Path to root directory of the template")
    parser.add_argument("--stability_score_thresh", nargs="?", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()

    os.makedirs(f"{args.template_dir}/cnos_results", exist_ok=True)
    run_inference(args.template_dir, stability_score_thresh=args.stability_score_thresh)