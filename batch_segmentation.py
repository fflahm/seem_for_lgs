import os
from PIL import Image

import torch
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
import argparse

from modeling.BaseModel import BaseModel
from modeling import build_model
from modeling.modules.postprocessing import sem_seg_postprocess
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.structures import ImageList
from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def process_batch_images(model, img_folder, data_list, save_folder, seg_folder):
    device = model.model.device
    image_oris = [np.asarray(Image.open(os.path.join(img_folder, p)).convert("RGB")) for p in data_list]
    image_oris = np.stack(image_oris, axis=0)  # [B, H, W, 3] We assume all images are the same size
    B, H, W, _ = image_oris.shape
    images = [torch.from_numpy(x.copy()).permute(2,0,1).to(device) for x in image_oris]
    images = [(x - model.model.pixel_mean) / model.model.pixel_std for x in images] # [B, 3, H, W]
    images = ImageList.from_tensors(images, model.model.size_divisibility)
    _, _, H_, W_ = images.tensor.shape

    features = model.model.backbone(images.tensor)
    mask_features, _, multi_scale_features = model.model.sem_seg_head.pixel_decoder.forward_features(features)
    outputs = model.model.sem_seg_head.predictor(multi_scale_features, mask_features)

    pred_masks_results = outputs['pred_masks'] # [B, Q, h, w]
    pred_masks_results = F.interpolate(
        pred_masks_results,
        size=(H_, W_),
        mode="bilinear",
        align_corners=False,
    ) # [B, Q, H', W']

    for i in range(B):
        pred_masks = pred_masks_results[i]  # [Q, h, w]
        pred_masks = sem_seg_postprocess(pred_masks, (H, W), H, W) # [Q, H, W]

        cls_masks = outputs['pred_logits'][i]  # [Q, C]
        v_emb = outputs['pred_captions'][i]    # [Q, d]
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

        panoptic_results = model.model.panoptic_inference(cls_masks, pred_masks)
        # panoptic_results[0] [H, W] int for segmentation/query/instance id filtered
        # panoptic_results[1] list of dicts for instance info
        pano_seg, pano_seg_info = panoptic_results
        seg_map = (pano_seg - 1).unsqueeze(0) # [1, H, W]
        img_embed = torch.zeros(len(pano_seg_info), v_emb.shape[1]).to(device) # [q, d]
        for j, seg in enumerate(pano_seg_info):
            img_embed[j] = v_emb[seg["instance_id"]]

        image_name = data_list[i].split('.')[0]
        save_path = os.path.join(save_folder, image_name)
        save_path_s = save_path + '_s.npy'
        save_path_f = save_path + '_f.npy'
        np.save(save_path_s, seg_map.detach().cpu().numpy())
        np.save(save_path_f, img_embed.detach().cpu().numpy())

        visual = Visualizer(image_oris[i], metadata=model.model.metadata)
        pano_img = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image
        pano_img = Image.fromarray(pano_img.get_image())
        pano_img.save(os.path.join(seg_folder, image_name) + '.png')
    del outputs
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/focall_unicl_lang_demo.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset folder")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing images")
    parser.add_argument("--seem_ckpt", type=str, help="Path to the SEEM checkpoint")
    cfg = parser.parse_args()
    dataset_path = cfg.dataset_path
    batch_size = cfg.batch_size
    pretrained_pth = cfg.seem_ckpt
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()
    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)
    seg_folder = os.path.join(dataset_path, 'seem_results')
    os.makedirs(seg_folder, exist_ok=True)

    opt = load_opt_from_config_files([cfg.conf_files])
    opt = init_distributed(opt)
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False
    model.model.metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    def split_list(lst, batch_size):
        return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

    batches = split_list(data_list, batch_size)
    torch.cuda.empty_cache()
    for batch in tqdm(batches):
        process_batch_images(model, img_folder, batch, save_folder, seg_folder)
        torch.cuda.empty_cache()
