import os
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.arguments import load_opt_from_config_files
from utils.distributed import init_distributed

def get_text_embeddings(model, text):
    t_emb = model.model.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(text, name='grounding', token=False, norm=False)['class_emb']
    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7) # [1, d]
    return t_emb

if __name__ == "__main__":
    # Load SEEM model
    opt = load_opt_from_config_files(["configs/seem/focall_unicl_lang_demo.yaml"])
    opt = init_distributed(opt)
    pretrained_pth = os.path.join("seem_focall_v0.pt")
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    # Get example text embeddings
    example_text = ["cat", "dog", "car"]
    text_embeddings = get_text_embeddings(model, example_text)
    print(text_embeddings.shape) # [3, d]