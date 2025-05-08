# seem_for_lgs

## Installation

- Install [seem_for_lgs](https://github.com/fflahm/seem_for_lgs#)

  ```sh
  git clone git@github.com:fflahm/seem_for_lgs.git
  conda create -n seem python=3.10
  conda activate seem
  pip install numpy==1.23.1
  # Install pytorch 2.1.0 fitting specific cuda version
  # For cuda 12.4, nvcc 11.1, "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118" is expected to work
  conda install -c conda-forge mpi4py mpich
  pip install -r seem_for_lgs/requirements.txt
  pip install git+https://github.com/arogozhnikov/einops.git
  pip install git+https://github.com/MaureenZOU/detectron2-xyz.git
  ```

- Clone [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)

  ```sh
  git clone git@github.com:UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git
  ```

- Download model checkpoints

  - Download https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt to `$SEEMCKPT`

  - **Optional**: Since huggingface connection may be unstable, it is recommended to manually download CLIP tokenizer with git

    ```sh
    cd $CKPTDIR
    git lfs install
    git clone https://huggingface.co/openai/clip-vit-base-patch32
    ```

    - After this, modify `Segment-Everything-Everywhere-All-At-Once/modeling/language/LangEncoder/__init__.py`

    - from

      ```python
          if config_encoder['TOKENIZER'] == 'clip':
              pretrained_tokenizer = config_encoder.get(
                  'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
              )
              tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
              tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
      ```

    - to

      ```python
          if config_encoder['TOKENIZER'] == 'clip':
              pretrained_tokenizer = config_encoder.get(
                  'PRETRAINED_TOKENIZER', '$CKPTDIR/clip-vit-base-patch32'
              )
              tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
              tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
      ```

- Modify SEEM codes
  - Copy `seem_for_lgs/batch_segmentation.py` to `Segment-Everything-Everywhere-All-At-Once/batch_segmentation.py`
  - Replace `Segment-Everything-Everywhere-All-At-Once/modeling/architectures/seem_model_demo.py` with `seem_for_lgs/seem_model_demo.py`
  - Replace `Segment-Everything-Everywhere-All-At-Once/modeling/interface/prototype/attention_data_struct_seemdemo.py` with `seem_for_lgs/attention_data_struct_seemdemo.py`

## Usage

- Make sure path `$DATASET` is like:

```sh
├── $DATASET
│   ├── images
│   │   ├── 00000.png
│   │   ├── 00001.png
│   │   ├── ...
```

```sh
cd Segment-Everything-Everywhere-All-At-Once
python batch_segmentation.py --dataset_path $DATASET --seem_ckpt $SEEMCKPT
```