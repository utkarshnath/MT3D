
---

# Deep Geometric Moments Promote Shape Consistency in Text-to-3D Generation

This repository contains the official implementation of the paper:  
**[Deep Geometric Moments Promote Shape Consistency in Text-to-3D Generation](https://arxiv.org/pdf/2408.05938)**  
📄 [Paper](https://arxiv.org/pdf/2408.05938) | 🌐 [Project Page](https://moment-3d.github.io/)

---

## 🛠️ Prerequisites

- Python 3.11.5  
- CUDA 11.8  

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Build Gaussian Splatting Extension

```bash
cd gs
./build.sh
```

---

## 🚀 Training

### Stage 1: Geometric Refinement

```bash
python main.py --config-name=corgi \
  stage=1 \
  init.prompt="Prompt for Point-E initialization" \
  guidance.control_obj_uid=<Objaverse UID> \
  prompt.prompt="<main prompt>" \
  guidance.guidance_scale=7.5 \
  loss.dgm_step=1 \
  loss.dgm=1e2 \
  guidance.type=controlnet_lora
```

### Stage 2: Texture Refinement

```bash
python main.py --config-name=corgi \
  stage=2 \
  init.prompt="Prompt for Point-E initialization" \
  guidance.control_obj_uid=<Objaverse UID> \
  prompt.prompt="<main prompt>" \
  guidance.guidance_scale=7.5 \
  auxiliary.enabled=false \
  renderer.densify.enabled=true \
  renderer.prune.enabled=true \
  loss.dgm_step=1 \
  loss.dgm=1e2 \
  guidance.type=controlnet_lora \
  ckpt="</path/to/stage_1/checkpoint/>"
```

---

## 📦 Exporting Outputs

Set the Python path:
```bash
export PYTHONPATH="."
```

### Export to `.ply`
```bash
python utils/export.py <your_ckpt> --type ply
```

### Export to `.splat`
```bash
python utils/export.py <your_ckpt> --type splat
```

### Export to Mesh (Shape only)
```bash
python utils/export.py <your_ckpt> \
  --type mesh \
  --batch_size 65536 \
  --reso 256 \
  --K 200 \
  --thresh 0.1
```

**Note:** `<your_ckpt>` can be either a `.pt` checkpoint path or the wandb run ID  
(e.g., `0|213630|2023-10-11|a_high_quality_photo_of_a_corgi`).  
Exported files are saved to `exports/<export-type>/`.

---