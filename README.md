# HiT-RSNet
This work introduces HiT-RSNet, a hybrid transformer–convolutional super-resolution framework for remote sensing imagery. Existing transformer-based SR methods often fail to preserve fine textures, sharp boundaries, and spectral–spatial consistency due to limited global modeling and weak local refinement.

HiT-RSNet addresses these issues through a dual-branch design:

1- Hierarchical Region Transformer Blocks (HRTB): Capture both local and global dependencies via channel-wise attention (CWSAB), hierarchical spatial attention (HSAB), and multi-layer feed-forward blocks (MLFFB).

2- Residual Convolutional Attention Modules (RCAM): Refine local structures with convolutional attention, enabling robust texture recovery across varied spatial scales.

3- Cross-Path Fusion (CPF): Integrates transformer and convolutional features adaptively for balanced global–local representation.

4- Multi-stage feature refinement: Ensures progressive enhancement of intermediate layers, reducing noise and preserving fine details.

The model is evaluated on four benchmark datasets (UCMerced, AID, RSCNN7, WHU-RS19) under ×2, ×3, and ×4 upscaling factors. Results consistently show state-of-the-art performance, with HiT-RSNet surpassing CNN-based (VDSR, HSENet) and transformer-based (TransENet, SwinIR) baselines in PSNR, SSIM, SCC, and SAM metrics.

Overall, HiT-RSNet demonstrates the importance of hybrid architectures for remote sensing SR, achieving sharper boundaries, improved edge alignment, and better spectral consistency, which are crucial for tasks like urban monitoring, land-use mapping, and disaster response.

# Coding Hierarchy
```bash
HiT-RSNet/
│── README.md  
│── requirements.txt
│── datasets/
│   ├── UCMerced/
│   ├── AID/
│   ├── RSCNN7
│   ├── WHU-RS19
│── models/
│   ├── HiT-RSNet
│   ├── SOTA Models
│── Training/
│   ├── options/
│   ├── utilities/
│── Testing/
│── results/
│   ├── Pre-trained/
│   ├── Output images/ 
│── LICENSE
│── CITATION.cff  # Citation info
```

# Environment and Dependencies
```bash
# create a new environment
conda create -n hit-rsnet python=3.10

# activate environment
conda activate hit-rsnet

# install dependencies
pip install -r requirements.txt
```

#Datasets

**HiT-RSNet** evaluated on four widely used remote sensing benchmark datasets.  

| Dataset    | Classes | Total Images | Resolution | Image Size | Link | Notes |
|------------|---------|--------------|------------|------------|--------|-------|
| **UCMerced** | 21      | 2,100        | 0.3 m      | 256×256    | [Link](http://weegee.vision.ucmerced.edu/datasets/landuse.html)  | Land-use dataset with diverse urban & natural scenes (e.g., agriculture, residential, freeways). |
| **AID**     | 30      | 10,000       | 0.5 m      | 600×600    | [Link](https://captain-whu.github.io/AID/)  | Large-scale aerial dataset covering airports, bridges, deserts, resorts, farmlands, and more. Very diverse. |
| **RSCNN7**  | 7       | 2,800        | 0.2 m      | 400×400    | [Link](https://figshare.com/articles/dataset/RSSCN7_Image_dataset/7006946) | Contains grassland, forest, farmland, parking lots, industrial regions, etc. High intra-class variation. |
| **WHU-RS19**| 19      | 1,005        | 0.5 m      | 600×600    | [Link](https://captain-whu.github.io/BED4RS/)   | Extracted from Google Earth. Includes beaches, ports, commercial areas, and urban infrastructure. |


#Training
```bash
# scale 2
python training.py --model=HiT-RSNet --dataset=UCMerced-dataset --scale=2
# scale 3
python training.py --model=HiT-RSNet --dataset=UCMerced-dataset --scale=3
# scale 4
python training.py --model=HiT-RSNet --dataset=UCMerced-dataset --scale=4
```

#Testing
```bash
# scale 2
python testing.py --model=HiT-RSNet --scale=2
# scale 3
python testing.py --model=HiT-RSNet --scale=3
# scale 4
python testing.py --model=HiT-RSNet --scale=4
```

# Acknowledgement
This code is built on [HSENet](https://github.com/Shaosifan/HSENet), [TransENet(Pytorch)](https://github.com/Shaosifan/TransENet) and [HAUNet(Pytorch)](https://github.com/likakakaka/HAUNet_RSISR). We are thankful to the authors for sharing their code for reproducibility. We are also thankful to King Mongkut's University of Technology Thonburi for funding support for the fiscal year 2025-2026.
