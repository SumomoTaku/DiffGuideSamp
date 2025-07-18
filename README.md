# Task-Specific Generative Dataset Distillation with Difficulty-Guided Sampling
The code used in the following paper:  
[Task-Specific Generative Dataset Distillation with Difficulty-Guided Sampling](https://arxiv.org/abs/2507.03331)


## How to Use
This method samples the final distilled dataset from a larger image pool that is obtained by SOTA generative dataset distillation methods, guided by the concept of DIFFICULTY, which is defined to be the opposite of classification probability.

1. Set the virtual environment with conda
```
git clone https://github.com/SumomoTaku/DiffGuideSamp.git
cd DiffGuideSamp
conda env create -f environment.yml
conda activate DiffGuide
```
2. Get the image pool (ip).  
   This project doesn't include the code for creating the image pool.  
   You can refer to the pages of other SOTA methods, like [Minimax](https://github.com/vimar-gu/MinimaxDiffusion).  
   The size of the image pool is recommended to $5 \times IPC$ following the experiments in the article.  

3. Sampling.  
  You can find the basic implementation in the ./scripts/sample.sh  
  You need to set the path of the image pool, the original dataset, and the output. As well as the IPC (the size of the distilled dataset).  
```
cd scripts
sh sample.sh
```
5. Train the downstream model.  
   This project doesn't include the code for training the downstream model.  
   You can refer to the code of other SOTA methods by putting the distilled dataset under their output path.  
   
   

