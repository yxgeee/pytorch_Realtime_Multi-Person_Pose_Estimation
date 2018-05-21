srun -p OCR --gres=gpu:1 -n1 --ntasks-per-node=1 \
  python landmark.py /mnt/lustre/geyixiao/ECCV2018/pytorch-CycleGAN-and-pix2pix/datasets/market/raw/Market-1501-v15.09.15/query \
                      /mnt/lustre/geyixiao/ECCV2018/pytorch-CycleGAN-and-pix2pix/datasets/market/raw/land/query
