
## Setup
```
pytorch=1.2.0
torchvision=0.4.0
```

## Quick start

For CIFAR10 dataset, you can reproduce the results in the paper by running
```
python fedavg.py --dataset=cifar --num_users=100 --iid=1 --gpu=0 --frac=0.1 --model=cnn --epoch=500 --mal_boost=5 --local_mal_ep=10 --pert_strength=0.4 --num_mal_samples=1 --defense=WBC
```

##Credits

This project is based on code by Sun et al. the authors of FL-WBC: Enhancing Robustness against Model Poisoning Attacks in Federated Learning from a Client Perspective
 and is available here: https://github.com/jeremy313/FL-WBC
