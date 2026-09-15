(torch) savannah@RUIJIEGONG:~/uq/comp3710$ python demo2/part3_dawnbench.py --mode train --epochs 100
/home/savannah/miniconda3/envs/torch/lib/python3.11/site-packages/torchvision/datasets/cifar.py:83: VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean but got `align=0`. Did you mean to pass a tuple to create a subarray type? (Deprecated NumPy 2.4)
  entry = pickle.load(f, encoding="latin1")

--- CIFAR-10 ---
Device: cuda
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Training images: 50000
Testing images: 10000
Batch size: 256
Mixed precision: True

--- Full Training ---
Epoch 001/100 | lr=0.01000 | train_loss=1.6306 train_acc=0.3925 | test_loss=1.4710 test_acc=0.4684 | epoch=15.67s | best=0.4684
Epoch 002/100 | lr=0.03250 | train_loss=1.2201 train_acc=0.5593 | test_loss=1.2484 test_acc=0.5796 | epoch=11.41s | best=0.5796
Epoch 003/100 | lr=0.05500 | train_loss=0.9156 train_acc=0.6802 | test_loss=1.0921 test_acc=0.6504 | epoch=11.40s | best=0.6504
Epoch 004/100 | lr=0.07750 | train_loss=0.7211 train_acc=0.7488 | test_loss=0.9405 test_acc=0.6953 | epoch=11.45s | best=0.6953
Epoch 005/100 | lr=0.10000 | train_loss=0.6149 train_acc=0.7880 | test_loss=1.0278 test_acc=0.6925 | epoch=11.48s | best=0.6953
Epoch 006/100 | lr=0.10000 | train_loss=0.5224 train_acc=0.8202 | test_loss=0.6856 test_acc=0.7796 | epoch=11.59s | best=0.7796
Epoch 007/100 | lr=0.09997 | train_loss=0.4584 train_acc=0.8429 | test_loss=0.6301 test_acc=0.7861 | epoch=11.52s | best=0.7861
Epoch 008/100 | lr=0.09989 | train_loss=0.4166 train_acc=0.8567 | test_loss=0.5740 test_acc=0.8111 | epoch=11.52s | best=0.8111
Epoch 009/100 | lr=0.09975 | train_loss=0.3847 train_acc=0.8670 | test_loss=0.7152 test_acc=0.7709 | epoch=11.46s | best=0.8111
Epoch 010/100 | lr=0.09956 | train_loss=0.3582 train_acc=0.8773 | test_loss=0.5529 test_acc=0.8140 | epoch=11.56s | best=0.8140
Epoch 011/100 | lr=0.09932 | train_loss=0.3394 train_acc=0.8835 | test_loss=0.5443 test_acc=0.8194 | epoch=11.48s | best=0.8194
Epoch 012/100 | lr=0.09902 | train_loss=0.3230 train_acc=0.8875 | test_loss=0.5324 test_acc=0.8199 | epoch=11.40s | best=0.8199
Epoch 013/100 | lr=0.09867 | train_loss=0.3076 train_acc=0.8940 | test_loss=0.4438 test_acc=0.8511 | epoch=11.32s | best=0.8511
Epoch 014/100 | lr=0.09826 | train_loss=0.2939 train_acc=0.8985 | test_loss=0.5064 test_acc=0.8342 | epoch=11.51s | best=0.8511
Epoch 015/100 | lr=0.09780 | train_loss=0.2828 train_acc=0.9031 | test_loss=0.6497 test_acc=0.8029 | epoch=11.44s | best=0.8511
Epoch 016/100 | lr=0.09729 | train_loss=0.2747 train_acc=0.9052 | test_loss=0.4552 test_acc=0.8494 | epoch=11.58s | best=0.8511
Epoch 017/100 | lr=0.09673 | train_loss=0.2703 train_acc=0.9064 | test_loss=0.4446 test_acc=0.8560 | epoch=11.53s | best=0.8560
Epoch 018/100 | lr=0.09612 | train_loss=0.2615 train_acc=0.9110 | test_loss=0.5496 test_acc=0.8262 | epoch=11.20s | best=0.8560
Epoch 019/100 | lr=0.09545 | train_loss=0.2546 train_acc=0.9136 | test_loss=0.5816 test_acc=0.8151 | epoch=11.63s | best=0.8560
Epoch 020/100 | lr=0.09474 | train_loss=0.2537 train_acc=0.9127 | test_loss=0.5093 test_acc=0.8364 | epoch=11.38s | best=0.8560
Epoch 021/100 | lr=0.09398 | train_loss=0.2431 train_acc=0.9162 | test_loss=0.5974 test_acc=0.8123 | epoch=11.25s | best=0.8560
Epoch 022/100 | lr=0.09317 | train_loss=0.2410 train_acc=0.9159 | test_loss=0.5859 test_acc=0.8173 | epoch=11.44s | best=0.8560
Epoch 023/100 | lr=0.09231 | train_loss=0.2405 train_acc=0.9170 | test_loss=0.5509 test_acc=0.8245 | epoch=11.25s | best=0.8560
Epoch 024/100 | lr=0.09141 | train_loss=0.2309 train_acc=0.9203 | test_loss=0.3918 test_acc=0.8709 | epoch=11.52s | best=0.8709
Epoch 025/100 | lr=0.09046 | train_loss=0.2294 train_acc=0.9208 | test_loss=0.4717 test_acc=0.8456 | epoch=11.24s | best=0.8709
Epoch 026/100 | lr=0.08947 | train_loss=0.2166 train_acc=0.9251 | test_loss=0.4470 test_acc=0.8596 | epoch=11.30s | best=0.8709
Epoch 027/100 | lr=0.08843 | train_loss=0.2179 train_acc=0.9254 | test_loss=0.4207 test_acc=0.8637 | epoch=11.46s | best=0.8709
Epoch 028/100 | lr=0.08735 | train_loss=0.2103 train_acc=0.9284 | test_loss=0.4483 test_acc=0.8603 | epoch=10.95s | best=0.8709
Epoch 029/100 | lr=0.08624 | train_loss=0.2053 train_acc=0.9293 | test_loss=0.5008 test_acc=0.8459 | epoch=11.48s | best=0.8709
Epoch 030/100 | lr=0.08508 | train_loss=0.2034 train_acc=0.9297 | test_loss=0.4625 test_acc=0.8594 | epoch=11.49s | best=0.8709
Epoch 031/100 | lr=0.08388 | train_loss=0.2000 train_acc=0.9314 | test_loss=0.4940 test_acc=0.8503 | epoch=11.36s | best=0.8709
Epoch 032/100 | lr=0.08265 | train_loss=0.1978 train_acc=0.9318 | test_loss=0.3412 test_acc=0.8887 | epoch=11.62s | best=0.8887
Epoch 033/100 | lr=0.08138 | train_loss=0.1867 train_acc=0.9351 | test_loss=0.4042 test_acc=0.8744 | epoch=11.40s | best=0.8887
Epoch 034/100 | lr=0.08007 | train_loss=0.1875 train_acc=0.9348 | test_loss=0.4672 test_acc=0.8535 | epoch=11.39s | best=0.8887
Epoch 035/100 | lr=0.07874 | train_loss=0.1800 train_acc=0.9393 | test_loss=0.4460 test_acc=0.8592 | epoch=11.49s | best=0.8887
Epoch 036/100 | lr=0.07737 | train_loss=0.1787 train_acc=0.9392 | test_loss=0.4359 test_acc=0.8647 | epoch=11.36s | best=0.8887
Epoch 037/100 | lr=0.07597 | train_loss=0.1721 train_acc=0.9407 | test_loss=0.4593 test_acc=0.8581 | epoch=11.42s | best=0.8887
Epoch 038/100 | lr=0.07455 | train_loss=0.1717 train_acc=0.9413 | test_loss=0.3711 test_acc=0.8788 | epoch=11.45s | best=0.8887
Epoch 039/100 | lr=0.07309 | train_loss=0.1651 train_acc=0.9428 | test_loss=0.3822 test_acc=0.8791 | epoch=11.29s | best=0.8887
Epoch 040/100 | lr=0.07162 | train_loss=0.1622 train_acc=0.9453 | test_loss=0.3839 test_acc=0.8777 | epoch=11.38s | best=0.8887
Epoch 041/100 | lr=0.07011 | train_loss=0.1603 train_acc=0.9444 | test_loss=0.3340 test_acc=0.8957 | epoch=11.34s | best=0.8957
Epoch 042/100 | lr=0.06859 | train_loss=0.1525 train_acc=0.9479 | test_loss=0.3938 test_acc=0.8758 | epoch=11.25s | best=0.8957
Epoch 043/100 | lr=0.06705 | train_loss=0.1481 train_acc=0.9501 | test_loss=0.3706 test_acc=0.8852 | epoch=11.59s | best=0.8957
Epoch 044/100 | lr=0.06549 | train_loss=0.1468 train_acc=0.9496 | test_loss=0.3562 test_acc=0.8907 | epoch=11.25s | best=0.8957
Epoch 045/100 | lr=0.06391 | train_loss=0.1429 train_acc=0.9510 | test_loss=0.4263 test_acc=0.8724 | epoch=11.42s | best=0.8957
Epoch 046/100 | lr=0.06231 | train_loss=0.1388 train_acc=0.9521 | test_loss=0.4108 test_acc=0.8778 | epoch=11.38s | best=0.8957
Epoch 047/100 | lr=0.06070 | train_loss=0.1344 train_acc=0.9547 | test_loss=0.3963 test_acc=0.8724 | epoch=11.42s | best=0.8957
Epoch 048/100 | lr=0.05908 | train_loss=0.1315 train_acc=0.9550 | test_loss=0.3488 test_acc=0.8898 | epoch=11.63s | best=0.8957
Epoch 049/100 | lr=0.05746 | train_loss=0.1192 train_acc=0.9595 | test_loss=0.3353 test_acc=0.8992 | epoch=11.36s | best=0.8992
Epoch 050/100 | lr=0.05582 | train_loss=0.1263 train_acc=0.9569 | test_loss=0.5969 test_acc=0.8433 | epoch=11.37s | best=0.8992
>>> Reached 90% accuracy at epoch 51 after 626.84s
Epoch 051/100 | lr=0.05417 | train_loss=0.1207 train_acc=0.9585 | test_loss=0.3212 test_acc=0.9004 | epoch=11.58s | best=0.9004
Epoch 052/100 | lr=0.05253 | train_loss=0.1091 train_acc=0.9627 | test_loss=0.3420 test_acc=0.8973 | epoch=11.39s | best=0.9004
Epoch 053/100 | lr=0.05088 | train_loss=0.1075 train_acc=0.9630 | test_loss=0.4034 test_acc=0.8824 | epoch=11.40s | best=0.9004
Epoch 054/100 | lr=0.04922 | train_loss=0.1042 train_acc=0.9647 | test_loss=0.3461 test_acc=0.8952 | epoch=11.51s | best=0.9004
Epoch 055/100 | lr=0.04757 | train_loss=0.0999 train_acc=0.9655 | test_loss=0.3809 test_acc=0.8860 | epoch=11.24s | best=0.9004
Epoch 056/100 | lr=0.04593 | train_loss=0.0930 train_acc=0.9686 | test_loss=0.3248 test_acc=0.9044 | epoch=11.06s | best=0.9044
Epoch 057/100 | lr=0.04428 | train_loss=0.0942 train_acc=0.9671 | test_loss=0.3345 test_acc=0.8993 | epoch=11.21s | best=0.9044
Epoch 058/100 | lr=0.04264 | train_loss=0.0891 train_acc=0.9698 | test_loss=0.3734 test_acc=0.8929 | epoch=11.38s | best=0.9044
Epoch 059/100 | lr=0.04102 | train_loss=0.0875 train_acc=0.9704 | test_loss=0.3222 test_acc=0.9041 | epoch=11.47s | best=0.9044
Epoch 060/100 | lr=0.03940 | train_loss=0.0733 train_acc=0.9749 | test_loss=0.3674 test_acc=0.8953 | epoch=11.30s | best=0.9044
Epoch 061/100 | lr=0.03779 | train_loss=0.0719 train_acc=0.9757 | test_loss=0.3569 test_acc=0.8982 | epoch=11.40s | best=0.9044
Epoch 062/100 | lr=0.03619 | train_loss=0.0695 train_acc=0.9771 | test_loss=0.3307 test_acc=0.9078 | epoch=11.43s | best=0.9078
Epoch 063/100 | lr=0.03461 | train_loss=0.0669 train_acc=0.9772 | test_loss=0.3298 test_acc=0.9082 | epoch=11.31s | best=0.9082
Epoch 064/100 | lr=0.03305 | train_loss=0.0627 train_acc=0.9786 | test_loss=0.2919 test_acc=0.9195 | epoch=11.42s | best=0.9195
Epoch 065/100 | lr=0.03151 | train_loss=0.0567 train_acc=0.9801 | test_loss=0.3336 test_acc=0.9062 | epoch=11.37s | best=0.9195
Epoch 066/100 | lr=0.02999 | train_loss=0.0535 train_acc=0.9821 | test_loss=0.2680 test_acc=0.9237 | epoch=11.40s | best=0.9237
Epoch 067/100 | lr=0.02848 | train_loss=0.0484 train_acc=0.9839 | test_loss=0.3112 test_acc=0.9145 | epoch=11.35s | best=0.9237
Epoch 068/100 | lr=0.02701 | train_loss=0.0456 train_acc=0.9852 | test_loss=0.2983 test_acc=0.9209 | epoch=11.47s | best=0.9237
Epoch 069/100 | lr=0.02555 | train_loss=0.0418 train_acc=0.9864 | test_loss=0.2565 test_acc=0.9278 | epoch=11.30s | best=0.9278
Epoch 070/100 | lr=0.02413 | train_loss=0.0320 train_acc=0.9895 | test_loss=0.2618 test_acc=0.9315 | epoch=11.39s | best=0.9315
Epoch 071/100 | lr=0.02273 | train_loss=0.0332 train_acc=0.9893 | test_loss=0.2842 test_acc=0.9199 | epoch=11.37s | best=0.9315
Epoch 072/100 | lr=0.02136 | train_loss=0.0303 train_acc=0.9904 | test_loss=0.2697 test_acc=0.9278 | epoch=11.55s | best=0.9315
Epoch 073/100 | lr=0.02003 | train_loss=0.0244 train_acc=0.9924 | test_loss=0.2528 test_acc=0.9327 | epoch=11.53s | best=0.9327
Epoch 074/100 | lr=0.01872 | train_loss=0.0210 train_acc=0.9935 | test_loss=0.3044 test_acc=0.9220 | epoch=11.28s | best=0.9327
Epoch 075/100 | lr=0.01745 | train_loss=0.0193 train_acc=0.9940 | test_loss=0.2617 test_acc=0.9346 | epoch=11.55s | best=0.9346
Epoch 076/100 | lr=0.01622 | train_loss=0.0148 train_acc=0.9958 | test_loss=0.2466 test_acc=0.9355 | epoch=11.36s | best=0.9355
>>> Reached 94% accuracy at epoch 77 after 943.12s
Epoch 077/100 | lr=0.01502 | train_loss=0.0124 train_acc=0.9967 | test_loss=0.2352 test_acc=0.9401 | epoch=11.38s | best=0.9401
Epoch 078/100 | lr=0.01386 | train_loss=0.0102 train_acc=0.9971 | test_loss=0.2380 test_acc=0.9398 | epoch=11.23s | best=0.9401
Epoch 079/100 | lr=0.01275 | train_loss=0.0082 train_acc=0.9981 | test_loss=0.2390 test_acc=0.9409 | epoch=11.54s | best=0.9409
Epoch 080/100 | lr=0.01167 | train_loss=0.0065 train_acc=0.9982 | test_loss=0.2379 test_acc=0.9407 | epoch=11.05s | best=0.9409
Epoch 081/100 | lr=0.01063 | train_loss=0.0060 train_acc=0.9986 | test_loss=0.2306 test_acc=0.9429 | epoch=11.35s | best=0.9429
Epoch 082/100 | lr=0.00964 | train_loss=0.0048 train_acc=0.9991 | test_loss=0.2216 test_acc=0.9439 | epoch=11.37s | best=0.9439
Epoch 083/100 | lr=0.00869 | train_loss=0.0040 train_acc=0.9993 | test_loss=0.2224 test_acc=0.9439 | epoch=11.40s | best=0.9439
Epoch 084/100 | lr=0.00779 | train_loss=0.0032 train_acc=0.9996 | test_loss=0.2228 test_acc=0.9434 | epoch=11.46s | best=0.9439
Epoch 085/100 | lr=0.00693 | train_loss=0.0028 train_acc=0.9996 | test_loss=0.2199 test_acc=0.9450 | epoch=11.52s | best=0.9450
Epoch 086/100 | lr=0.00612 | train_loss=0.0028 train_acc=0.9997 | test_loss=0.2133 test_acc=0.9467 | epoch=11.32s | best=0.9467
Epoch 087/100 | lr=0.00536 | train_loss=0.0022 train_acc=0.9998 | test_loss=0.2123 test_acc=0.9467 | epoch=11.41s | best=0.9467
Epoch 088/100 | lr=0.00465 | train_loss=0.0019 train_acc=0.9999 | test_loss=0.2120 test_acc=0.9467 | epoch=11.53s | best=0.9467
Epoch 089/100 | lr=0.00398 | train_loss=0.0019 train_acc=0.9999 | test_loss=0.2111 test_acc=0.9475 | epoch=11.53s | best=0.9475
Epoch 090/100 | lr=0.00337 | train_loss=0.0020 train_acc=0.9999 | test_loss=0.2085 test_acc=0.9472 | epoch=11.41s | best=0.9475
Epoch 091/100 | lr=0.00281 | train_loss=0.0018 train_acc=0.9999 | test_loss=0.2080 test_acc=0.9475 | epoch=11.52s | best=0.9475
Epoch 092/100 | lr=0.00230 | train_loss=0.0018 train_acc=1.0000 | test_loss=0.2063 test_acc=0.9480 | epoch=11.37s | best=0.9480
Epoch 093/100 | lr=0.00184 | train_loss=0.0017 train_acc=1.0000 | test_loss=0.2052 test_acc=0.9483 | epoch=11.55s | best=0.9483
Epoch 094/100 | lr=0.00143 | train_loss=0.0019 train_acc=0.9998 | test_loss=0.2075 test_acc=0.9480 | epoch=11.49s | best=0.9483
Epoch 095/100 | lr=0.00108 | train_loss=0.0017 train_acc=0.9999 | test_loss=0.2060 test_acc=0.9483 | epoch=11.47s | best=0.9483
Epoch 096/100 | lr=0.00078 | train_loss=0.0019 train_acc=0.9999 | test_loss=0.2051 test_acc=0.9487 | epoch=11.55s | best=0.9487
Epoch 097/100 | lr=0.00054 | train_loss=0.0017 train_acc=0.9999 | test_loss=0.2039 test_acc=0.9484 | epoch=11.38s | best=0.9487
Epoch 098/100 | lr=0.00035 | train_loss=0.0017 train_acc=0.9999 | test_loss=0.2057 test_acc=0.9482 | epoch=11.49s | best=0.9487
Epoch 099/100 | lr=0.00021 | train_loss=0.0016 train_acc=0.9999 | test_loss=0.2046 test_acc=0.9482 | epoch=11.55s | best=0.9487
Epoch 100/100 | lr=0.00013 | train_loss=0.0017 train_acc=0.9999 | test_loss=0.2058 test_acc=0.9480 | epoch=11.44s | best=0.9487

--- Final Result ---
Best test accuracy: 0.9487
Training compute time: 1145.41s
Training-loop wall time: 1224.24s
Time to 90% accuracy: 626.84s
Time to 94% accuracy: 943.12s
Best checkpoint: /home/savannah/uq/comp3710/demo2/resnet18_cifar10_best.pt
(torch) savannah@RUIJIEGONG:~/uq/comp3710$ 