# RBLT-tracker :package:
:fire: Accepted by ICRA 2024~ 
+ **Joint Response and Background Learning for UAV Visual Tracking**
+ We use a subset of the UAVTrack112 dataset as our validation set for tuning all hyper-parameters.
## RBLT 

```python
git clone https://github.com/Wangbiao2/RBLT-tracker.git
```

```python
cd RBLT-tracker/RBLT
```

```python
demo_RBLT.m
```

## DeepRBLT

```
git clone https://github.com/Wangbiao2/RBLT-tracker.git
```

download  `imagenet-vgg-m-2048.mat` from `[Pretrained CNNs - MatConvNet (vlfeat.org)](https://www.vlfeat.org/matconvnet/pretrained/)`

```
cd RBLT-tracker/DeepRBLT
```

```
install.m
```

```
setup_paths.m
```

```
demo_DeepRBLT.m
```

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```matlab
% run_DeepRBLT.m
% Run tracker without coarse-to-fine scale search strategy
results = tracker_DeepRBLT(params);

% Run tracker with coarse-to-fine scale search strategy
% results = tracker_DeepRBLT_Scale(params);
```

## DeepRBLT++

+ We used ResNet-50 as the feature extraction network and made a series of changes, but this part was not presented in the paper due to its slow speed.

```
git clone https://github.com/Wangbiao2/RBLT-tracker.git
```

download  `imagenet-resnet-50-dag.mat` from `[Pretrained CNNs - MatConvNet (vlfeat.org)](https://www.vlfeat.org/matconvnet/pretrained/)`

```
cd RBLT-tracker/DeepRBLT++
```

```
install.m
```

```
setup_paths.m
```


+ Change the path setting of the 10-th line of the demo

```
demo.m
```

