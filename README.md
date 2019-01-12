# Retail Product Checkout Dataset Tools

Project : [RPC Dataset Project Page](http://rpc-dataset.github.io)

Dataset : [RPC Dataset](https://www.kaggle.com/diyer22/retail-product-checkout-dataset)

## 1. Install

```
pip install rpctool
```
or
```
pip install git+https://github.com/DIYer22/retail_product_checkout_tools
```
## 2. Usage


```
python -m rpctool {result json} {ground truth json}
```
## 3. Help
```
$ python -m rpctool -h

Evaluate resFile with annFile and return evaluation result in markdown format

positional arguments:
  FILE        path to result json(support bbox and check out list)
  FILE        path to ground truth json

optional arguments:
  -h, --help  show this help message and exit
  --mmap      Evaluate mAP50 and mmAP
```

## 4. example

Input:   
```
python -m rpctool bbox_results.json ~/retail_product_checkout/instances_test2019.json 
```

Return:   
```
## result on RPC-Dataset
|     diff |  method |   cAcc |  mCIoU |  ACD | mCCD |
|     ---: |    ---: |   ---: |   ---: | ---: | ---: |
|     easy | default | 63.19% | 90.64% | 0.72 | 0.11 |
|   medium | default | 43.02% | 90.64% | 1.24 | 0.11 |
|     hard | default | 31.01% | 90.41% | 1.77 |  0.1 |
| averaged | default |  45.6% | 90.58% | 1.25 |  0.1 |
```

