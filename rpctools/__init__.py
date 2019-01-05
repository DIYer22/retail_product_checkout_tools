#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

'''
Tools for Retail-Product-Checkout-Dataset(RPC)
'''
__version__ = "0.1.0"
__short_description__ = "Tools for Retail-Product-Checkout-Dataset(RPC)"
__author__ = "DIYer22, Cui Quan, "
__author_email__ = "yanglei@megvii.com; cuiquan@megvii.com"
__github_url__ = "https://github.com/DIYer22/retail_product_checkout_tools"
__support__ = "https://github.com/DIYer22/retail_product_checkout_tools/issues"

import boxx
import pycocotools.coco


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate two jsp and return md")
    parser.add_argument(
        "res-path",
        metavar="FILE",
        help="path to result json(support bbox and chech out list)",
    )
    parser.add_argument(
        "ann-path",
        metavar="FILE",
        default="~/retail_product_checkout/instances_test2017.json",
        help="path to ground truth json(support bbox and chech out list)",
    )
    parser.add_argument(
        "--mmap",
        action='store_true',
        help="Evaluate mAP50 and mmAP",
    )
    
    pass
