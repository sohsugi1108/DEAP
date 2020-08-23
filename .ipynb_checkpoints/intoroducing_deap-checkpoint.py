# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 勉強会用 DEAP（Python用遺伝的アルゴリズムライブラリ）の使い方
# 参考ドキュメント
# * https://qiita.com/shiro-kuma/items/0cb8955bd85027d58c8e 

# +
import random

from deap import base
from deap import creator
from deap import tools
# -

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
