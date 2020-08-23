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

# # 勉強会用 DEAPを用いた特徴量生成
# 参考ドキュメント：解説
# * https://qiita.com/overlap/items/e7f1077ef8239f454602

import random
from deap import base
from deap import creator
from deap import tools

# * 1行目は適合度を最大化することで最適化されるような適合度クラスを作成します。
# * weightsが配列になっていますが、単目的最適化ではサイズ1で良く(ただし","は必要みたいです)、多目的最適化では各目的の重みを配列にします。最小化で最適化を行う場合は(-1.0,)を設定します。

#creatorはbaseのクラスを継承して新たなクラスを作成します。
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# * toolboxは関数を作成します。3行目のregisterではattr_boolという関数をrandom.randintに0,1の引数を与えて作成しています。attr_boolは0か1かをランダムで生成する関数になります。
# * initRepeatはToolboxにあらかじめ用意されている関数で、1つめの引数がデータを入れるコンテナの型、2つめが個々のデータを生成する関数、3つめがコンテナのサイズです。ここではindividual(個体)という関数とpopulation(集団)という関数をinitRepeatから作成しています。

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# * 次に評価関数、交差、突然変異、選択用の関数を作成します。(評価関数の返り値に","があるのにも注意してください。)

# +
def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoints)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
# -

# * こまででGAの設定が完了したので、実際の進化計算ルーチンを作成します。

# +
random.seed(64)
# 初期の個体群を生成
pop = toolbox.population(n=300)
CXPB, MUTPB, NGEN = 0.5, 0.2, 40 # 交差確率、突然変異確率、進化計算のループ回数

print("Start of evolution")

# 初期の個体群の評価
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

print("  Evaluated %i individuals" % len(pop))

# 進化計算開始
for g in range(NGEN):
    print("-- Generation %i --" % g)

    # 次世代の個体群を選択
    offspring = toolbox.select(pop, len(pop))
    # 個体群のクローンを生成
    offspring = list(map(toolbox.clone, offspring))

    # 選択した個体群に交差と突然変異を適応する
    # 偶数番目と奇数番目の個体を取り出して交差
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 適合度が計算されていない個体を集めて適合度を計算
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    # 次世代群をoffspringにする
    pop[:] = offspring

    # すべての個体の適合度を配列にする
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

print("-- End of (successful) evolution --")

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
# -


