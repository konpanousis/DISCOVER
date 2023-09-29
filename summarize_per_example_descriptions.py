import csv
from itertools import zip_longest


target_models = ['deit-tiny-8', 'deit-tiny-relu', 'deit-tiny-16', 'deit-small-2', 'deit-small-12',
                 'deit-tiny-24', 'resnet-18-places-8', 'resnet-18-places-1', 'resnet-18-places-4']

modes = ['class']
dprobe = 'imagenet_val'

examples = [3597, 29501, 6604, 27361, 36711, 4767, 47974, 20205, 16131, 7077, 13951,
            22797, 34378, 40658, 31172, 23083, 30764, 1928, 16031, 6370, 4088,
            39205, 40204, 29276, 24477, 40696, 1706, 39201, 22383, 13140, 10855]
block_num = 11
sims = ['soft_wpmi', 'wpmi', 'cos_similarity', 'rank_reorder', 'cos_similarity_cubed']
layer_res = 'layer4'

for sim in sims:
    base_path = 'concepts_per_example/sim_{}/'.format(sim)

    for example in examples:
        summary = []
        for model in target_models:
            deit_flag = 'deit' in model
            for mode in modes:
                cur_path = base_path + 'model_{}{}_{}_mode_{}/'.format(model,
                                                                       '_block' if deit_flag else '',
                                                                       block_num if deit_flag else layer_res,
                                                                       mode)

                neuron_act = []
                neuron_contr = []
                example_path = cur_path + 'example_{}/'.format(example)
                with open(example_path + 'image_level_concepts.txt', 'r') as f:
                    for line in f:
                        act, cont = line.split('!!')
                        neuron_act.append(act.strip())
                        neuron_contr.append(cont.strip())
                summary.append(neuron_act)
                summary.append(neuron_contr)
        rows = zip_longest(*summary)

        model_col = [val for val in target_models for _ in range(2)]
        with open(base_path + 'image_{}_summary.csv'.format(example), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(model_col)
            for row in rows:
                writer.writerow(row)
