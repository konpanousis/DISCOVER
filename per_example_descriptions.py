import pandas as pd
import torch
import data_utils
import numpy as np
import os
import matplotlib.pyplot as plt

target_models = ['deit-tiny-8',
                 #'deit-tiny-relu', 'deit-tiny-16', 'deit-small-2', 'deit-small-12',
                 #'deit-tiny-24', 'resnet-18-places-8', 'resnet-18-places-1', 'resnet-18-places-4'
                 ]
examples = list(np.random.choice(np.arange(50000), size=10, replace=False)) + [13951]
print(examples)
modes = ['class']
dprobe = 'imagenet_val'
cs = 'imagenet'

# examples = [9534]
with open('chosen_examples.txt', 'w') as f:
    for ex in examples:
        f.write(str(ex) + ', ')

block_num = 11
sims = ['soft_wpmi']#, 'wpmi', 'cos_similarity', 'rank_reorder', 'cos_similarity_cubed']
layer_res = 'layer4'

for example in examples:
    base_path = 'concepts_per_example/probe_{}_cs_{}/example_{}'.format(dprobe, cs, example)
    os.makedirs(base_path, exist_ok=True)

    pil_data = data_utils.get_data(dprobe)
    im, label = pil_data[example]
    im = im.resize([375, 375])
    plt.imshow(im)
    plt.axis('off')
    plt.savefig(base_path + '/{}.pdf'.format(example), bbox_inches='tight')

    for model in target_models:
        deit_flag = 'deit' in model
        base_experiment_path = 'experiments/{}_FINAL/{}'.format(model, dprobe)
        act_path = '{}/saved_activations/{}_{}_{}.pt'.format(base_experiment_path, dprobe, model,
                                                             'blocks[{}].mlp.act'.format(block_num)
                                                             if deit_flag else layer_res)

        acts = torch.load(act_path).cpu().numpy()

        for sim in sims:
            print('Example: {}, Model: {}, Concept Set: {}, Sim: {}'.format(example, model, cs, sim))
            cur_path = '{}/{}_{}/{}'.format(base_path, model, 'blocks[{}].mlp.act'.format(block_num)
                                                             if deit_flag else layer_res,
                                            sim)
            os.makedirs(cur_path, exist_ok=True)

            results_path = '{}/results/{}/{}'.format(base_experiment_path, cs, sim)
            column_names = ['Layer', 'Neuron', 'Description', 'Similarity']
            descriptions = pd.read_csv('{}/descriptions.csv'.format(results_path), names=column_names)
            filter = descriptions.Layer == 'blocks[{}].mlp.act'.format(block_num) if deit_flag else layer_res
            description_layer = descriptions[filter]
            concs, scores = description_layer['Description'].values, description_layer['Similarity'].values

            with open(cur_path + '/number_of_activated', 'w') as f:
                if 'relu' in model:
                    f.write(str((np.abs(acts) > 1e-4).mean()))
                else:
                    U = int(model.split('-')[-1])
                    acts_re = acts.reshape(-1, acts.shape[-1] // U, U).argmax(-1)
                    f.write(str(acts_re.shape))

            concepts = []
            print(model)
            print('relu' in model)
            if 'relu' in model:
                print(acts.shape)
                print(concs.shape)
                inds = np.where(np.abs(acts[example]) > 1e-4)[0]
                for ind in inds:
                    concepts.append((concs[ind], acts[example][ind]))
            else:
                U = int(model.split('-')[-1])
                acts_re = acts[example].reshape(acts.shape[1] // U, U)
                inds = acts_re.argmax(-1)

                concs_re = concs.reshape(concs.shape[-1] // U, U)

                for i, ind in enumerate(inds):
                    concepts.append((concs_re[i][ind], acts_re[i][ind]))

            concepts = sorted(concepts, key=lambda x: x[1], reverse=True)
            with open(cur_path + '/image_level_concepts.txt', 'w') as f:
                for concept in concepts:
                    f.write('{} !! {:.6f}\n'.format(concept[0], concept[1]))
            # np.savetxt(example_path + 'image_level_concepts.txt', concepts, fmt='%s', delimiter='\n')
