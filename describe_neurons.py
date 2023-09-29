import os
import argparse
import datetime
import json
import pandas as pd
import torch
from models.DeiT import models
import itertools
import utils
import similarity
import sys,os
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(description='DISCOVER')

parser.add_argument("--clip_model", type=str, default="ViT-B/16", 
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                   help="Which CLIP-model to use")
parser.add_argument("--target_model", type=str, default="deit_tiny-16",
                   help=""""Which model to dissect, supported options are pretrained imagenet models from
                        torchvision and resnet18_places""")
parser.add_argument("--target_layers", type=str, default="conv1,layer1,layer2,layer3,layer4",
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--d_probe", type=str, default="imagenet_val",
                    choices = ["imagenet_broden", "cifar100_val", "imagenet_val", "broden"])
parser.add_argument("--concept_set", type=str, default="20k.txt", help="Path to txt file containing concept set")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="saved_activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="results", help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="cos_similarity", choices=["soft_wpmi", "wpmi", "rank_reorder",
                                                                               "cos_similarity", "cos_similarity_cubed",
                                                                               ])

parser.parse_args()

if __name__ == '__main__':

    similarities = ["soft_wpmi"]#, "wpmi", "rank_reorder", "cos_similarity", "cos_similarity_cubed"]
    target_models = ['deit-small-12',
                     #'deit-tiny-8', 'deit-tiny-16',
                     #'deit-small-2', 'deit-small-12', 'deit-tiny-24'
                     ]
                     #'resnet-18-places-8', 'resnet-18-places-1', 'resnet-18-places-4']

    d_probes = ['imagenet_broden']#'cifar100_val', 'imagenet_val']#, 'broden', 'imagenet_broden']
    concept_sets = ['imagenet.txt' ]#'3k.txt', '10k.txt', 'imagenet.txt', '20k.txt', ]

    combs = itertools.product(*[target_models, d_probes, concept_sets, similarities])
    deit_layers = ['blocks[0].mlp.act']#'blocks[{}].mlp.act'.format(i) for i in range(0,12,11)] # blocks[{}].mlp.act'.format(i) for i in range(11,12)
    resnet_layers = ['layer1','layer2','layer3','layer4','fc']
    for tmodel, dprob, cs, sim in combs:

        if dprob in ['cifar100', 'broden', 'imagenet_broden'] and cs not in ['20k.txt', 'imagenet.txt',
                                                                             'broden_labels_clean.txt']:
            continue

        if 'resnet' in tmodel and dprob != 'broden' and cs != 'broden_labels_clean.txt':
            continue

        print(tmodel)
        print(dprob)
        print(cs)
        print(sim)

        args = parser.parse_args()
        args.target_model = tmodel
        args.d_probe = dprob
        args.concept_set = cs
        args.similarity_fn = sim

        # change the activation dir to diff models and dprobs
        folder_name = '{}_TODELETE/{}/'.format(tmodel, dprob)
        args.activation_dir = 'experiments/' + folder_name + args.activation_dir

        args.result_dir = 'experiments/' + folder_name + args.result_dir + '/{}/'.format(cs.split('.')[0])

        args.target_layers = deit_layers if 'deit' in tmodel else resnet_layers
        similarity_fn = eval("similarity.{}".format(args.similarity_fn))

        utils.save_activations(clip_name = args.clip_model, target_name = args.target_model,
                               target_layers = args.target_layers, d_probe = args.d_probe,
                               concept_set = args.concept_set, batch_size = args.batch_size,
                               device = args.device, pool_mode=args.pool_mode,
                               save_dir = args.activation_dir)

        outputs = {"layer":[], "unit":[], "description":[], "similarity":[]}
        with open('data/concept_sets/'+args.concept_set, 'r') as f:
            words = (f.read()).split('\n')

        for target_layer in args.target_layers:
            save_names = utils.get_save_names(clip_name = args.clip_model, target_name = args.target_model,
                                      target_layer = target_layer, d_probe = args.d_probe,
                                      concept_set = args.concept_set, pool_mode = args.pool_mode,
                                      save_dir = args.activation_dir)
            target_save_name, clip_save_name, text_save_name = save_names

            similarities = utils.get_similarity_from_activations(
                target_save_name, clip_save_name, text_save_name, similarity_fn,
                return_target_feats=False, device=args.device
            )
            vals, ids = torch.max(similarities, dim=1)

            del similarities
            torch.cuda.empty_cache()

            descriptions = [words[int(idx)] for idx in ids]

            outputs["unit"].extend([i for i in range(len(vals))])
            outputs["layer"].extend([target_layer]*len(vals))
            outputs["description"].extend(descriptions)
            outputs["similarity"].extend(vals.cpu().numpy())

        df = pd.DataFrame(outputs)
        stats_dir = args.result_dir + '{}/'.format(args.similarity_fn)
        os.makedirs(stats_dir, exist_ok = True)

        df.to_csv(stats_dir + "descriptions.csv", mode='a', index=False, header = False)
        with open(stats_dir + "args.txt", 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        del vals, descriptions
        torch.cuda.empty_cache()

