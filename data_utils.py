import os
import torch
import pandas as pd
from torchvision import datasets, transforms, models
from models.ResNet.resnet import resnet18, resnet50
from models.DeiT import models, models_v2
from timm import create_model
from timm.models.layers import LWTA
from tqdm import tqdm
from torch.utils.data import DataLoader

DATASET_ROOTS = {"imagenet_val": "data/imagenet/val",
                 "broden": "data/broden1_224/images/",
                 "places365_val": "data/places365/"}


def get_deit_model(model, num_classes, input_size, state_dict, device, comps):
    act_layer = LWTA
    model = create_model(model,
                         pretrained=False,
                         num_classes=num_classes,
                         img_size=input_size,
                         act_layer=act_layer,
                         competitors=comps,
                         ).to(device)
    checkpoint = torch.load(state_dict, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model


def get_resnet_custom(model, num_classes, state_dict, device, comps):
    if model == 'resnet18':
        model = resnet18(U=comps, num_classes=num_classes)
    elif model == 'resnet50':
        model = resnet50(U=comps, num_classes=num_classes)

    model = model.to(device)

    checkpoint = torch.load(state_dict, map_location='cpu')
    #print(checkpoint.keys())
    state_dict = checkpoint['state_dict']
    #print(state_dict.keys())
    for key in list(state_dict.keys()):
        if key.startswith('module.'):
            state_dict[key[7:]] = state_dict.pop(key)
        elif key.startswith('_orig_mod'):
            state_dict[key[17:]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_target_model(target_name, device, sanity_check=False):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {resnet18_places, resnet18, resnet34, resnet50, resnet101, resnet152}
                 except for resnet18_places this will return a model trained on ImageNet from torchvision
                 
    To Dissect a different model implement its loading and preprocessing function here
    """
    if target_name == 'resnet18_places':
        target_model = models.resnet18(num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    elif "vit_b" in target_name:
        target_name_cap = target_name.replace("vit_b", "ViT_B")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    # these are custom
    elif "resnet" in target_name:
        if target_name not in ['resnet-18-places-2', 'resnet-18-places-8', 'resnet-18-places-1',
                               'resnet-18-places-4']:
            raise ValueError('Model: {} not available'.format(target_name))
        model_name = ''.join(target_name.split('-')[:2])
        num_classes = 365 if 'places' in target_name else 1000
        comps = int(target_name.split('-')[-1])

        print(os.getcwd())
        ckpt_path = os.getcwd() + '/checkpoints/ablation/{}/best_checkpoint.pth'.format(target_name)
        if not os.path.exists(ckpt_path):
            raise ValueError('No checkpoint path for {} was found.'.format(target_name))
        target_model = get_resnet_custom(model_name, num_classes, ckpt_path, device, comps=comps)
        preprocess = get_resnet_imagenet_preprocess()

    elif "deit" in target_name:
        if 'relu' in target_name:
            comps = 1
        else:
            comps = int(target_name.split('-')[-1])
        name = 'tiny' if 'tiny' in target_name else 'small'
        ckpt_path = os.getcwd() + '/checkpoints/ablation/{}/best_checkpoint.pth'.format(target_name)
        target_model = get_deit_model('deit_{}_patch16_224'.format(name), 1000,
                                      224, ckpt_path, device=device, comps=comps)
        preprocess = get_resnet_imagenet_preprocess()
    else:
        raise ValueError('Wrong target model: {} is not implemented.'.format(target_name))

    if sanity_check:
        print('=> Performing sanity check for having correctly loaded the model....')
        dataset_name = 'imagenet_val' if 'deit' in target_name else 'places365_val'
        dataset = get_data(dataset_name, preprocess = preprocess)
        with torch.no_grad():
            acc = 0.
            acc5_tot = 0.
            i = 0
            for images, labels in tqdm(DataLoader(dataset, 100, num_workers=4, pin_memory=False)):
                images = images.to(device)
                labels = labels.to(device)
                output = target_model(images)
                i += 1

                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                acc += acc1
                acc5_tot += acc5
        print('=> Sanity Check Accuracy: Top-1: {}, Top-5: {}'.format(acc / i, acc5_tot / i))
        sys.exit()

    return target_model, preprocess


def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                     transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess


def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                 transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                 transform=preprocess)

    elif dataset_name == 'places365_val':
        data = datasets.Places365(root=DATASET_ROOTS['places365_val'], download=False,
                                      split='val', small=True, transform=preprocess)
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)

    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess),
                                               datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    else:
        raise ValueError('Wrong Dataset: {} is not supported.'.format(dataset_name))

    return data


def get_places_id_to_broden_label():
    with open("data/categories_places365.txt", "r") as f:
        places365_classes = f.read().split("\n")

    broden_scenes = pd.read_csv('data/broden1_224/c_scene.csv')
    id_to_broden_label = {}
    for i, cls in enumerate(places365_classes):
        name = cls[3:].split(' ')[0]
        name = name.replace('/', '-')

        found = (name + '-s' in broden_scenes['name'].values)

        if found:
            id_to_broden_label[i] = name.replace('-', '/') + '-s'
        if not found:
            id_to_broden_label[i] = None
    return id_to_broden_label


def get_cifar_superclass():
    cifar100_has_superclass = [i for i in range(7)]
    cifar100_has_superclass.extend([i for i in range(33, 69)])
    cifar100_has_superclass.append(70)
    cifar100_has_superclass.extend([i for i in range(72, 78)])
    cifar100_has_superclass.extend([101, 104, 110, 111, 113, 114])
    cifar100_has_superclass.extend([i for i in range(118, 126)])
    cifar100_has_superclass.extend([i for i in range(147, 151)])
    cifar100_has_superclass.extend([i for i in range(269, 281)])
    cifar100_has_superclass.extend([i for i in range(286, 298)])
    cifar100_has_superclass.extend([i for i in range(300, 308)])
    cifar100_has_superclass.extend([309, 314])
    cifar100_has_superclass.extend([i for i in range(321, 327)])
    cifar100_has_superclass.extend([i for i in range(330, 339)])
    cifar100_has_superclass.extend([345, 354, 355, 360, 361])
    cifar100_has_superclass.extend([i for i in range(385, 398)])
    cifar100_has_superclass.extend([409, 438, 440, 441, 455, 463, 466, 483, 487])
    cifar100_doesnt_have_superclass = [i for i in range(500) if (i not in cifar100_has_superclass)]

    return cifar100_has_superclass, cifar100_doesnt_have_superclass

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
