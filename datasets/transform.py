from __future__ import absolute_import
from __future__ import division
import torchvision.transforms as transforms

train_transform = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=63 / 255),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
                                    )
test_transform = transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
                                    )

def get_transforms(image_size):
    
    if image_size == 224:
        # cifar
        cifar_train = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=63 / 255),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
                                    )
        cifar_test  = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
                                    )
        
        # miniImageNet
        mini_train = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=63 / 255),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
                                    )
        mini_test = transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
                                    )
    else:
        # cifar
        cifar_train = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])]
                                    )
        
        cifar_test = transforms.Compose([
                                        transforms.Resize([36, 36]),
                                        transforms.CenterCrop(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])]
                                    )
    
        # miniImageNet
        mini_train = transforms.Compose([
                                        transforms.RandomResizedCrop(84),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])]
                                    )
        
        mini_test = transforms.Compose([
                                        transforms.Resize([92, 92]),
                                        transforms.CenterCrop(84),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])]
                                            )
    # CUB200          
    cub_train = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                )

    cub_test = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                                )
    
    # ImageNet-R
    imr_train = transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=63 / 255),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
                                )
    imr_test = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
                                )
    
    # ImageNet
    imn_train = transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=63 / 255),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
                                )
    imn_test   = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
                                )
    
    transforms_train_inc = {
        'cifar100': cifar_train,
        'miniImageNet': mini_train,
        'ImageNet_R': imr_train,
        'ImageNet': imn_train,
        'cub_200': cub_train,
        'mnist': train_transform,
        'flowers': train_transform,
        'food101': train_transform,
        'car196': train_transform,
        'cifar10': cifar_train,
        'aircraft102': train_transform
    }

    
    transforms_test_inc = {
        'cifar100': cifar_test,
        'miniImageNet': mini_test,
        'ImageNet_R': imr_test,
        'ImageNet': imn_test,
        'cub_200': cub_test,
        'mnist': test_transform,
        'flowers': test_transform,
        'food101': test_transform,
        'car196': test_transform,
        'cifar10': cifar_test,
        'aircraft102': test_transform
    }

    return transforms_train_inc, transforms_test_inc

def image_augment(state='train', dataset='miniImageNet',image_size=224):
    """
    @state: in which stage, e.g., training stage, validation stage, or testing stage
    @dataset: currently used dataset
    """
    transforms_train_inc, transforms_test_inc = get_transforms(image_size)
    if state == 'train':
        return transforms_train_inc[dataset]
    else:
        return transforms_test_inc[dataset]

