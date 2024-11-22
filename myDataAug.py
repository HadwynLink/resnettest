from torchvision.transforms import v2 

transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    #v2.RandomVerticalFlip(p=0.5),
    #v2.RandomResizedCrop(size=(224, 224), antialias=True),
    #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])