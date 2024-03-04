import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
import torch.distributed as dist

from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler, CustomDistributedSamplerWrapper
from .transforms import build_transform

INTERPOLATION_MODES = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "nearest": Image.NEAREST,
}


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins,
    )

    if is_train:
        sampler = CustomDistributedSamplerWrapper(data_source, sampler, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    other = {}
    if dataset_wrapper is None and is_train:
        dataset_wrapper = DatasetWrapper
    elif dataset_wrapper is None and not is_train and not cfg.eval_only:
        dataset_wrapper = DatasetWrapper
    elif dataset_wrapper is None and not is_train:
        dataset_wrapper = DatasetWrapperWithBlock
        other = {'multi_scale': cfg.TEST.multi_scale}

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train, **other),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        # if cfg.TRAINER.NAME == 'Caption_distill':
        if 'distill' in cfg.DATASET.NAME:
            dataset_wrapper_ = DatasetWrapper_caption
        else:
            dataset_wrapper_ = None
        if dataset.train_x is not None:
            train_loader_x = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=dataset.train_x,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper_,
            )
        else:
            train_loader_x = None

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper,
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper,
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        print("***** Dataset statistics *****")

        print("  Dataset: {}".format(cfg.DATASET.NAME))

        if cfg.DATASET.SOURCE_DOMAINS:
            print("  Source domains: {}".format(cfg.DATASET.SOURCE_DOMAINS))
        if cfg.DATASET.TARGET_DOMAINS:
            print("  Target domains: {}".format(cfg.DATASET.TARGET_DOMAINS))

        print("  # classes: {:,}".format(self.num_classes))

        if self.dataset.train_x:
            print("  # train_x: {:,}".format(len(self.dataset.train_x)))

        if self.dataset.train_u:
            print("  # train_u: {:,}".format(len(self.dataset.train_u)))

        if self.dataset.val:
            print("  # val: {:,}".format(len(self.dataset.val)))

        print("  # test: {:,}".format(len(self.dataset.test)))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


class DatasetWrapper_caption(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        # self.transform = transform  # accept list (tuple) as input
        # self.is_train = is_train
        # # Augmenting an image K>1 times is only allowed during training
        # self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        # self.return_img0 = cfg.DATALOADER.RETURN_IMG0

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": torch.tensor(item[1]),
            "img": torch.tensor(item[0])  # caption_tokenized
        }
        return output

class DatasetWrapperWithBlock(DatasetWrapper):  
  
    def __init__(self, cfg, data_source, transform=None, is_train=False, multi_scale=[2,3,4,5]):
        self.multi_scale = multi_scale
        super(DatasetWrapperWithBlock, self).__init__(cfg, data_source, transform, is_train)
  
    def __getitem__(self, idx):  
        item = self.data_source[idx]  
  
        output = {  
            "label": item.label,  
            "domain": item.domain,  
            "impath": item.impath  
        }  
  
        img0 = read_image(item.impath)  
  
        if self.transform is not None:  
            if isinstance(self.transform, (list, tuple)):  
                for i, tfm in enumerate(self.transform):  
                    img, img_blocks = self._transform_image(tfm, img0)  
                    keyname = "img"  
                    if (i + 1) > 1:  
                        keyname += str(i + 1)  
                    output[keyname] = img  
                    output[keyname + "_blocks"] = img_blocks  
            else:  
                img, img_blocks = self._transform_image(self.transform, img0)  
                output["img"] = img
                if len(img_blocks) > 0:  
                    output["img_blocks"] = img_blocks  
  
        if self.return_img0:  
            output["img0"] = self.to_tensor(img0)  
  
        return output  
  
    def _transform_image(self, tfm, img0):  
        img_list = []  
  
        for k in range(self.k_tfm):  
            img_list.append(tfm(img0))  
  
        img = img_list  
        if len(img) == 1:  
            img = img[0]  

        img_blocks = []
        for block_size in self.multi_scale:
            # Add padding to make the image dimensions divisible by block_size  
#             w, h = img0.size  
#             img0_padded = F.pad(img0, (0, -w % block_size, 0, -h % block_size), padding_mode='reflect')  
#             h, w = img0_padded.size[::-1]  
#             block_h, block_w = h // block_size, w // block_size

#             # Convert img0_padded to tensor  
#             img0_padded_tensor = F.to_tensor(img0_padded)  

#             # Create block_sizexblock_size image blocks  
#             img_block = []  
#             for i in range(block_size):  
#                 for j in range(block_size):  
#                     block = img0_padded_tensor[:, i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]  
#                     # Convert block tensor back to PIL image  
#                     block = F.to_pil_image(block)  
#                     block_transformed = tfm(block)  
#                     img_block.append(block_transformed)  
    
#             img_block = torch.stack([block.contiguous() for block in img_block])  
#             img_blocks.append(img_block)
            

            ## Add sliding window
            w, h = img0.size  
            slide_num = block_size*2
            block_h, block_w = h // block_size, w // block_size
            stride_h, stride_w = ((block_size - 1) * block_h) // (slide_num - 1) + 1, ((block_size - 1) * block_w) // (slide_num - 1) + 1
            padding_h ,padding_w = stride_h * (slide_num - 1) - ((block_size - 1) * block_h) - h % block_size, stride_w * (slide_num - 1) - ((block_size - 1) * block_w) - w % block_size
            img0_tensor = F.to_tensor(img0)
            img0_padded = F.pad(img0_tensor, (0, padding_w, 0, padding_h), padding_mode='reflect')
            img_block = []  
            for i in range(slide_num):  
                for j in range(slide_num):  
                    block = img0_padded[:, i * stride_h:i * stride_h + block_h, j * stride_w:j * stride_w + block_w]  
                    # Convert block tensor back to PIL image  
                    block = F.to_pil_image(block)  
                    block_transformed = tfm(block)  
                    img_block.append(block_transformed)
                    
            ## 1*2, 2*1 sliding window
            block_h_w_list = [(h // block_size, w * 2 // block_size), (h * 2 // block_size, w // block_size)]
            slide_num_h_w_list = [(block_size * 2, block_size), (block_size, block_size * 2)]

            for block_hw, slide_num_hw in zip(block_h_w_list, slide_num_h_w_list):
                # block_h, block_w = h // block_size, w * 2 // block_size  # Adjust block_h and block_w here  
                
                # # Select suitable slide_num_h and slide_num_w based on the size of block_h and block_w  
                # slide_num_h, slide_num_w = block_size * 2, block_size  
                block_h, block_w = block_hw
                slide_num_h, slide_num_w = slide_num_hw
                stride_h, stride_w = ((block_size - 1) * block_h) // (slide_num_h - 1) + 1, ((block_size - 1) * block_w) // (slide_num_w - 1) + 1  
                
                img0_tensor = F.to_tensor(img0)  
                img0_padded = img0_tensor
                
                for i in range(slide_num_h):  
                    for j in range(slide_num_w):  
                        # Adjust block size when the window extends outside the image  
                        current_block_h = min(block_h, h - i * stride_h)  
                        current_block_w = min(block_w, w - j * stride_w)
                        if current_block_h <= 0 or current_block_w <= 0:
                            continue
                        
                        block = img0_padded[:, i * stride_h:i * stride_h + current_block_h, j * stride_w:j * stride_w + current_block_w]  
                        # Convert block tensor back to PIL image  
                        block = F.to_pil_image(block)  
                        block_transformed = tfm(block)  
                        img_block.append(block_transformed)

            block_h_w_list = [(h // block_size, w * 3 // (2 * block_size)), (h * 3 // (2 * block_size), w // block_size)]
            slide_num_h_w_list = [(block_size * 2 // 1, block_size * 2 * 2 // 3), (block_size * 2 * 2 // 3, block_size * 2 // 1)]

            for block_hw, slide_num_hw in zip(block_h_w_list, slide_num_h_w_list):
                # block_h, block_w = h // block_size, w * 2 // block_size  # Adjust block_h and block_w here  
                
                # # Select suitable slide_num_h and slide_num_w based on the size of block_h and block_w  
                # slide_num_h, slide_num_w = block_size * 2, block_size  
                block_h, block_w = block_hw
                slide_num_h, slide_num_w = slide_num_hw
                stride_h, stride_w = ((block_size - 1) * block_h) // (slide_num_h - 1) + 1, ((block_size - 1) * block_w) // (slide_num_w - 1) + 1  
                
                img0_tensor = F.to_tensor(img0)  
                img0_padded = img0_tensor
                
                for i in range(slide_num_h):  
                    for j in range(slide_num_w):  
                        # Adjust block size when the window extends outside the image  
                        current_block_h = min(block_h, h - i * stride_h)  
                        current_block_w = min(block_w, w - j * stride_w)
                        if current_block_h <= 0 or current_block_w <= 0:
                            continue
                        
                        block = img0_padded[:, i * stride_h:i * stride_h + current_block_h, j * stride_w:j * stride_w + current_block_w]  
                        # Convert block tensor back to PIL image  
                        block = F.to_pil_image(block)  
                        block_transformed = tfm(block)  
                        img_block.append(block_transformed)
            
            if block_size >= 3:
                block_h_w_list = [(h * 2 // block_size, w * 3 // (block_size)), (h * 3 // (block_size), w * 2 // block_size)]
                slide_num_h_w_list = [(block_size * 2 // 2, block_size * 2 // 3), (block_size * 2 // 3, block_size * 2 // 2)]

                for block_hw, slide_num_hw in zip(block_h_w_list, slide_num_h_w_list):
                    # block_h, block_w = h // block_size, w * 2 // block_size  # Adjust block_h and block_w here  
                    
                    # # Select suitable slide_num_h and slide_num_w based on the size of block_h and block_w  
                    # slide_num_h, slide_num_w = block_size * 2, block_size  
                    block_h, block_w = block_hw
                    slide_num_h, slide_num_w = slide_num_hw
                    stride_h, stride_w = ((block_size - 1) * block_h) // (slide_num_h - 1) + 1, ((block_size - 1) * block_w) // (slide_num_w - 1) + 1  
                    
                    img0_tensor = F.to_tensor(img0)  
                    img0_padded = img0_tensor
                    
                    for i in range(slide_num_h):  
                        for j in range(slide_num_w):  
                            # Adjust block size when the window extends outside the image  
                            current_block_h = min(block_h, h - i * stride_h)  
                            current_block_w = min(block_w, w - j * stride_w)
                            if current_block_h <= 0 or current_block_w <= 0:
                                continue
                            
                            block = img0_padded[:, i * stride_h:i * stride_h + current_block_h, j * stride_w:j * stride_w + current_block_w]  
                            # Convert block tensor back to PIL image  
                            block = F.to_pil_image(block)  
                            block_transformed = tfm(block)  
                            img_block.append(block_transformed)                
    
            img_block = torch.stack([block.contiguous() for block in img_block])  
            img_blocks.append(img_block)

        return img, img_blocks  
