
log_dir = 'exp'
workers = 2
seed = 3032

network = dict(
    backbone="MocHRBackbone",
    sub_arch='hrnet48',
    counter_type='withMOE',
    resolution_num=[0, 1, 2, 3],
    loss_weight=[1., 1 / 2, 1 / 4., 1 / 8.],
    sigma=[4],
    gau_kernel_size=15,
    baseline_loss=False,
    pretrained_backbone="./data/hrnetv2_w48_imagenet_pretrained.pth",

    head=dict(
        type='CountingHead',
        fuse_method='cat',
        in_channels=96,
        stages_channel=[384, 192, 96, 48],
        inter_layer=[64, 32, 16],
        out_channels=1)
)

dataset = dict(
    name='QNRF',
    root='./data/QNRF/',
    test_set='test.txt',
    train_set='train.txt',
    loc_gt='test_gt_loc.txt',
    num_classes=len(network['resolution_num']),
    den_factor=100,  # core parameter
    extra_train_set=None
)

optimizer = dict(
    NAME='adamw',
    BASE_LR=1.0e-5,  # core parameter
    BETAS=(0.9, 0.999),
    WEIGHT_DECAY=1e-4,  # core parameter
    EPS=1.0e-08,
    MOMENTUM=0.9,
    AMSGRAD=False,
    NESTEROV=True,
)

lr_config = dict(
    NAME='cosine',
    WARMUP_EPOCHS=10,
    WARMUP_LR=5.0e-07,
    MIN_LR=1.0e-07
)

total_epochs = 210

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

train = dict(
    counter='normal',
    image_size=(512, 512),  # core parameter
    route_size=(256, 256),  # core parameter
    base_size=None,
    batch_size_per_gpu=8,  # parameter
    shuffle=True,
    end_epoch=800,
    resume_path=None,
    flip=True,
    multi_scale=True,
    scale_factor=(0.5, 1 / 0.5),
    downsamplerate=1,
    ignore_label=255
)

test = dict(
    base_size=3072,  # core parameter
    batch_size_per_gpu=1,
    multi_scale=False
)

CUDNN = dict(
    BENCHMARK=True,
    DETERMINISTIC=False,
    ENABLED=True)
