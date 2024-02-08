'''
python tools/train.py configs_finetune_new/ViT_L/ViT_L_inria.py
'''

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/standard_512x512_foundationdataset.py',
    '../_base_/models/ViT_L.py',
    '../_base_/schedules/schedule_default.py',
]

# You can change dataloader parameters here
bs=2
gpu_nums = 8
bs_mult = 1
num_workers = 8
persistent_workers = True

# data_list path !!!! must change this !!!!
train_data_list = '/mnt/public/usr/wangmingze/opencd/data_list/inria/data_list_inria_train.txt'
test_data_list = '/mnt/public/usr/wangmingze/opencd/data_list/inria/data_list_inria_test.txt'

# training schedule for pretrain
max_iters = 4e4
val_interval = 200
base_lr = 0.0001 * (bs * gpu_nums / 16) * bs_mult # lr is related to bs*gpu_num, default 16-0.0001


# If you want to train with some backbone init, you must change the dir for your personal save dir path
# But I think you will use our pretrained weight, you may do not need backbone_checkpoint
backbone_checkpoint = None
# load_from = 'the checkpoint path' # !!!! must change this !!!!
resume_from = None

# If you want to use wandb, make it to 1
wandb = 0

# You can define which dir want to save checkpoint and loggings
names = 'ViT_L_inria'
work_dir = '/mnt/public/usr/wangmingze/work_dir/finetune/' + names



""" ************************** 模型 **************************"""
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=backbone_checkpoint) if backbone_checkpoint else None
    ),
    finetune_cfg=None, #
)


""" ************************** data **************************"""
train_dataset = dict(
    dataset=dict(
        data_list=train_data_list,
    )
)
train_dataloader = dict(
    batch_size=bs,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
)

val_dataloader = dict(
    batch_size=bs,
    num_workers=num_workers,
    persistent_workers=persistent_workers, 
    dataset=dict(
        data_list=test_data_list,
    )
)

""" ************************** schedule **************************"""
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=base_lr
        ),
    )

""" ************************** 可视化 **************************"""
if wandb:
    vis_backends = [dict(type='CDLocalVisBackend'),
                    dict(
                        type='WandbVisBackend',
                        save_dir=
                        '/mnt/public/usr/wangmingze/opencd/wandb/try2',
                        init_kwargs={
                            'entity': "wangmingze",
                            'project': "opencd_all_v4",
                            'name': names,}
                            )
                    ]
