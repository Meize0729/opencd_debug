_base_ = [
    '../_base_/models/bit_r18.py', 
    '../common/standard_512x512_40k_second_bs2.py']

wandb = 0
names = 'bit_r18_512x512_40k_second.py'
work_dir = '/mnt/public/usr/wangmingze/work_dir/CD_others/' + names


if wandb:
    vis_backends = [dict(type='CDLocalVisBackend'),
                    dict(
                        type='WandbVisBackend',
                        save_dir=
                        '/mnt/public/usr/wangmingze/opencd/wandb/try',
                        init_kwargs={
                            'entity': "wangmingze",
                            'project': "opencd_2",
                            'name': names,}
                            )
                    ]
        
