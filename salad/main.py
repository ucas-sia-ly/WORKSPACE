import pytorch_lightning as pl

from vpr_model import VPRModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule

if __name__ == '__main__':
    datamodule = GSVCitiesDataModule(
        batch_size=60,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False, # 表示是否在整个数据集上进行随机打乱，通常在训练时设置为True，在验证或测试时设置为False
        random_sample_from_each_place=True,# 如果超过 img_per_place，则从每个地点随机采样图像
        image_size=(224, 224),
        num_workers=10,# 进程越多，数据加载速度越快，但也会占用更多的系统资源。根据你的硬件配置和数据集大小进行调整。
        show_data_stats=True,# 初始化时显示数据集的统计信息，如每个地点的图像数量、总图像数量等。
        val_set_names=['pitts30k_val', 'pitts30k_test', 'msls_val'], # pitts30k_val, pitts30k_test, msls_val
    )

    model = VPRModel(
        #---- Encoder
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
        lr = 6e-5,
        optimizer='adamw',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='pitts30k_val/R1',
        filename=f'{model.encoder_arch}' + '_({epoch:02d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode='max'
    )

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,# 一个GPU
        default_root_dir=f'./logs/', # 默认保存日志和检查点的目录
        num_nodes=1,# 一台机器
        num_sanity_val_steps=0, # 正式训练前不进行验证集的sanity check
        precision='16-mixed', # 使用混合精度训练（16位浮点数），可以加快训练速度并减少显存占用
        max_epochs=4,
        check_val_every_n_epoch=1, # 每一个epoch进行一次验证集评估
        callbacks=[checkpoint_cb],# 根据checkpoint_cb回调函数保存模型检查点
        reload_dataloaders_every_n_epochs=1, # 每一个epoch重新加载数据集，打乱数据顺序
        log_every_n_steps=20,# 每20个step记录一次日志
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
