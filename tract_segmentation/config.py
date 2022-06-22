class CFG:
    seed = 101
    debug = False  # set debug=False for Full Training
    exp_name = "Baselinev2"
    model_name = "Unet++"
    backbone = "efficientnet-b6"
    img_size = [384, 384]
    comment = f"{model_name}-{backbone}-{img_size[0]}x{img_size[1]}"
    train_bs = 32
    valid_bs = train_bs * 2
    epochs = 15
    lr = 2e-3
    scheduler = "CosineAnnealingLR"
    min_lr = 1e-6
    T_max = int(30000 / train_bs * epochs) + 50
    T_0 = 25
    warmup_epochs = 0
    wd = 1e-6
    n_accumulate = max(1, 32 // train_bs)
    n_fold = 5
    num_classes = 3
    device = "cuda"
    device_ids = [0]
    data_path = "/kaggle/input/uwmgi-mask-dataset/train.csv"
    labels = {
        1:'large_bowel', 2:'small_bowel', 3:'stomach'
    }
    threshold = 0.45
    checkpoints = f'/kaggle/input/tract-segmentation-models//{comment}'
