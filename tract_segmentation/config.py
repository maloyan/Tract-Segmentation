class CFG:
    KAGGLE_DIR = "/kaggle/"
    INPUT_DIR = f"{KAGGLE_DIR}/input"
    OUTPUT_DIR = f"{KAGGLE_DIR}/working"

    INPUT_DATA_DIR = f"{INPUT_DIR}/uw-madison-gi-tract-image-segmentation"
    INPUT_DATA_NPY_DIR = f"{INPUT_DIR}/uw-madison-gi-tract-image-segmentation-masks"

    SPATIAL_SIZE = (224, 224, 96)
    N_SPLITS = 5
    RANDOM_SEED = 2022
    VAL_FOLD = 0
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    OPTIMIZER = "Adam"
    LEARNING_RATE = 2e-3
    WEIGHT_DECAY = 1e-6
    SCHEDULER = None
    MIN_LR = 1e-6

    FAST_DEV_RUN = False # Debug training
    GPUS = 1
    MAX_EPOCHS = 200
    PRECISION = 32

    DEVICE = "cuda"
    THR = 0.45

    DEBUG = False # Debug complete pipeline

    LOGS_PATH = "/kaggle/input/tract-segmentation-models/logs"
    PRETRAINED_PATH = "/root/tract_segmentation/checkpoints/model_swinvit.pt"
