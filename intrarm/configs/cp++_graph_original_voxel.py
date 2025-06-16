import itertools

tasks = [
    dict(num_class=3, class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST']),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

fix_weight_path = None #"pretrained/cp++_twostage.pth"

head = dict(
    type="MiXHead", # MixHead
    input_channels=320,
    model_cfg=dict(
        CLASS_AGNOSTIC=True,
        GRAPH_OUT_DIM=256,
        GRAPH_TYPE='t',
        REDUCE_NODE=True,
        SHARED_FC=[256, 256],
        CLS_FC=[256, 256],
        REG_FC=[256, 256],
        DP_RATIO=0.3,
        NUM_UPDATES=4,
        EDGE_RADIUS=2.0,
        TARGET_CONFIG=dict(
            SAMPLE_ROI_BY_EACH_CLASS=True,
            CLS_SCORE_TYPE='roi_iou',
            CLS_FG_THRESH=0.9,
            CLS_BG_THRESH=0.5,
            CLS_BG_THRESH_LO=0.1,
            HARD_BG_RATIO=0.8,
            REG_FG_THRESH=0.55
        ),
        NMS_CONFIG=dict(
            APPLY=False,
            SCORE_THRESH=0.1,
            PRE_MAX_SIZE=4096,
            NMS_THRESH=0.3,
            POS_MAX_SIZE=500
        ),
        LOSS_CONFIG=dict(
            CLS_LOSS='BinaryCrossEntropy',
            REG_LOSS='L1',
            LOSS_WEIGHTS={
                'rcnn_cls_weight': 1.0,
                'rcnn_reg_weight': 1.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
            }
        )
    ),
    code_size=9
)

data = dict(
    train=dict(
        batch_size=8,
        num_workers=8,
        sampling_interval=1
    ),
    val=dict(
        batch_size=8,
        num_workers=8,
        sampling_interval=1
    )
)

radius_t = 2.0 # useless value
log_level = "INFO"
weight_decay = 0.01
total_epochs = 10
