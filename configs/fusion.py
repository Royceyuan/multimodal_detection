"""Configuration for RGB-anchored multimodal feature adaptation."""

import os

_base_ = ["../vendor/Co-DETR/projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py"]

custom_imports = dict(
    imports=["trifusion.pipelines", "trifusion.model", "trifusion.training"],
    allow_failed_imports=False,
)
data_root = os.environ.get("TRIMODAL_DATA_ROOT", "data")
annotation_root = os.environ.get(
    "TRIMODAL_ANNOTATION_ROOT", "data/annotations")
classes = (
    "person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
    "light", "garbage can", "uav", "tricycle",
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadAlignedMultimodal", infrared_root=os.path.join(data_root, "train", "infrared"),
         depth_root=os.path.join(data_root, "train", "depth")),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="ResizeAlignedMultimodal", img_scale=[(1333, 750), (1536, 864)],
         keep_ratio=True, multiscale_mode="value"),
    dict(type="NormalizeAlignedMultimodal", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadAlignedMultimodal", infrared_root=os.path.join(data_root, "val", "infrared"),
         depth_root=os.path.join(data_root, "val", "depth")),
    dict(type="MultiScaleFlipAug", img_scale=(1333, 750), flip=False, transforms=[
        dict(type="ResizeAlignedMultimodal", keep_ratio=True),
        dict(type="RandomFlip"),
        dict(type="NormalizeAlignedMultimodal", **img_norm_cfg),
        dict(type="Pad", size_divisor=32),
        dict(type="ImageToTensor", keys=["img"]),
        dict(type="Collect", keys=["img"]),
    ]),
]
data = dict(
    _delete_=True, samples_per_gpu=1, workers_per_gpu=2,
    train=dict(type="CocoDataset", classes=classes, filter_empty_gt=False,
               ann_file=os.path.join(annotation_root, "instances_train.json"),
               img_prefix=os.path.join(data_root, "train", "RGB"), pipeline=train_pipeline),
    val=dict(type="CocoDataset", classes=classes,
             ann_file=os.path.join(annotation_root, "instances_valid.json"),
             img_prefix=os.path.join(data_root, "val", "RGB"), pipeline=test_pipeline),
    # Use the validation split for development-time evaluation.
    test=dict(type="CocoDataset", classes=classes,
              ann_file=os.path.join(annotation_root, "instances_valid.json"),
              img_prefix=os.path.join(data_root, "val", "RGB"), pipeline=test_pipeline),
)

# Lists replace the inherited lists in MMCV: retain each original head and loss,
# changing only its class count. Query/RPN dictionaries merge into the official base.
model = dict(
    type="TriModalCoDETR",
    query_head=dict(num_classes=12),
    roi_head=[dict(
        type="CoStandardRoIHead",
        bbox_roi_extractor=dict(type="SingleRoIExtractor",
                                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                                out_channels=256, featmap_strides=[4, 8, 16, 32, 64], finest_scale=56),
        bbox_head=dict(type="ConvFCBBoxHead", num_shared_convs=4, num_shared_fcs=1,
                       in_channels=256, conv_out_channels=256, fc_out_channels=1024,
                       roi_feat_size=7, num_classes=12,
                       bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0., 0., 0., 0.],
                                       target_stds=[0.05, 0.05, 0.1, 0.1]),
                       reg_class_agnostic=True, reg_decoded_bbox=True,
                       norm_cfg=dict(type="GN", num_groups=32),
                       loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=12.0),
                       loss_bbox=dict(type="GIoULoss", loss_weight=120.0)))],
    bbox_head=[dict(
        type="CoATSSHead", num_classes=12, in_channels=256, stacked_convs=1, feat_channels=256,
        anchor_generator=dict(type="AnchorGenerator", ratios=[1.0], octave_base_scale=8,
                              scales_per_octave=1, strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=12.0),
        loss_bbox=dict(type="GIoULoss", loss_weight=24.0),
        loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=12.0))],
)
optimizer = dict(
    _delete_=True, type="AdamW", lr=2e-5, weight_decay=0.01,
    constructor="DefaultOptimizerConstructor",
    paramwise_cfg=dict(custom_keys={"tri_fusion": dict(lr_mult=5.0)}),
)
# The custom hook owns native AMP, accumulation and finite/effective-gradient checks.
# A non-None fp16 value would make train_detector replace it with the standard hook.
fp16 = None
optimizer_config = dict(
    _delete_=True, type="Stage1OptimizerHook", cumulative_iters=4,
    grad_clip=dict(max_norm=0.1, norm_type=2), loss_scale="dynamic",
)
custom_hooks = [
    dict(type="Stage1FreezeHook", priority="VERY_HIGH"),
]
# Warmup is 100 microsteps (25 accumulated updates), not 100 optimizer updates.
lr_config = dict(_delete_=True, policy="step", warmup="linear", warmup_iters=100,
                 warmup_ratio=0.01, step=[4])
runner = dict(type="EpochBasedRunner", max_epochs=6)
evaluation = dict(interval=1, metric="bbox", classwise=True, proposal_nums=(1, 10, 100),
                  save_best="bbox_mAP", rule="greater")
checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_optimizer=True)
log_config = dict(interval=20, hooks=[dict(type="TextLoggerHook")])
load_from = None  # The launcher loads the audited COCO initialization before training.
resume_from = None
seed = 42
deterministic = False
stage_phase = "fusion"
