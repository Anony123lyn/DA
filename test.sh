#############     Source_only
CUDA_VISIBLE_DEVICES=0 python tools/test_GTA_source_only.py --cfg experiments/GTA/LTIR_vgg_NEW.yaml TEST.TEST_FLIP True  VIS_DIR BB

CUDA_VISIBLE_DEVICES=0 python tools/test_GTA_source_only.py --cfg experiments/GTA/LTIR_vgg_gta.yaml TEST.TEST_FLIP True VIS_DIR aa

CUDA_VISIBLE_DEVICES=1 python tools/test_GTA_source_only.py --cfg experiments/GTA/Train_StoL_V2_1.yaml TEST.TEST_FLIP True


CUDA_VISIBLE_DEVICES=0 python tools/test_GTA_source_only.py --cfg experiments/GTA/LTIR_vgg_gta.yaml TEST.TEST_FLIP True