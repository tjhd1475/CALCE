CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/CALCE/eval/qvh_stage2_eval.yaml
