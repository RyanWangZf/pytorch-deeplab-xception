# deeplab-resnet
# python train.py --backbone resnet --lr 0.01 --workers 4 --epochs 50 --batch-size 8 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --dataset construction

# deeplab-resnet
# python train.py --backbone resnet --lr 0.01 --workers 4 --epochs 30 --loss-type "focal" --batch-size 8 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --dataset construction

# balanced loss
python train.py --backbone resnet --lr 0.01 --workers 4 --epochs 50 --batch-size 8 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --dataset construction --use-balanced-weights

