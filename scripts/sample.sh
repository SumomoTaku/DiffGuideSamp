PATH_IMAGE_POOL=/home/user/Sumomo/Project/Dataset/woof20
PATH_ORI=/home/user/Sumomo/Project/Dataset/imageWoof/train
PATH_OUT=/home/user/Sumomo/Project/Dataset/woof20/test

python ../sampling.py --ip-root ${PATH_IMAGE_POOL} --ori-dataset ${PATH_ORI} --save-path ${PATH_OUT} \
--ipc 10 --sample-distribution "scale"  --th-file "../threshold.jsonl" --repeat 1
