
# +
#python manage.py --run 'dp_all_vit224/reg/train' --ex 'code test' --model_cfgs 'configs/deeplant/dp_all_vit224.json' --epochs 10 --sanity true
#python manage.py --run 'dp_all_vit224/reg/test' --ex 'code test' --model_cfgs 'configs/deeplant/dp_all_vit224.json' --epochs 10 --sanity true --mode 'test'
# -


#python manage.py --run 'dp_all_vit224/cla/train' --ex 'code test' --model_cfgs 'configs/classification/dp_all_vit224.json' --epochs 10 --sanity true
python manage.py --run 'dp_all_vit224/cla/test' --ex 'code test' --model_cfgs 'configs/classification/dp_all_vit224.json' --epochs 10 --sanity true --mode 'test'
