# +
python manage.py --run 'dp_all_vit224/MSE' --ex 'Deeplant' --model_cfgs 'configs/deeplant/dp_all_vit224.json' --epochs 50
python manage.py --run 'dp_all_resnet152/MSE' --ex 'Deeplant' --model_cfgs 'configs/deeplant/dp_all_resnet152.json' --epochs 50
python manage.py --run 'dp_all_resnet270/MSE' --ex 'Deeplant' --model_cfgs 'configs/deeplant/dp_all_resnet270.json' --epochs 50
python manage.py --run 'dp_all_vit448_token_grade/MSE' --ex 'Deeplant' --model_cfgs 'configs/deeplant/dp_all_vit448_token_grade.json' --epochs 50 


