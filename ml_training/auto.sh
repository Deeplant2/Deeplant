# +
python manage.py --run 'dp_all_vit448/MSE' --ex 'Deeplant' --model_cfgs 'configs/deeplant/dp_all_vit448.json' --epochs 50
python manage.py --run 'dp_all_vit448_ech_c_grade/MSE' --ex 'Deeplant' --model_cfgs 'configs/deeplant/dp_all_vit448_ech_c_grade.json' --epochs 50 
python manage.py --run 'dp_all_vit448_ech_g_gradeColor/MSE' --ex 'Deeplant' --model_cfgs 'configs/deeplant/dp_all_vit448_ech_g_gradeColor.json' --epochs 50
python manage.py --run 'dp_all_vit448_ech_g_gradeMarbling/MSE' --ex 'Deeplant' --model_cfgs 'configs/deeplant/dp_all_vit448_ech_g_gradeMarbling.json' --epochs 50 
python manage.py --run 'dp_all_vit448_ech_g_gradeTexture/MSE' --ex 'Deeplant' --model_cfgs 'configs/deeplant/dp_all_vit448_ech_g_gradeTexture.json' --epochs 50 
python manage.py --run 'dp_all_vit448_ech_g_gradeSurface/MSE' --ex 'Deeplant' --model_cfgs 'configs/deeplant/dp_all_vit448_ech_g_gradeSurface.json' --epochs 50 
python manage.py --run 'dp_all_vit448_ech_g_gradeTotal/MSE' --ex 'Deeplant' --model_cfgs 'configs/deeplant/dp_all_vit448_ech_g_gradeTotal.json' --epochs 50 


