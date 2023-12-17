# +
python manage.py --run 'age_all_vit224_ech_g_gender/MSE' --ex 'age-prediction2' --model_cfgs 'configs/age_all_vit224_ech_g_gender.json' --epochs 50 --data_path '/home/work/AGE'
python manage.py --run 'age_all_vit224_ech_g_race_gender/MSE' --ex 'age-prediction2' --model_cfgs 'configs/age_all_vit224_ech_g_race_gender.json' --epochs 50 --data_path '/home/work/AGE'
python manage.py --run 'age_all_vit224_ech_g_race/MSE' --ex 'age-prediction2' --model_cfgs 'configs/age_all_vit224_ech_g_race.json' --epochs 50 --data_path '/home/work/AGE'


