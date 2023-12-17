# +
python manage.py --run 'age_all_vit224_token_gender/MSE' --ex 'age-prediction2' --model_cfgs 'configs/age_all_vit224_token_gender.json' --epochs 50 --data_path '/home/work/AGE'
python manage.py --run 'age_all_vit224_token_race/MSE' --ex 'age-prediction2' --model_cfgs 'configs/age_all_vit224_token_race.json' --epochs 50 --data_path '/home/work/AGE'
python manage.py --run 'age_all_vit224_token_race_gender/MSE' --ex 'age-prediction2' --model_cfgs 'configs/age_all_vit224_token_race_gender.json' --epochs 50 --data_path '/home/work/AGE'


