import pandas as pd

#0.764
sub1 = pd.read_csv('../submit/kzh5543.csv')
#0.759
sub2 = pd.read_csv('../submit/sub.csv')

submit = sub1.merge(sub2, on = 'train_file_name', how = 'left')
submit['life'] = submit['life_x'] * 0.5 + submit['life_y'] * 0.5
submit[['train_file_name', 'life']].to_csv('../submit/end_755.csv',index=False)
