
batch_size=5
for model in 'GPPNN' 'MDCUN' 'NLRNet'
do
  python main.py train --model $model --train-params batch_size=$batch_size max_epochs=1
done