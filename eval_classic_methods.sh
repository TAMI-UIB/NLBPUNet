
batch_size=5
for model in 'Bicubic' 'Brovey' 'GSA'
do
  python classic_methods main.py --model $model --train-params batch_size=$batch_size max_epochs=1
done