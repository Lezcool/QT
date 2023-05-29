source /home/lez/Documents/QT/venvQT/bin/activate
data='/home/lez/Documents/QT/QT/data'
save_path='/home/lez/Documents/QT/QT/results/vote(ksrma)'
#if path doenst exist, create it
if [ ! -d "$save_path" ]; then
  mkdir $save_path
fi

pyfile=/home/lez/Documents/QT/QT/code/test.py
setting='/home/lez/Documents/QT/QT/code/setting.yml'
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'vote' --maperiod 10
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'sma' --optimize 
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'ai' --optimize
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'vote' --maperiod 10

# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'sma' --optimize --folder_mode --data '/zhome/dc/1/174181/docs/QT/data' > /zhome/dc/1/174181/docs/QT/results/sma_results.txt

method='vote'
python3 $pyfile --method $method --folder_mode --data $data --save_path $save_path --plot --config $setting > $save_path/$method.txt

# below is for forcast
# method='vote'
# save_path='/zhome/dc/1/174181/docs/QT/forcast'
# python3 $pyfile --forcast --method $method --folder_mode --data $data --save_path $save_path
