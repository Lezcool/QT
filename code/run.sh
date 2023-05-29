source /zhome/dc/1/174181/docs/QT/venvQT/bin/activate
data='/zhome/dc/1/174181/docs/QT/data'
save_path='/zhome/dc/1/174181/docs/QT/results'
pyfile=/zhome/dc/1/174181/docs/QT/code/test.py
setting='/zhome/dc/1/174181/docs/QT/code/setting.yml'
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'vote' --maperiod 10
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'sma' --optimize 
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'ai' --optimize
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'vote' --maperiod 10

# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'sma' --optimize --folder_mode --data '/zhome/dc/1/174181/docs/QT/data' > /zhome/dc/1/174181/docs/QT/results/sma_results.txt

method='vote'
python3 $pyfile --method $method --folder_mode --data $data --save_path $save_path --plot> $save_path/$method.txt

# below is for forcast
# method='vote'
# save_path='/zhome/dc/1/174181/docs/QT/forcast'
# python3 $pyfile --forcast --method $method --folder_mode --data $data --save_path $save_path
