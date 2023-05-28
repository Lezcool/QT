source /home/ec2-user/aria2-downloads/backtrader/venvQT/bin/activate
data='/home/ec2-user/aria2-downloads/backtrader/QT/data'
save_path='/home/ec2-user/aria2-downloads/backtrader/QT/results'
pyfile='/home/ec2-user/aria2-downloads/backtrader/QT/code/test.py'
setting='/home/ec2-user/aria2-downloads/backtrader/QT/code/setting.yml'
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'vote' --maperiod 10
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'sma' --optimize 
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'ai' --optimize
# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'vote' --maperiod 10

# python3 /zhome/dc/1/174181/docs/QT/code/test.py --method 'sma' --optimize --folder_mode --data '/zhome/dc/1/174181/docs/QT/data' > /zhome/dc/1/174181/docs/QT/results/sma_results.txt

# method='macd'
# python3 $pyfile --method $method --folder_mode --data $data --save_path $save_path --config $setting > $save_path/$method.txt

#below if forcast
method='vote'
save_path='/home/ec2-user/aria2-downloads/backtrader/QT/forcast'
python3 $pyfile --forcast --method $method --folder_mode --data $data --save_path $save_path --config $setting > $save_path/$method.txt
