source /home/ec2-user/aria2-downloads/backtrader/venvQT/bin/activate
data='/home/ec2-user/aria2-downloads/backtrader/QT/data'
save_path='/home/ec2-user/aria2-downloads/backtrader/QT/results'
pyfile=/home/ec2-user/aria2-downloads/backtrader/QT/code/test.py
setting='/home/ec2-user/aria2-downloads/backtrader/QT/code/setting.yml'

# download stocks info
python /home/ec2-user/aria2-downloads/backtrader/QT/code/dl.py

# run analysis
method='vote'
save_path='/home/ec2-user/aria2-downloads/backtrader/QT/forcast'
python3 $pyfile --forcast --method $method --folder_mode --data $data --save_path $save_path --config $setting > $save_path/$method.txt

#send email
python3 /home/ec2-user/aria2-downloads/backtrader/QT/code/sendnotify.py