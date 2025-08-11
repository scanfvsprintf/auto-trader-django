import easytrader
import json
f=open('config.json','r',encoding='utf-8')
t=""
for line in f:
    line = line.strip()
    t=t+line
    if not line:
        continue
f.close()
config=json.loads(t)
print(config)
user = easytrader.use('ht_client')
user.connect(config['easytrader']['client_path'])
user.prepare(config['easytrader']['user'])
user.refresh()
print(user.balance)
user.exit()