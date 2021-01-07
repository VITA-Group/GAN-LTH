import os
import shutil
from glob import glob
percent = ['0.8', '0.64', '0.512', '0.4096', '0.32768', '0.262144', '0.2097152', '0.1677722'
,'0.1342177','0.1073742']

Alist = sorted(glob(os.path.join(percent[0], 'horse2zebra', 'A', '*.png')))
Blist = sorted(glob(os.path.join(percent[0], 'horse2zebra', 'B', '*.png')))
Aname = []
Bname = []
for i in Alist:
    Aname.append(i.split('/')[-1].split('.')[0])

for i in Blist:
    Bname.append(i.split('/')[-1].split('.')[0])
    
for p in Aname:
    if not os.path.exists(p):
        os.mkdir(p)
    for per in percent:
        shutil.copy(os.path.join(per, 'horse2zebra', 'A', p + '.png'), os.path.join(p, per + '_A.png'))

for p in Bname:
    if not os.path.exists(p):
        os.mkdir(p)
    for per in percent:
        shutil.copy(os.path.join(per, 'horse2zebra', 'B', p + '.png'), os.path.join(p, per + '_B.png'))