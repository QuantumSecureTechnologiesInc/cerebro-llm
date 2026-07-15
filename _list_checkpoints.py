import os
for f in os.listdir('checkpoints'):
    p = os.path.join('checkpoints', f)
    if os.path.isfile(p):
        print(f'{f}  {os.path.getsize(p)/1e6:.1f} MB')
