import numpy as np
import jupyter

def hex2num(c):
    c = c.replace('#','')
    return np.array([int(f"0x{c[ii:ii + 2]}", 16) for ii in range(0, 6, 2)])


x = "#34E2E2"
t = "#FFFFFF"

x = hex2num(x)
t = hex2num(t)

alpha = 0.67

y = alpha * x + (1 - alpha) * t

y = '#' + ''.join([hex(int(channel)).upper()[-2:] for channel in y])

print(y)
