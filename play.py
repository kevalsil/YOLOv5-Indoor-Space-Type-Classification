
import detect
import os
'''
display = utils.notebook_init() # checks
'''
#import importlib
#importlib.reload(detect)
'''
for i in range(1, 100):
  imgname = 'C:/Users/korea/OneDrive/문서/작업/거실T/livingroom interior{0}.jpg'.format(i)
  if os.path.exists(imgname):
    detect.run(source = imgname)'''

imgname = 'livingroom interior34.jpg'
if os.path.exists(imgname):
  detect.run(source = imgname)

#detect.run(source = 'C:/Users/korea/OneDrive/문서/작업/거실/거실{0}.jpg'.format(40))
#display.Image(filename='runs/detect/exp/거실40.jpg', width=600)