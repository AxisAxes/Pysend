import serial
import time
def read_serial():
    print('ENTRY METHOD 1')
    timeout = time.time() + 30  
    print('EXECUTATE')
    ser = serial.Serial('COM7', 9600, timeout=1) 
    while (True):
        line = ser.readline()
        print(str(line))
   
   
   
