def read_serial(port):
    import serial
    ser = serial.Serial(port)
    data_rcv = ser.read()
    return data_rcv