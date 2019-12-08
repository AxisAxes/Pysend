def build_data(port, file_name):
    from ReadSerial import read_serial
    import pandas as pd
    import numpy as np
    data = read_serial(port)
    real_data = np.array(data)
    df = pd.DataFrame(real_data, index=[ ind for ind, x in enumerate(real_data)] columns='TEMPERATURE HUMIDITY'.split())
    return df.to_csv(file_name, encoding='utf-8', index=False)
