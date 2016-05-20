def test_1twr10xC1():
    out = read_data.read_cdf(fname, data_product='1twr10xC1')
    out = out['1twr10xC1']
    soll = pd.read_csv(fname+'1twr10xC1_rh.csv', index_col=0,
                       dtype={'rh_25m': np.float32, 'rh_60m': np.float32}
                      )
    assert np.all(out.relative_humidity.data == soll)

    # create the test file
    out = read_data.read_cdf(fname, data_product='1twr10xC1')
    out = out['1twr10xC1']
    #     out.relative_humidity.data.to_csv(fname+'1twr10xC1_rh.csv')