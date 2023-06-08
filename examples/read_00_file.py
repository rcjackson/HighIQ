import highiq

ds = highiq.io.read_00_data('sgpdlacfC1.00.20220107.141001.raw.aet_Stare_107_20220107_14.raw',
                            'sgpdlprofcalC1.home_point')
my_spectra = highiq.calc.get_psd(ds)
print(ds)