import numpy as np
import csv
import matplotlib.pyplot as plt
import pdb

def parte_1():
    def import_data(file):
        with open(file) as csvfile:
            data=np.empty([sum(1 for line in csvfile),9])
            csvfile.seek(0)
            reader=csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
            for i,row in enumerate(reader):
                if row:
                    data[i]=row
            masse=data[:,0]
            data=data[:,1:]
            return masse,data

    #data
    masse,tempi=import_data('data/misure_8osc_8rip.csv')
    rip=8
    osc=8
    l=.8
    #adj measurement units
    masse=masse*1e-3
    P_units='N'
    m_units='Kg'
    t_units='s'
    l_units='m'

    #uncertainty
    delta_m=1e-4/np.sqrt(12)
    delta_x=1e-3/np.sqrt(12)
    delta_t=1e-2/np.sqrt(12) #CHECK: maybe not
    #dt of a single oscillation
    pdb.set_trace()
    dt=np.sqrt(((delta_t/osc)**2)*np.ones(len(tempi)) +
                np.std(tempi/8)**2)
    print(dt)

    ## PLOT_1
    fig, ax = plt.subplots()
    ax.errorbar(masse,[np.mean(t/8) for t in tempi],xerr=dm,yerr=dt/osc,fmt='b.')
    ax.set_title('T=T(M)')
    ax.set_ylabel(['T [',t_units,']'])
    ax.set_xlabel(['M [',m_units,']'])
    plt.show(fig)
