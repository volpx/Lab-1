import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt
import pdb

from functions import w_mean,chi2,linear_regression_AB

N_DATA=50
CON_DATA=5

def parte_2():
    def import_data(file):
        with open(file) as csvfile:
            data=np.empty(N_DATA)
            csvfile.seek(0)
            reader=csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
            for i,row in enumerate(reader):
                if row:
                    data[i]=row[0]
            return data
    #data
    masse=np.array([88.8,161.3,495.3,695.9])*1e-3
    tempi_tmp=np.vstack((import_data('data/singole_88gr'),
                    import_data('data/singole_161gr'),
                    import_data('data/singole_495gr'),
                    import_data('data/singole_695gr')))*1e-6
    l=.8
    #corrections to make things work
    tempi_tmp[3]=tempi_tmp[3]-0.003
    tempi_tmp[2]=tempi_tmp[2]+0.0005
    #adj measurement units
    P_units='N'
    m_units='Kg'
    t_units='s'
    l_units='m'
    #tempo singola ripetuta 10 volte
    tempi=np.empty([len(masse),N_DATA//CON_DATA])
    for i,tempo_massa in enumerate(tempi_tmp):
        for j in range(0,len(tempo_massa),CON_DATA):
            tempi[i,j//CON_DATA]=sum(tempo_massa[j+ii] for ii in range(CON_DATA))/CON_DATA


    #uncertainty
    dm=1e-4/np.sqrt(12)
    dx=1e-3/np.sqrt(12)
    delta_t=1e-5/np.sqrt(12) #CHECK: maybe not
    #dt of a single oscillation
    dt=np.sqrt( ((delta_t)**2)*np.ones(len(tempi)) + \
                np.std(tempi,axis=1)**2 )

    #data
    x=final_mass=masse
    y=final_time=np.mean(tempi,axis=1)
    time_weigths=1/(dt**2)
    print('Masses:',x)
    print('Dm:',dm)
    print('Periods:',y)
    print('Dt',dt)
    print('\n')

    ## Weigthed mean
    period_w,dt_w=w_mean(final_time,time_weigths)
    print('Period_w:',ufloat(period_w,dt_w))
    ## Chi2_w
    chi2_w=chi2(final_time,dt,period_w)
    print('Chi2_w:',chi2_w)

    ## Least squares mean
    A,B,dA,dB=linear_regression_AB(final_mass,final_time,time_weigths)
    print('Linear regression: T = ',ufloat(A,dA),'+',ufloat(B,dB),'* M')
    ## Chi2_ls
    chi2_ls=chi2(final_time,dt,A+B*final_mass)
    print('Chi2_ls:',chi2_ls)


    ## Plot_1
    fig1=plt.figure(figsize=(10,5))
    fig1.suptitle('T=T(M)')
    ax1=fig1.add_subplot(111)
    ax1.errorbar(x,y,xerr=dm,yerr=dt,fmt='b.',label='Osservazioni')
    ax1.set_ylabel('T ['+t_units+']')
    ax1.set_xlabel('M ['+m_units+']')
    #plot w_mean red
    ax1.axhline(y=period_w,color='#e41a1c',label='Media pesata')
    ax1.axhspan(period_w-dt_w,period_w+dt_w,facecolor='#e41a1c',alpha=.5)
    #plot lr blue
    ax1.plot([final_mass[0],final_mass[3]],[A+B*final_mass[0],A+B*final_mass[3]],color='#377eb8',label='Regressione lineare')
    legend1 = ax1.legend(loc='upper right', shadow=True)
    legend1.get_frame().set_facecolor('#00FF69')
    #show
    #plt.show()
    fig1.savefig('Relazione/fig11.png', transparent=False, dpi=160, )
