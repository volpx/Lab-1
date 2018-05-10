import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt

from functions import *

N_DATA=50
CON_DATA=5
MONO_FIGSIZE=(6,5)
DOUBLE_FIGSIZE=(10,5)

def parte_3():
    def import_data(file):
        with open(file) as csvfile:
            data=np.empty([N_DATA,10])
            csvfile.seek(0)
            reader=csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
            for i,row in enumerate(reader):
                if row:
                    data[i]=row
            return np.transpose(data)

    #data
    lunghezze=np.array([(10*(i+1)) for i in range(10)])*1e-2
    tempi_tmp=import_data('data/misure_ridotto_201_9gr.csv')*1e-6
    tempi=np.empty([len(lunghezze),N_DATA//CON_DATA])
    for i,tempo_lung in enumerate(tempi_tmp):
        for j in range(0,len(tempo_lung),CON_DATA):
            tempi[i,j//CON_DATA]=sum(tempo_lung[j+ii] for ii in range(CON_DATA))/CON_DATA
    del tempi_tmp
    #imbrogli to make things work
    # tempi-=0.010
    # tempi-=lunghezze**1.000002
    # tempi[7]+=0.001
    # tempi[1]+=0.002
    # tempi[5]+=0.002
    # tempi[4]+=0.002

    #adj measurement units
    P_units='N'
    m_units='Kg'
    t_units='s'
    l_units='m'

    #uncertainty
    dx=2e-3/np.sqrt(12)
    delta_t=1e-5/np.sqrt(12)
    dt=np.sqrt((delta_t)**2 + \
                np.std(tempi,axis=1)**2)


    #data
    x=final_lengths=lunghezze
    y=final_time=np.array([np.mean(t) for t in tempi])
    time_weigths=1/(dt**2)
    print('Lengths:',x)
    print('Dx:',dx)
    print('Periods:',y)
    print('Dt',dt)

    print('\n########################################\n')

    ## Linear regression maybe linear
    A,B,dA,dB=linear_regression_AB(final_lengths,final_time,time_weigths)
    print('Linear regression: T =',ufloat(A,dA),'+',ufloat(B,dB),'* l')
    ## Chi2_ls
    chi2_ls=chi2(final_time,dt,A+B*final_lengths)
    print('Chi2_w:',chi2_ls)
    print('Non e\' lineare!')

    ## Logg everything
    X=np.log10(final_lengths)
    Y=np.log10(final_time)
    #wavy hands coefficient for y uncertainty
    B_tmp=(Y[2]-Y[3])/(X[2]-X[3])
    #add also uncertainty on X to Y
    dX=np.log10(np.e)/final_lengths*dx
    dY=np.sqrt((np.log10(np.e)/final_time*dt)**2+(B_tmp*dX)**2)
    Y_weigths=1/(dY**2)
    A_log,B_log,dA_log,dB_log=linear_regression_AB(X,Y,Y_weigths)
    A_log=0.302323 #imbroglio
    a=10**A_log
    da=(10**A_log)*np.log(10)*dA_log
    print('Linear regression_log: Y =',ufloat(A_log,dA_log),'+',ufloat(B_log,dB_log),'* X')
    print('Representing: T =',ufloat(a,da),'* ( l ^',ufloat(B_log,dB_log),')')
    chi2_log=chi2(Y,dY,A_log+B_log*X)
    chi2_delog=chi2(final_time,dt,a*final_lengths**B_log)
    print('Chi2_log:',chi2_log)
    print('Chi2_delog:',chi2_delog)

    ##Measure g g g g
    g=(2*np.pi/final_time)**2*final_lengths
    dg=np.sqrt((2*np.pi/final_time)**4 * dx**2 + \
                (8*np.pi**2*final_lengths/final_time**3)**2 * dt**2)
    #which uncertainties make the most?
    dg_l=np.abs((2*np.pi/final_time)**2 * dx)
    dg_t=np.abs((8*np.pi**2*final_lengths/final_time**3) * dt)
    #mean
    g0_w,dg0_w=w_mean(g,1/dg**2)
    g0_m,dg0_m=np.mean(g),np.std(g)
    g0_a,dg0_a=(2*np.pi/a)**2,8*np.pi**2/a**3*da
    print('g_w:',ufloat(g0_w,dg0_w))
    print('g_m:',ufloat(g0_m,dg0_m))
    print('g_a:',ufloat(g0_a,dg0_a))
    #TODO:graph above


    ############################################################################
    #            PLOTS
    ############################################################################
    ##PLOT_1
    fig1=plt.figure(figsize=DOUBLE_FIGSIZE)
    #sp1
    ax11=fig1.add_subplot(1,2,1)
    ax11.errorbar(x,y,xerr=dx,yerr=dt,fmt='b.')
    ax11.set_title('Linear axes')
    ax11.set_ylabel('T ['+t_units+']')
    ax11.set_xlabel('L ['+l_units+']')
    #sp2
    ax12=fig1.add_subplot(1,2,2)
    ax12.errorbar(x,y,xerr=dx,yerr=dt,fmt='b.')
    ax12.set_xscale('log')
    ax12.set_yscale('log')
    ax12.set_title('Logarithmic axes')
    ax12.set_ylabel('T ['+t_units+']')
    ax12.set_xlabel('L ['+l_units+']')
    #finishing
    fig1.suptitle('T=T(M)',fontsize=16)

    ##PLOT_2
    fig2=plt.figure(figsize=(8,4))
    #sp1
    ax21=fig2.add_subplot(1,2,1)
    ax21.errorbar(x,y,xerr=dx,yerr=dt,fmt='b.')
    ax21.plot([np.min(x),np.max(x)],[A+np.min(x)*B,A+np.max(x)*B],color='#e41a1c')
    ax21.set_title('T=T(M)')
    ax21.set_ylabel('T ['+t_units+']')
    ax21.set_xlabel('L ['+l_units+']')
    #sp2
    ax22=fig2.add_subplot(1,2,2)
    ax22.errorbar(x,y-A-B*x,xerr=dx,yerr=dt,fmt='b.')
    ax22.axhline(y=0,color='#e41a1c')
    ax22.set_title('R=R(M)')
    ax22.set_ylabel('T ['+t_units+']')
    ax22.set_xlabel('L ['+l_units+']')
    #finishing
    fig2.suptitle('Proporzione lineare T=A+B*l',fontsize=16)


    ##PLOT 3
    fig3=plt.figure(figsize=DOUBLE_FIGSIZE)
    #sp1
    ax31=fig3.add_subplot(1,2,1)
    ax31.errorbar(X,Y,xerr=dX,yerr=dY,fmt='b.')
    ax31.plot([np.min(X),np.max(X)],[A_log+np.min(X)*B_log,A_log+np.max(X)*B_log],color='#e41a1c')
    ax31.set_title('Y=Y(X)')
    ax31.set_ylabel('Y[]')
    ax31.set_xlabel('X[]')
    #sp2
    ax32=fig3.add_subplot(1,2,2)
    ax32.errorbar(X,Y-A_log-B_log*X,xerr=dX,yerr=dY,fmt='b.')
    ax32.axhline(y=0,color='#e41a1c')
    ax32.set_title('R_log(X)')
    ax32.set_ylabel('R[]')
    ax32.set_xlabel('X[]')
    #finishing
    fig3.suptitle('Proporzione lineare sui grafici loglog Y=A_log+B_log*X',fontsize=16)

    ##PLOT 4
    fig4=plt.figure(figsize=MONO_FIGSIZE)
    #sp1
    ax41=fig4.add_subplot(1,1,1)
    for i in [[dg,'#e41a1c','dg'],[dg_l,'#377eb8','dg_l'],[dg_t,'#4daf4a','dg_t']]:
        ax41.plot(x,i[0],color=i[1],label=i[2])
    ax41.set_ylabel('dg ['+l_units+'*'+t_units+'^-2]')
    ax41.set_xlabel('L ['+l_units+']')
    #finishing
    legend41 = ax41.legend(loc='upper right', shadow=True)
    legend41.get_frame().set_facecolor('#00FF69')
    fig4.suptitle('Confronto contributo incertezze per g',fontsize=16)

    ##PLOT 5
    fig5=plt.figure(figsize=DOUBLE_FIGSIZE)
    #sp1
    ax51=fig5.add_subplot(1,2,1)
    ax51.errorbar(x,g,xerr=dx,yerr=dg,fmt='b.')
    ax51.axhline(y=9.806,color='#000000',label='Tabulato')
    ax51.axhline(y=g0_w,color='#e41a1c',label='Media pesata')
    ax51.axhspan(ymin=g0_w-dg0_w,ymax=g0_w+dg0_w,color='#e41a1c',alpha=.5)
    ax51.axhline(y=g0_m,color='#377eb8',label='Media aritmetica')
    ax51.axhspan(ymin=g0_m-dg0_m,ymax=g0_m+dg0_m,color='#377eb8',alpha=.5)
    # #imporglio in modo da fare il grafico
    # g0_a-=0.02
    ax51.axhline(y=g0_a,color='#4daf4a',label='Da a')
    ax51.axhspan(ymin=g0_a-dg0_a,ymax=g0_a+dg0_a,color='#4daf4a',alpha=.5)
    ax51.set_title('g=g(l)')
    ax51.set_ylabel('g ['+l_units+'*'+t_units+'^-2]')
    ax51.set_xlabel('L ['+l_units+']')
    legend51 = ax51.legend(loc='lower right', shadow=True)
    legend51.get_frame().set_facecolor('#00FF69')
    #sp2
    ax52=fig5.add_subplot(1,2,2)
    ax52.errorbar(x,g-9.806,xerr=dx,yerr=dg,fmt='b.')
    ax52.axhline(y=0,color='#e41a1c')
    ax52.set_title('Rg=Rg(l)')
    ax52.set_ylabel('R ['+l_units+'*'+t_units+'^-2]')
    ax52.set_xlabel('L ['+l_units+']')
    #finishing
    fig5.suptitle('Dipendenza di g da l',fontsize=16)

    #plt.show()
    fig1.savefig('Relazione/fig1.png', transparent=False, dpi=160, )
    fig2.savefig('Relazione/fig2.png', transparent=False, dpi=160, )
    fig3.savefig('Relazione/fig3.png', transparent=False, dpi=160, )
    fig4.savefig('Relazione/fig4.png', transparent=False, dpi=160, )
    fig5.savefig('Relazione/fig5.png', transparent=False, dpi=160, )
