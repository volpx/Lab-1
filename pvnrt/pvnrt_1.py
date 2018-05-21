import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from functions import *

ZERO_C=273.15
MONO_FIGSIZE=(6,5)
DOUBLE_FIGSIZE=(10,5)
RHO_HG=13.534*1e3
RHO_H2O=1*1e3
G_ACC=9.806
R=8.31
D_TEMP=11+ZERO_C

def parte1():
    def import_misure(file_name):
        with open(file_name) as csvfile:
            data=[]
            reader=csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
            for i,row in enumerate(reader):
                if row and (not isinstance(row[1],str)):
                    data.append(row)
            data=np.array(data)
            return data[:,0],data[:,1],data[:,2]

    ## UNCERT
    dh_m=2e-3/np.sqrt(12)
    dh_a=2e-3/np.sqrt(12)
    dtemp=1e-2/np.sqrt(12)

    ## DATA
    time_m,h_m,temp=import_misure('data/misure_lun_14_5_18.csv')
    h_m*=1e-2
    temp+=ZERO_C
    time_a,h_a,pres_a=import_misure('data/pr_atm_lun_14_5_18.csv')
    h_a*=1e-2
    pres_a*=1e2
    ##calculating atmospheric pression at measured time
    p_a=RHO_HG*G_ACC*h_a+pres_a
    dp_a=RHO_HG*G_ACC*dh_a
    p_a_mt=np.empty(h_m.shape)
    for i in range(len(p_a_mt)):
        #i_b is the index in time_a for the time before time_m[i]
        i_b=0
        while(not (time_m[i] < time_a[i_b])):
            i_b+=1
        i_b-=1
        p_a_mt[i]=p_a[i_b]+(p_a[i_b+1]-p_a[i_b])/(time_a[i_b+1]-time_a[i_b])*(time_m[i]-time_a[i_b])
    ##nice perameters
    h_0=95*1e-2
    i_zero=np.unravel_index(np.argmin(temp),temp.shape)[0]
    h_m_wei=1/dh_m**2

    p_m=p_a_mt+RHO_H2O*G_ACC*(h_m-h_0)
    dp_m=np.sqrt(dp_a**2+2*(RHO_H2O*G_ACC*dh_m)**2)
    print('dp_m:',dp_m)
    p_m_wei=1/dp_m**2
    p_0=p_m[i_zero]
    alpha=1/ZERO_C

    ############################################################################
    #             ANALYSIS
    ############################################################################
    A1,B1,dA1,dB1=linear_regression_AB(temp,h_m,h_m_wei)
    print('Linear regression 1: h =',ufloat(A1,dA1),'+',ufloat(B1,dB1),'* T')
    chi2_ht1=chi2(h_m,dh_m,A1+B1*temp)
    print('Chi2 1: ',chi2_ht1,'non lineare')

    print('Considero solo i valori con T>D_TEMP')
    temp2=[]
    h_m2=[]
    p_m2=[]
    for i,t in enumerate(temp):
        if t > D_TEMP:
            temp2.append(t)
            h_m2.append(h_m[i])
            p_m2.append(p_m[i])
    temp2=np.array(temp2)
    h_m2=np.array(h_m2)
    p_m2=np.array(p_m2)
    i_min2=np.unravel_index(np.argmin(temp2),temp.shape)[0]

    A21,B21,dA21,dB21=linear_regression_AB(temp2[:i_min2],h_m2[:i_min2],h_m_wei)
    print('Linear regression scendendo: h =',ufloat(A21,dA21),'+',ufloat(B21,dB21),'* T')
    chi2_ht21=chi2(h_m2[:i_min2],dh_m,A21+B21*temp2[:i_min2])
    print('Chi2 1: ',chi2_ht21)

    A22,B22,dA22,dB22=linear_regression_AB(temp2[i_min2:],h_m2[i_min2:],h_m_wei)
    print('Linear regression salendo: h =',ufloat(A22,dA22),'+',ufloat(B22,dB22),'* T')
    chi2_ht22=chi2(h_m2[i_min2:],dh_m,A22+B22*temp2[i_min2:])
    print('Chi2 1: ',chi2_ht22)

    A3,B3,dA3,dB3=linear_regression_AB(temp2,p_m2,p_m_wei)
    print('Linear regression total: P =',ufloat(A3,dA3),'+',ufloat(B3,dB3),'* T')
    chi2_pt3=chi2(p_m2,dp_m,A3+B3*temp2)
    print('Chi2 1: ',chi2_pt3)

    A31,B31,dA31,dB31=linear_regression_AB(temp2[:i_min2],p_m2[:i_min2],p_m_wei)
    print('Linear regression scendendo: P =',ufloat(A31,dA31),'+',ufloat(B31,dB31),'* T')
    chi2_pt31=chi2(p_m2[:i_min2],dp_m,A3+B3*temp2[:i_min2])
    print('Chi2 1: ',chi2_pt31)

    A32,B32,dA32,dB32=linear_regression_AB(temp2[i_min2:],p_m2[i_min2:],p_m_wei)
    print('Linear regression salendo: P =',ufloat(A32,dA32),'+',ufloat(B32,dB32),'* T')
    chi2_pt32=chi2(p_m2[i_min2:],dp_m,A32+B32*temp2[i_min2:])
    print('Chi2 1: ',chi2_pt32)

    #TODO:redo when chi2 works
    # zero_guess=-(np.mean(p_a_mt)/RHO_H2O/G_ACC+A2)/B2
    # dzero_guess=np.sqrt((dp_a/(B2*RHO_H2O*G_ACC))**2+(dA2/B2)**2+((np.mean(p_a_mt)/RHO_H2O/G_ACC+A2)/B2)**2)
    # print('zero guess :',ufloat(zero_guess,dzero_guess))


    ############################################################################
    #             PLOTS
    ############################################################################
    #PLOT1
    fig1=plt.figure(figsize=DOUBLE_FIGSIZE)
    ax11=fig1.add_subplot(1,1,1)
    ax11.errorbar(temp[:i_zero],h_m[:i_zero],yerr=dh_m,xerr=dtemp,fmt='b.',label='Temp scendendo')
    ax11.errorbar(temp[i_zero:],h_m[i_zero:],yerr=dh_m,xerr=dtemp,fmt='r.',label='Temp salendo')
    ax12=ax11.twinx()
    ax12.set_ylabel('Pgas [Pa]')
    ax12.errorbar(temp,p_m,yerr=dh_m,xerr=dtemp,fmt='g.',label='Temp salendo')
    ax11.set_xlabel('T [K]')
    ax11.set_ylabel('h [m]')
    ax11.grid()
    ax11_zoom=fig1.add_axes((.65,.2,.2,.2))
    ax11_zoom.errorbar(temp[10],h_m[10],yerr=dh_m,xerr=dtemp)
    ax11_zoom.set_xlabel('T [K]')
    ax11_zoom.set_ylabel('h [m]')
    mark_inset(ax11, ax11_zoom, loc1=1, loc2=3, fc="none", ec="0.5")
    fig1.suptitle('Dipendenza di h da T',fontsize=16)


    #PLOT2
    fig2=plt.figure(figsize=DOUBLE_FIGSIZE)
    fig2.suptitle('Dipendenza di h da T',fontsize=16)
    ax21=fig2.add_subplot(1,1,1)
    ax21.errorbar(temp2[:i_min2],h_m2[:i_min2]-A21-B21*temp2[:i_min2],yerr=dh_m,xerr=dtemp,fmt='b.',label='Temp scendendo')
    ax21.errorbar(temp2[i_min2:],h_m2[i_min2:]-A22-B22*temp2[i_min2:],yerr=dh_m,xerr=dtemp,fmt='r.',label='Temp salendo')
    ax21.axhline(y=0)
    ax21.set_xlabel('T [K]')
    ax21.set_ylabel('h [m]')

    #PLOT3
    fig3=plt.figure(figsize=DOUBLE_FIGSIZE)
    ax31=fig3.add_subplot(1,1,1)
    ax31.errorbar(temp2[:i_min2],p_m2[:i_min2]-A31-B31*temp2[:i_min2],yerr=dp_m,xerr=dtemp,fmt='b.',label='Temp scendendo')
    ax31.axhline(y=0)
    ax31.set_xlabel('T [K]')
    ax31.set_ylabel('P [Pa]')
    fig3.suptitle('Dipendenza di P da T scendendo',fontsize=16)

    #PLOT4
    fig4=plt.figure(figsize=DOUBLE_FIGSIZE)
    fig4.suptitle('Dipendenza di P da T salendo',fontsize=16)
    ax41=fig4.add_subplot(1,1,1)
    ax41.errorbar(temp2[i_min2:],p_m2[i_min2:]-A32-B32*temp2[i_min2:],yerr=dp_m,xerr=dtemp,fmt='b.',label='Temp scendendo')
    ax41.axhline(y=0)
    ax41.set_xlabel('T [K]')
    ax41.set_ylabel('P [Pa]')

    plt.show()
