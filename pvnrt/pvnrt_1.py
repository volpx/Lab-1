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
    dh_m=1e-3/np.sqrt(12)
    dh_a=1e-3/np.sqrt(12)
    dtemp=1e-2/np.sqrt(12)

    ## DATA
    time_m,h_m,temp=import_misure('data/misure_lun_14_5_18.csv')
    h_m*=1e-2
    temp+=ZERO_C
    time_a,h_a,pres_a=import_misure('data/pr_atm_lun_14_5_18.csv')
    h_a*=1e-2
    pres_a*=1e2
    ##atmospheric pression at measured time
    p_a=RHO_HG*G_ACC*h_a+pres_a
    dp_a=RHO_HG*G_ACC*dh_a
    p_a_mt=np.empty(h_m.shape)
    for i in range(len(p_a_mt)):
        #i_b is the index in time_a for the time befor time_m[i]
        i_b=0
        while(not (time_m[i] < time_a[i_b])):
            i_b+=1
        i_b-=1
        p_a_mt[i]=p_a[i_b]+(p_a[i_b+1]-p_a[i_b])/(time_a[i_b+1]-time_a[i_b])*(time_m[i]-time_a[i_b])
    ##nice perameters
    h_0=95*1e-2
    i_zero=np.unravel_index(np.argmin(temp),temp.shape)[0]
    h_m_wei=1/dh_m**2

    ############################################################################
    #             ANALYSIS
    ############################################################################
    A,B,dA,dB=linear_regression_AB(temp,h_m,h_m_wei)
    print('Linear regression: h =',ufloat(A,dA),'+',ufloat(B,dB),'* T')
    #TODO: chi2 is crazy
    chi2_ht=chi2(h_m,dh_m,A+B*temp)
    print('Chi2 : ',chi2_ht)
    p_m=p_a_mt+RHO_H2O*G_ACC*(h_m-h_0)
    dp_m=dp_a#TODO: uncertainties?????????

    zero_guess=-(np.mean(p_a_mt)/RHO_H2O/G_ACC+A)/B
    dzero_guess=np.sqrt((dp_a/(B*RHO_H2O*G_ACC))**2+(dA/B)**2+((np.mean(p_a_mt)/RHO_H2O/G_ACC+A)/B)**2)
    print(zero_guess)
    print(dzero_guess)


    ############################################################################
    #             PLOTS
    ############################################################################
    #PLOT1
    fig1=plt.figure(figsize=MONO_FIGSIZE)
    ax11=fig1.add_subplot(1,1,1)
    ax11.errorbar(temp[:i_zero],h_m[:i_zero],yerr=dh_m,xerr=dtemp,fmt='b.',label='Temp scendendo')
    ax11.errorbar(temp[i_zero:],h_m[i_zero:],yerr=dh_m,xerr=dtemp,fmt='r.',label='Temp salendo')
    ax12=ax11.twinx()
    ax12.set_ylabel('Pgas [Pa]')
    ax12.errorbar(temp,p_m,yerr=dh_m,xerr=dtemp,fmt='g.',label='Temp salendo')
    ax11.set_xlabel('T [K]')
    ax11.set_ylabel('h [m]')
    ax11.grid()
    ax11_zoom=fig1.add_axes((.7,.2,.15,.2))
    ax11_zoom.errorbar(temp[10],h_m[10],yerr=dh_m,xerr=dtemp)
    ax11_zoom.set_xlabel('T [K]')
    ax11_zoom.set_ylabel('h [m]')
    mark_inset(ax11, ax11_zoom, loc1=1, loc2=3, fc="none", ec="0.5")
    fig1.suptitle('Dipendenza di h da T',fontsize=16)


    #PLOT2
    fig2=plt.figure(figsize=DOUBLE_FIGSIZE)
    ax21=fig2.add_subplot(1,1,1)
    ax21.errorbar(temp[:i_zero],h_m[:i_zero]-A-B*temp[:i_zero],yerr=dh_m,xerr=dtemp,fmt='b.',label='Temp scendendo')
    ax21.errorbar(temp[i_zero:],h_m[i_zero:]-A-B*temp[i_zero:],yerr=dh_m,xerr=dtemp,fmt='r.',label='Temp salendo')
    ax21.set_xlabel('T [K]')
    ax21.set_ylabel('h [m]')
    fig2.suptitle('Dipendenza di h da T',fontsize=16)

    # #PLOT3
    # fig3=plt.figure(figsize=DOUBLE_FIGSIZE)
    # ax31=fig3.add_subplot(1,1,1)
    # ax31.errorbar(temp,p_m,yerr=dp,xerr=dtemp,fmt='b.')
    # ax31.set_xlabel('T [K]')
    # ax31.set_ylabel('h [m]')
    # fig3.suptitle('Dipendenza di h da T',fontsize=16)

    plt.show()
