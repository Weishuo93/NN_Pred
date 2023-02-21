description = """ Plot the pressure samples."""
import os, sys
import math
from matplotlib import pylab
from pylab import *
from numpy import loadtxt
def addToPlots( timeName ):
    fileName = "postProcessing/singleGraph/" + timeName + "/line_U.xy"
    i=[]
    time=[]
    abc =loadtxt(fileName)
    for z in abc:
        time.append(z[0])
        i.append(z[1])
    legend = "Velocity at " + timeName
    plot(time,i,label="Time " + timeName )

def contrastPlots( caseName, timeName ):
    fileName = caseName + "/postProcessing/singleGraph/" + timeName + "/line_U.xy"
    i=[]
    time=[]
    abc =loadtxt(fileName)
    for z in abc:
        time.append(z[0])
        i.append(z[1])
    legend = "Velocity at " + caseName
    plot(time,i,label="Case " + caseName )

def contrastaijPlots( caseName, timeName,ncomponent ):
    fileName = caseName + "/postProcessing/singleGraph/" + timeName + "/line_aij.xy"
    x=[]
    y=[]
    for i in range(0,6):
        y.append([])
    abc =loadtxt(fileName)
    d = {"11":1,"12":2,"13":3,"22":4,"23":5,"33":6}
    for z in abc:
        x.append(z[0])
        y[0].append(z[1])
        y[1].append(z[2])
        y[2].append(z[3])
        y[3].append(z[4])
        y[4].append(z[5])
        y[5].append(z[6])
    legend = "a"+ncomponent+ " at " + caseName
    ylabel(" a" +ncomponent + " "); xlabel(" y+ "); title(" Reynolds Stress Anisotropic Tensor a" + ncomponent)
    bcd = loadtxt(caseName + "/postProcessing/singleGraph/" + timeName + "/line_U.xy")
    U=[]
    xp=[]
    for tt in bcd:
        U.append(tt[1])
    utao=sqrt((U[1]-U[0])/(x[1]-x[0])*2e-05)
    for xx in x:
        xp.append(xx*utao/2e-05)
    plot(xp,y[d[ncomponent]-1][:],label="Case " + caseName )

def contrastupypPlots( caseName, timeName ):
    fileName = caseName + "/postProcessing/singleGraph/" + timeName + "/line_U.xy"
    i=[]
    time=[]
    abc =loadtxt(fileName)
    for z in abc:
        time.append(z[0])
        i.append(z[1])
    utao=sqrt((i[1]-i[0])/(time[1]-time[0])*2e-05)
    up=[]
    yp=[]
    for u in i:
        up.append(u/utao)
    for y in time:
        yp.append(y*utao/2e-05)
    legend = "U+ at " + caseName
    plot(yp,up,label="Case " + caseName )
    ylabel(" u+ "); xlabel(" y+ "); title(" Velocity Profile")
    plt.xscale('log')

def contrastaijypPlots( caseName, timeName,ncomponent ):
    fileName = caseName + "/postProcessing/singleGraph/" + timeName + "/line_aij.xy"
    x=[]
    y=[]
    xp=[]
    for i in range(0,6):
        y.append([])
    abc =loadtxt(fileName)
    d = {"11":1,"12":2,"13":3,"22":4,"23":5,"33":6}
    for z in abc:
        x.append(z[0])
        y[0].append(z[1])
        y[1].append(z[2])
        y[2].append(z[3])
        y[3].append(z[4])
        y[4].append(z[5])
        y[5].append(z[6])
    legend = "a"+ncomponent+ " at " + caseName
    for tt in x:
        xp.append(tt*365)
    plot(xp,y[d[ncomponent]-1][:],label="Case " + caseName )
    ylabel(" a" +ncomponent + " "); xlabel(" y+ "); title(" Reynolds Stress Anisotropic Tensor a" + ncomponent) 
    plt.xscale("log")

def addBetaToPlots( timeName ):
    fileName = "postProcessing/singleGraph/" + timeName + "/line_Beta1_.xy"
    i=[]
    time=[]
    abc =loadtxt(fileName)
    for z in abc:
        time.append(z[0])
        i.append(z[1])
    legend = "Beta at " + timeName
    plot(time,i,label="Time " + timeName )

def addaijToPlots( timeName ):
    fileName = "postProcessing/singleGraph/" + timeName + "/line_aij.xy"
    i1=[]
    i2=[]
    time=[]
    abc =loadtxt(fileName)
    for z in abc:
        time.append(z[0])
        i1.append(z[1])
        i2.append(z[2])
    legend = "Aij at " + timeName
    plot(time,i2,label="Time " + timeName )

def contrastkyPlots( caseName, timeName ):
    fileName = caseName + "/postProcessing/singleGraph/" + timeName + "/line_k.xy"
    i=[]
    time=[]
    abc =loadtxt(fileName)
    for z in abc:
        time.append(z[0])
        i.append(z[1])
    legend = "k at " + caseName
    plot(time,i,label="Case " + caseName )
    ylabel(" k "); xlabel(" y "); title(" TKE ")

def contrastuvPlots( caseName, timeName ):
    fileName = caseName + "/postProcessing/singleGraph/" + timeName + "/line_R.xy"
    i=[]
    x=[]
    abc =loadtxt(fileName)
    for z in abc:
        x.append(z[0])
        i.append(z[2])
    legend = "uv at " + caseName
    plot(x,i,label="Case " + caseName )
    ylabel(" uv "); xlabel(" y "); title(" uv ")


def addvarToPlots( timeName ,varName,ncomp):
    fileName = "postProcessing/singleGraph/" + timeName + "/line_" + varName + ".xy"
    var=[]
    y=[]
    abc =loadtxt(fileName)
    for z in abc:
        y.append(z[0])
        var.append(z[ncomp])
    legend = varName + " at " + timeName
    plot(y,var,label= varName + "Time " + timeName )


ncomp="33"
figure(1);
#ylabel(" a"+ncomp); xlabel(" y/H "); title(" Reynold Stress Anisotropic Tensor a"+ncomp)i
#ylabel(" Velocity (m/s)"); xlabel(" y/H "); title(" Velocity Profile")

grid()
#contrastPlots( "ML-EARSM-Channel395","1000")
#contrastPlots( "Original-EARSM-Channel395","1000")
#contrastupypPlots( "ML-EARSM-Channel395","400")
#contrastupypPlots( "ML_EARSM_CASE","800")
#contrastupypPlots( "Original-EARSM","1084.6")
#contrastuvPlots("kOmegaSST","2000")
#contrastuvPlots("kOmegaSST_Mapped","20")


#contrastaijPlots("Original-EARSM","1084.6","12")
#contrastaijPlots("ML_EARSM_CASE","800","12")
# #hold(True)
#for dirStr in os.listdir("postProcessing/singleGraph/"):
#    addvarToPlots(dirStr,'nut',1)
# #addToPlots("100")
#for strr in ['0','2000']:
#  addToPlots(strr)

addvarToPlots( '200', 'nut_U', 1)
addvarToPlots( '5000', 'nut_U', 1)
#addvarToPlots( '58000', 'nut_U', 1)
#addvarToPlots( '60000', 'nut_U', 1)
#addvarToPlots( '62000', 'nut_U', 1)
# #for dirStr in os.listdir("postProcessing/singleGraph/"):
# addBetaToPlots('0')
# addBetaToPlots('100')
# addBetaToPlots('200')
# addBetaToPlots('300')
# addBetaToPlots('400')
# addBetaToPlots('500')
# #addBeta2ToPlots('0')
# #contrastPlots("KEtestsimple", "5000")
# #contrastPlots("EARSM_testsimple", "5000")
# #contrastPlots("KOMGSSTtestsimple", "140")
legend(loc="best")
# savefig("myPlot.png")
show() #Problems with ssh
