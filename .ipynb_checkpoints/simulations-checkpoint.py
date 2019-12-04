'''
Module to simplify optimization simulations 
Battery Systems, Final Project
Michaela, Velvet, Michelle

make_dispatch_constraints & _objective: calculates an optimization with a varying amount of dispatch, DAM money is received for capacity available and RTM value is received for dispatch if greater than DAM. 

make_tou_constraints & objective: calculates an optimization to minimize TOU cost for vehicle charging. This is a no services scenario, "every day behavior" of an ev

make_battery_constraints & objective: calculates the optimization as if the vehicle is a stationary battery providing services 100% of the time. 


'''



import pandas as pd
import random
import scipy as sp
import numpy as np
import cvxpy as cvx
import mosek
import matplotlib.pyplot as plt
import math

def make_dispatch_constraints(arrSOCconstraint, arrChargeRate, arrCanCharge,dctNodeIdentity,lsNodes, \
                     arrConsumptionFinal,PeakLoads,intTotalNodes,dfNetNodeLoad,\
                     fltBatteryCap,fltDt,lsCostElectric,\
                     lsRuIdentity, lsRdIdentity):
    
    intVehicles = np.shape(arrSOCconstraint)[0]
    intTimes = np.shape(arrSOCconstraint)[1]
    
    # make variables
    varCharge = cvx.Variable((intVehicles,intTimes)) # charging rate of vehicles
    varDischarge = cvx.Variable((intVehicles,intTimes)) # discharging rate of vehicles
    varSOCs = cvx.Variable((intVehicles,intTimes)) # SOCs of the vehicles
    
    varNodeSum=cvx.Variable((intTotalNodes, intTimes))
    
    # define regulation up and down variables
    varRegUp = cvx.Variable((np.shape(arrCanCharge))) #this array is # vehiclesxtimesteps
    varRegDown = cvx.Variable((np.shape(arrCanCharge)))
    varNumberOfCycles=cvx.Variable(intVehicles)
    varBatteryLife=cvx.Variable((intVehicles)) #battery life of the vehicle remaining
    varDegradationCost=cvx.Variable(intVehicles)
 
    # initialzie constraints
    constraints = []
    
    # define the charge rate rate constraints
    constraints += [varCharge >= 0] # positively defined
    constraints += [varDischarge <= 0] # negatively defined
    constraints += [varSOCs >= 0] # can't have negative SOC
    constraints += [varSOCs <= fltBatteryCap] # can't over charge
    constraints += [varNodeSum >= 0]
    constraints += [varRegUp >= 0]
    constraints += [varRegDown >= 0]
    constraints += [varNumberOfCycles >= 0]
    
    for v in range(intVehicles):
        
        # ensure we still meet our battery demand through regular charging, this may not be relevant since reg down could supply?
        constraints += [(cvx.sum(varCharge[v,:]) + cvx.sum(varDischarge[v,:]))*fltDt + \
                        cvx.sum(varRegDown[v,:]*lsRdIdentity)*fltDt/15 - cvx.sum(varRegUp[v,:]*lsRuIdentity)*fltDt/15 \
                                 >= cvx.sum(arrConsumptionFinal[v,:])*fltDt]           
        
        # loop through each time to model charging and SOC constraints
        for t in range(intTimes):
            
            # can only charge/discharge if present and max rate is given by array
            constraints += [varCharge[v, t] <= arrChargeRate[v,t]*arrCanCharge[v, t]]
            constraints += [-1*arrChargeRate[v,t]*arrCanCharge[v, t] <= varDischarge[v,t]]     
            
            # RegUp and RegDown Definition
            #can only charge up to max charge rate less whatever was needed to charge the vehicle 
            constraints += [varRegDown[v,t] <= (arrChargeRate[v,t] - varCharge[v,t])*arrCanCharge[v,t]]
            # or what is left to charge batteries capacity
            constraints += [varRegDown[v,t] <= ((fltBatteryCap - varSOCs[v,t])/fltDt \
                                                            - varCharge[v,t])*arrCanCharge[v,t]]
            
            # reg up constraints 
            # it is at most the charge rate if it's avalable
            constraints += [varRegUp[v,t] <= (arrChargeRate[v,t] + varDischarge[v,t])*arrCanCharge[v,t]]
            # or what it's minimum SOC requriement is
            constraints += [varRegUp[v,t] <= ((varSOCs[v,t] - arrSOCconstraint[v,t])/fltDt \
                                                          + varDischarge[v,t])*arrCanCharge[v,t]]
            
            # vehicle's SOC must be greater than the minimum requirement
            constraints += [varSOCs[v,t] >= arrSOCconstraint[v,t]]
           
        # must end at the same SOC it started at 
        constraints += [varSOCs[v,0] == varSOCs[v,-1]]
        #has to initialize, option to pass ending values
        #onstraints += [varSOCs[v,0]== varInit[v]]
            
        # SOC function  
        for t in range(1,intTimes): # loop through from time 1 to end to set SOC, 
            
            constraints += [varSOCs[v,t]  == varSOCs[v,t-1] + fltDt*(varCharge[v,t] + varDischarge[v,t] \
                                                                     + arrConsumptionFinal[v,t])\
                                                + varRegDown[v,t]*lsRdIdentity[t]*fltDt/15  \
                                                - varRegUp[v,t]*lsRuIdentity[t]*fltDt/15]

    for n, node in enumerate(lsNodes):
        
        # find the transformer size 
        TransformerMax = PeakLoads[node]*1.5  # max is 20% above max load
        # load has to be less than that 
        constraints += [varNodeSum[n,:] <= TransformerMax]
        
        # gather node identities 
        arrNodeIdentities= dctNodeIdentity[node]
        for t in range (intTimes):
            constraints+= [varNodeSum[n,t] == cvx.sum(arrNodeIdentities[:,t] * (varCharge[:,t] \
                                    + varDischarge[:,t] - varRegDown[:,t]*lsRdIdentity[t]\
                                    + varRegUp[:,t]*lsRuIdentity[t]) \
                                                      + dfNetNodeLoad[node].iloc[t])]
            
    for v in range(intVehicles):
        
        # set up number of cycles based on the sum of all charge & discharge including reg up and down 
        constraints+= [varNumberOfCycles[v] >= ((cvx.sum(varCharge[v,:]) - cvx.sum(varDischarge[v,:]) \
                                                             - cvx.sum(arrConsumptionFinal[v,:]))*fltDt \
                                                + (cvx.sum(varRegDown[v,:]*lsRdIdentity) \
                                                     + cvx.sum(varRegUp[v,:]*lsRuIdentity))*fltDt/15)/fltBatteryCap]
        
     
        constraints+= [varBatteryLife[v] == (6.015*(.000017*varNumberOfCycles[v]))]
        
        constraints+= [varDegradationCost[v] == 7000*varBatteryLife[v]/0.2]
       
   
    return constraints, varRegDown,varRegUp,varCharge,varNumberOfCycles, varDegradationCost, varSOCs

def make_dispatch_objectives(constraints, varRegDown,varRegUp,varCharge,varDegradationCost,\
                    lsCostElectric,lsRdIdentity, lsRuIdentity,lsDAMrdMax,lsDAMruMax,lsNetru,lsNetrd,fltDt):
    
    
    obj_value = cvx.sum(lsDAMrdMax*cvx.sum(varRegDown,axis=0)) + \
                cvx.sum(lsDAMruMax*cvx.sum(varRegUp,axis=0)) + \
                cvx.sum(lsNetrd*cvx.sum(varRegDown,axis=0)*lsRdIdentity/15)+\
                cvx.sum(lsNetru*cvx.sum(varRegUp,axis=0)*lsRuIdentity/15)- \
                cvx.sum(cvx.multiply(lsCostElectric,cvx.sum(varCharge,axis=0))*fltDt) - \
                cvx.sum(varDegradationCost) -\
                    cvx.sum(lsCostElectric*cvx.sum(varRegDown,axis=0)*lsRdIdentity*fltDt/15)
    
    return constraints, obj_value, varRegUp, varRegDown,varCharge, varDegradationCost

def make_tou_constraints(arrSOCconstraint, arrChargeRate, arrCanCharge,\
                     fltDt, fltBatteryCap, arrConsumptionFinal, PeakLoads,intTotalNodes, dfNetNodeLoad,lsNodes,dctNodeIdentity):
    

    intVehicles = np.shape(arrSOCconstraint)[0]
    intTimes = np.shape(arrSOCconstraint)[1]
    
    # make variables
    varCharge = cvx.Variable((intVehicles,intTimes)) # charging rate of vehicles
    # no need for discharge variable
    varSOCs = cvx.Variable((intVehicles,intTimes)) # SOCs of the vehicles
    varNodeSum = cvx.Variable((intTotalNodes, intTimes))
    
    
    # initialzie constraints
    constraints = []
    
    # define the charge rate rate constraints
    constraints += [varCharge >= 0] # positively defined
    constraints += [varSOCs >= 0] # can't have negative SOC
    constraints += [varSOCs <= fltBatteryCap] # can't over charge
    constraints += [varNodeSum >= 0]
      
    # define charging limits for each vehicle at each time
    for v in range(intVehicles):
        
        # need to have energy used restored over the entire time period
        constraints += [cvx.sum(varCharge[v,:])*fltDt >= cvx.sum(arrConsumptionFinal[v,:])*fltDt]           
        
      # loop through each time to model charging and SOC constraints
        for t in range(intTimes):
            
            # can only charge if present and max rate is given by array
            constraints += [varCharge[v, t] <= arrChargeRate[v,t]*arrCanCharge[v, t]]
            
            # vehicle's SOC must be greater than the minimum requirement
            constraints += [varSOCs[v,t] >= arrSOCconstraint[v,t]]
        
        # must end at the same SOC it started at 
        constraints += [varSOCs[v,0] == varSOCs[v,-1]]
        
        for t in range(1,intTimes): # loop through from time 1 to end to set SOC
            
            constraints += [varSOCs[v,t]  == varSOCs[v,t-1] + varCharge[v,t]*fltDt + arrConsumptionFinal[v,t]*fltDt]
    
        for n, node in enumerate(lsNodes):
        
        # find the transformer size 
            TransformerMax = PeakLoads[node]*1.5  # max is 20% above max load
        # load has to be less than that 
            constraints += [varNodeSum[n,:] <= TransformerMax]
        
        # gather node identities 
            arrNodeIdentities= dctNodeIdentity[node]
            for t in range (intTimes):
                constraints+= [varNodeSum[n,t] == cvx.sum(arrNodeIdentities[:,t] * varCharge[:,t] + dfNetNodeLoad[node].iloc[t])]
        
    
    return constraints, varCharge, varSOCs

def make_tou_objectives(constraints, varSOCs,varCharge,arrSOCconstraint,fltDt,fltBatteryCap,lsCostElectric,arrConsumptionFinal):
    
    
    intVehicles = np.shape(arrSOCconstraint)[0]
    intTimes = np.shape(arrSOCconstraint)[1]
    
    varCostToCharge = cvx.Variable(intTimes) # payments to charge vehicle
    varNumberOfCycles = cvx.Variable(intVehicles)
    varBatteryLife = cvx.Variable(intVehicles)
    varDegradationCost = cvx.Variable(intVehicles)
    
    constraints += [varDegradationCost >= 0] #positively constrained 
    constraints += [varNumberOfCycles >= 0]
    
    for v in range(intVehicles):
        constraints+= [varNumberOfCycles[v] == (cvx.sum(varCharge[v,:])-np.sum(arrConsumptionFinal[v,:]))*fltDt/fltBatteryCap]
        constraints+= [varBatteryLife[v] == (6.015*(.000017*varNumberOfCycles[v]))]
        
        constraints+= [varDegradationCost[v] == 7000*varBatteryLife[v]/0.2]
     
    for t in range(intTimes):
        
        constraints += [varCostToCharge[t] == cvx.sum(varCharge[:,t])*lsCostElectric[t]*fltDt]

    obj_value = cvx.sum(varCostToCharge) + cvx.sum(varDegradationCost)
    
    return constraints, obj_value, varCharge,varNumberOfCycles, varDegradationCost

def make_battery_constraints(arrSOCconstraint, arrChargeRate, arrCanCharge,dctResIdentity,lsResNodes, \
                     arrConsumptionFinal,PeakLoads,intTotalNodes,dfNetNodeLoad,\
                     fltBatteryCap,fltDt,lsCostElectric,\
                     lsRuIdentity, lsRdIdentity, fltWorkRate):
    
    intVehicles = np.shape(arrSOCconstraint)[0]
    intTimes = np.shape(arrSOCconstraint)[1]
    
    # make variables
    varCharge = cvx.Variable((intVehicles,intTimes)) # charging rate of vehicles
    varDischarge = cvx.Variable((intVehicles,intTimes)) # discharging rate of vehicles
    varSOCs = cvx.Variable((intVehicles,intTimes)) # SOCs of the vehicles
    varBatteryLife=cvx.Variable((intVehicles)) #battery life of the vehicle remaining
    varNodeSum=cvx.Variable((intTotalNodes, intTimes))
    
    # define regulation up and down variables
    varRegUp = cvx.Variable((np.shape(arrCanCharge))) #this array is # vehiclesxtimesteps
    varRegDown = cvx.Variable((np.shape(arrCanCharge)))
    varNumberOfCycles=cvx.Variable(intVehicles)
    varBatteryLife = cvx.Variable(intVehicles)
    varDegradationCost=cvx.Variable(intVehicles)
 
    # initialzie constraints
    constraints = []
    
    # define the charge rate rate constraints
    constraints += [varCharge >= 0] # positively defined
    constraints += [varDischarge <= 0] # negatively defined
    constraints += [varCharge <= fltWorkRate]
    constraints += [varSOCs >= 0] # can't have negative SOC
    constraints += [varDischarge >= -1*fltWorkRate]
    constraints += [varSOCs <= fltBatteryCap] # can't over charge
    constraints += [varNodeSum >= 0]
    constraints += [varRegUp >= 0]
    constraints += [varRegDown >= 0]
    constraints += [varDegradationCost >= 0] #positively constrained 
    constraints += [varNumberOfCycles >= 0]
    
    for v in range(intVehicles):
              
        # loop through each time to model charging and SOC constraints
        for t in range(intTimes):
             
            # RegUp and RegDown Definition
            #can only charge up to max charge rate less whatever was needed to charge the vehicle 
            constraints += [varRegDown[v,t] <= (fltWorkRate - varCharge[v,t])]
            # or what is left to charge batteries capacity
            constraints += [varRegDown[v,t] <= ((fltBatteryCap - varSOCs[v,t])/fltDt - varCharge[v,t])]
            
            # reg up constraints 
            # it is at most the charge rate if it's avalable
            constraints += [varRegUp[v,t] <= (arrChargeRate[v,t] + varDischarge[v,t])]
            # or what it's minimum SOC requriement is
            constraints += [varRegUp[v,t] <= varSOCs[v,t]/fltDt + varDischarge[v,t]]
              
        # must end at the same SOC it started at 
        constraints += [varSOCs[v,0] == varSOCs[v,-1]]
            
        # SOC function  
        for t in range(1,intTimes): # loop through from time 1 to end to set SOC, 
            
            constraints += [varSOCs[v,t]  == varSOCs[v,t-1] + fltDt*(varCharge[v,t] + varDischarge[v,t])\
                                                + varRegDown[v,t]*lsRdIdentity[t]*fltDt/15  \
                                                - varRegUp[v,t]*lsRuIdentity[t]*fltDt/15]

     
    # residential node constraints        
    for n, node in enumerate(lsResNodes):
        
        # find the transformer size 
        TransformerMax=PeakLoads[node]*1.5  #max is 20% above max load
        # load has to be less than that 
        constraints+= [varNodeSum[n,:] <= TransformerMax]
        
        # gather node identities 
        arrNodeIdentities= dctResIdentity[node]
        for t in range (intTimes):
            constraints+= [varNodeSum[n,t] == cvx.sum(arrNodeIdentities[:,t] * (varCharge[:,t] \
                                    + varDischarge[:,t] - varRegDown[:,t]*lsRdIdentity[t]\
                                    + varRegUp[:,t]*lsRuIdentity[t]) \
                                                      + dfNetNodeLoad[node].iloc[t])]
    
    for v in range(intVehicles):
        
        # set up number of cycles based on the sum of all charge & discharge including reg up and down 
        
        # set up number of cycles based on the sum of all charge & discharge including reg up and down 
        constraints+= [varNumberOfCycles[v] >= ((cvx.sum(varCharge[v,:]) - cvx.sum(varDischarge[v,:]))*fltDt \
                                                + (cvx.sum(varRegDown[v,:]*lsRdIdentity) \
                                                     + cvx.sum(varRegUp[v,:]*lsRuIdentity))*fltDt/15)/fltBatteryCap]
        
     
        constraints+= [varBatteryLife[v] == (6.015*(.000017*varNumberOfCycles[v]))]
        
        constraints+= [varDegradationCost[v] == 7000*varBatteryLife[v]/0.2]
       
             
   
    return constraints, varRegDown,varRegUp,varCharge,varNumberOfCycles, varDegradationCost

def make_battery_objectives(constraints, varRegDown,varRegUp,varCharge,varDegradationCost,\
                    lsCostElectric,lsRdIdentity, lsRuIdentity,lsDAMrdMax,lsDAMruMax,lsNetru,lsNetrd,fltDt):
    
        
    obj_value = cvx.sum(lsDAMrdMax*cvx.sum(varRegDown,axis=0)) + \
                cvx.sum(lsDAMruMax*cvx.sum(varRegUp,axis=0)) - \
                cvx.sum(lsNetrd*cvx.sum(varRegDown,axis=0)*lsRdIdentity/15)+\
                cvx.sum(lsNetru*cvx.sum(varRegUp,axis=0)*lsRuIdentity/15)- \
                cvx.sum(lsCostElectric*cvx.sum(varCharge,axis=0)*fltDt) - \
                cvx.sum(lsCostElectric*cvx.sum(varRegDown,axis=0)*lsRdIdentity*fltDt/15) -\
                cvx.sum(varDegradationCost)
    
    return constraints, obj_value, varRegUp, varRegDown,varCharge, varDegradationCost
