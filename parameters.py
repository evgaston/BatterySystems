'''
Module to initialize simulation of EV network
Battery Systems, Final Project
Michaela, Velvet, Michelle
This module constrains the following functions:
createCarsNodes: creates of dataframe of trip Ids and the node id of their home
                 and workplace as well as the number of houses.
convertLoadData: processes load data from 1 minute to 15 minute basis and assoc.
                 it with nodes and vehciles.
SolarGenData: Processes solar data to different capacities
MapCarNodesTime: Finds the charge rate of the vehicles at each time, whether it
                 can charge, where it can charge, and it's home station
netLoadChargeLoc: find the net kw at each charge pt location
findVehicleConsumption: calculates consumption from battery from driving pattern
getSOCconstraints: find minimum SOC requirement to serve driving needs
getChargeRateConstraings: finds the maximum charging rate at each time interval
getInitialSoc: randomly decides vehicles initial SOC so long as minimum
               requirement is met
getAsValues: process as values for regup and reg down
dispatches: creates identity of dispatach signals
'''

# import
import pandas as pd
import random
import numpy as np
import warnings


def createCarsNodes(intNumHouseNodes, intNumCommNodes, intAvgNumCarsNode, intSdNumCarsNode, \
                    intProfiles, dfTripData):

    '''
    Create nodes car trips and assign work and home nodes
    '''

    #initialize data frame to hold master
    dfNodesTrips = pd.DataFrame(columns=['TripId','ResNode','CommNode','HouseID'])
    house=int(0)
    for intNode in range(intNumHouseNodes):

        # find how many vehicles will be at that residential node
        intNodeNumCars = int(random.gauss(intAvgNumCarsNode,intSdNumCarsNode))
        # assign cars to that node
        lsCarIds = [] # container to hold trip id for that node
        lsCommNode = [] # container to hold where the commerical node
        lsResNode = [] # container to hold the res node
        lsVehicleHouses=[] #container to hold the house related to each vehicle/trip
        for intCar in range(intNodeNumCars):

            # find a car from the data frame
            ind = random.randint(0,intProfiles)
            lsCarIds.append(dfTripData['ID'].iloc[ind]) # add to car id container

            # save res node, added one because 1 initialized index was mismached with comm and load nodes
            lsResNode.append(intNode)

            # find which commercial node that car goes to
            lsCommNode.append(random.randint(0,intNumCommNodes-1))

            #add a house id to associate with the trip id
            lsVehicleHouses.append(house)

            #iterate up house id, 0 based indexing to match the array calls/nodes
            house+=int(1)

        # create data frame and add to master
        dfNode = pd.DataFrame(list(zip(lsCarIds,lsResNode,lsCommNode,lsVehicleHouses)), \
                              columns=['TripId','ResNode','CommNode','HouseID'])

        dfNodesTrips = dfNodesTrips.append(dfNode)

    dfNodesTrips.reset_index(inplace=True, drop=True)

    intVeHouses=house

    return dfNodesTrips,intVeHouses


def convertLoadData(dfLoadData,dfNodesTrips,int_run_days,int_run_hours,int_minutes, intVeHouses,intNumCommNodes,intNumHouseNodes):
    '''
    process use data from 1 minute intervals to be on 15 minute interals and for specified run time, and associate with nodes and vehicles
    Each 24 hour time frame goes from 0 (midnight) to 24 (midnight)
    '''
    #initialize time frame to pull
    minutes=60*int_run_hours
    run_length=minutes*int_run_days
    intIntervals=int(run_length/int_minutes)

    #clean dataframe and convert to numpty for manipulation

    dfLoadData = dfLoadData.dropna(axis=1,thresh=2) #pull only nonzeros columns, but all of them have like one so just set a threshhold
    npLoadData=dfLoadData.to_numpy()
    run_loads=npLoadData[0:run_length,2:]    # run lenth extracted from dataset, 130+ loads included, skip timestep/day
    run_loads=run_loads.astype("float32")#I think iteration is faster in an np array?
    #Initialize profile length to pull
    intNumLoads = intVeHouses
    nLoads=len(run_loads[0,:])

    #container for house loads
    arrHouseLoads=np.zeros((intNumLoads, intIntervals))  #loadsxtime
    #process to 15 minutes fgor houses
    for n in range(intNumLoads):
        random_select=random.randrange(0,nLoads,1)
        house_load_1min=run_loads[:,random_select] #1 house one vehicle
        o=int(0)
        j=0
        while o < (run_length-int_minutes):
            start=o
            end=o+int_minutes
            with warnings.catch_warnings():  ##needed to catch exceptions for mean of 0 slices, removed too much data to exclude the columns
                warnings.filterwarnings('error')
            try:
                new_interval=np.nanmean(house_load_1min[start:end])
            except RuntimeWarning:
                new_interval=0 

            arrHouseLoads[n,j]=new_interval
            o+=int_minutes
            j+=1


    #calculate cumulative home nodes
    dfNodeLoads=pd.DataFrame()  #rows will be t, columns will be node ID information, reverse of other df for easy lookup by node
    dctHomeNode=dict() #save node to id mapping, may be useful
    cumulativeLoads=np.zeros((intIntervals))
    i=0 #columns in the node dataframe

    for n in range(intNumHouseNodes):
        #find all the houses that match that Node
        NodeID='ResNode'+str(n)
        lsHomeNode=dfNodesTrips[dfNodesTrips['ResNode']==int(n)].index.tolist()#all home/vehicles that have that resnode associated in dfNodesTrips
        sumrows=arrHouseLoads[lsHomeNode,:]
        cumulativeLoads=np.nansum(sumrows,axis=0)  #this should just be one column
        dctHomeNode[NodeID]=lsHomeNode
        dfNodeLoads.insert(i,NodeID,cumulativeLoads,True)
        i+=1

    arrCommercialLoads=np.zeros((intIntervals,intNumCommNodes))
    #commercial node loads, related to multiple vehicles
    for k in range(intNumCommNodes):  #home x10 for commercial, should replace with real commercial values if possible!
        random_select=random.randrange(0,nLoads,1)
        comm_load_1min=50*run_loads[:,random_select]
        o=int(0)
        j=0
        while o < (run_length-int_minutes):
            start=o
            end=o+int_minutes
            new_interval=np.nanmean(comm_load_1min[start:end])
            arrCommercialLoads[j,k]=new_interval
            o+=int_minutes
            j+=1
        dfNodeLoads.insert(i,"CommNode"+str(k),arrCommercialLoads[:,k],True)
        i+=1

    PeakTimes=dfNodeLoads.idxmax()
    PeakLoads=dfNodeLoads.max()


    lsNodes=dfNodeLoads.columns.values.tolist()
    return   arrHouseLoads,dfNodeLoads, PeakTimes,PeakLoads, lsNodes, dctHomeNode

def SolarGenData(dfGenData,int_run_days,int_run_hours,int_minutes,intVeHouses,\
                intNumCommNodes,intNumHouseNodes, dctHomeNode):
    '''
    Process solar data to different solar capacities for each house (no commercial solar)
    '''
    #gendata is capfac at 15 minute intervals for 24 hours
    run_schedule=int((60/int_minutes)*int_run_hours) #intervals
    intIntervals=int(run_schedule*int_run_days)
    intNumHomeSolar=intVeHouses
    #extract solar data
    npGenData=dfGenData.to_numpy() #I think iteration is faster in an np array?
    npGenData=npGenData[:,2]

    #initial house solar gen container
    arrHouseGen=np.zeros((intNumHomeSolar,intIntervals))
    #every household must average 1 kw generation

    house_sol_scale=1 #kw
    comm_sol_scale=10 #kw
    #total solar gen for each house associated with a vehicle
    for n in range(intNumHomeSolar):
        for d in range(int_run_days):
            start=0
            end=start+run_schedule
            arrHouseGen[n,start:end]=np.multiply(npGenData[:],house_sol_scale)

    arrCommGen=np.zeros((intIntervals,intNumCommNodes))
    #node level generation
    dfNodeGens=pd.DataFrame()
    i=0
    for n in range(intNumHouseNodes):
        NodeID='ResNode'+str(n)
        lsHomeNode=dctHomeNode[NodeID]
        sumrows=arrHouseGen[lsHomeNode,:]

        cumulativesolar=np.nansum(sumrows,axis=0)

        dfNodeGens.insert(i,NodeID,cumulativesolar,True)
        i+=1
    for k in range(intNumCommNodes):  #home x10 for commercial, should replace with real commercial values!
        for d in range(int_run_days):
            start=0
            end=start+run_schedule
            arrCommGen[start:end,k]=npGenData[:]*comm_sol_scale #5 house sum per node
        dfNodeGens.insert(i,"CommNode"+str(k),arrCommGen[:,k],True)
    i+=1

    return arrHouseGen,dfNodeGens,npGenData


def MapCarNodesTime(int_minutes,int_run_hours,int_run_days, intVeHouses,\
                    dfNodesTrips,dfTripData, fltHomeRate, fltWorkRate):
    #make an array that has x vehicles by 96, with location ID, sum of total for each loc
    '''
    Function to find the maximum charging rate at each time interval and whether it is
    available to charge, and the node at which it is present
    '''

    intervals=int((60/int_minutes)*int_run_hours) #intervals for one run
    intIntervals=int(intervals*int_run_days)
    intNumProfs=intVeHouses
    # this will hold the charging rate in kW
    arrChargeRate = np.zeros((intNumProfs, intervals))
    # this will hold the binary of whether the vehicle can charge
    arrCanCharge = np.zeros((intNumProfs, intervals))
    arrChargeLoc = np.empty((intNumProfs, intervals),dtype=object)
    arrHomeStation = np.empty((intNumProfs, intervals),dtype=object) # for stationary battery optimization

    for ind, strId in enumerate(dfNodesTrips['TripId']):

        # get location sequence for that trip
        lsLocations = dfTripData.loc[dfTripData['ID'] == strId].iloc[0,3:-1].values

        # iterate throught the locations to get its charge rate and availability
        for intInterval, strLoc in enumerate(lsLocations):
            if strLoc == 'H': # if at home max charge
                arrChargeRate[ind, intInterval] = fltHomeRate
                arrCanCharge[ind,intInterval] = 1
                arrChargeLoc[ind,intInterval]='ResNode'+str(dfNodesTrips.iloc[ind].loc['ResNode'])
            elif strLoc == 'W': # working charge rate
                arrChargeRate[ind, intInterval] = fltWorkRate
                arrChargeRate[ind,intInterval] = 1
                arrChargeLoc[ind,intInterval]='CommNode'+str(dfNodesTrips.iloc[ind].loc['CommNode'])
            else: # you are moving and can't charge
                arrChargeRate[ind, intInterval] = 0
                arrChargeRate[ind, intInterval] = 0
                arrChargeLoc[ind,intInterval]="Away"

            # just fill with residential node for stationary battery
            arrHomeStation[ind,intInterval] = 'ResNode'+str(dfNodesTrips.iloc[ind].loc['ResNode'])

    x=arrChargeRate
    y=arrCanCharge
    z=arrChargeLoc
    a = arrHomeStation
    for n in range(1,int_run_days):
        arrChargeRate=np.append(arrChargeRate,x,axis=1)
        arrCanCharge=np.append(arrCanCharge,y,axis=1)
        arrChargeLoc=np.append(arrChargeLoc,z,axis=1)
        arrHomeStation = np.append(arrHomeStation, a, axis=1)



    return arrChargeRate, arrCanCharge, arrChargeLoc, arrHomeStation


def netLoadChargeLoc(arrHouseGen,arrHouseLoads,arrChargeLoc,dfNodeLoads,dfNodeGens,\
                     intVeHouses,intTotalNodes,lsNodes,int_minutes,int_run_hours,\
                     int_run_days, arrHomeStation):
    '''
    Find the net kw at each charge pt locations
    '''
    intNumProfs=intVeHouses
    intervals=int(60/int_minutes*int_run_hours)
    intIntervals=int(intervals*int_run_days)
    arrNetNodeLoad=np.zeros((intTotalNodes,intIntervals))
    dfNetNodeLoad=pd.DataFrame()
    #use charge location as values to update to load value, ie charge location=column name,

    #by house/vehicle net load & gen arrays now match for homes
    arrNetHomeLoad=np.subtract(arrHouseLoads,arrHouseGen)

    #total net load per node; aggregated house gen/solar gen and commercial load/solar gen
    dfNetNodeLoad=dfNodeLoads.subtract(dfNodeGens)

    NameDict={}
    dctNodeIdentity=dict()
    dctResIdentity = dict()

    #create identity arrays for the charge locations for each node at each timestep

    for n in lsNodes:

        # fill dictionary with identity
        name=n
        arrayOut=np.zeros((intNumProfs,intIntervals))
        for i in range(intIntervals):
            lsLoc=np.where(arrChargeLoc[:,i]==n)
            arrayOut[lsLoc,i]=1
        dctNodeIdentity[name]=arrayOut

        # if it's a residential node fill res identity dict
        arrRes = np.zeros((intNumProfs, intIntervals))
        if "Res" in name:
            lsCars = np.where(arrHomeStation[:,0] == name)
            arrRes[lsCars,:] = 1
        dctResIdentity[name] = arrRes

    return dfNetNodeLoad,arrNetHomeLoad,dctNodeIdentity, dctResIdentity

def findVehicleConsumption(dfNodesTrips, dfTripData, fltCarEff, fltDt, int_run_days):

    '''
    Function to find the power consumption of the vehicle at each time step
    from driving
    '''

    intNumProfs = dfNodesTrips.shape[0]
    arrConsumption = np.zeros((intNumProfs, 96))

    for ind, strId in enumerate(dfNodesTrips['TripId']):

        # find miles travelled
        fltMiles = dfTripData.loc[dfTripData['ID'] == strId,'DIST_M'].values[0]

        # get a list of locations
        lsLocations = dfTripData.loc[dfTripData['ID'] == strId].values.flatten()[3:-1].tolist()

        intAway = sum(1 if strLoc == "A" else 0 for strLoc in lsLocations)

        # miles per interval of travel and kwh per interval
        fltMiPerInt = fltMiles / intAway
        fltEnergyInt = fltMiPerInt / fltCarEff / fltDt # miles / (miles/kWh) / 0.25 hrs

        lsCarEnergy = [fltEnergyInt*-1 if strLoc == 'A' else 0 for strLoc in lsLocations ]

        arrConsumption[ind,:] = lsCarEnergy

    # adjust for number of days of simulations
    arrConsumptionFinal = arrConsumption
    for ind in range(int_run_days-1):

        arrConsumptionFinal = np.hstack((arrConsumptionFinal, arrConsumption))

    return arrConsumptionFinal

def getSOCconstraints(dfNodesTrips, dfTripData, fltCarEff, fltChargeRate,\
                      fltDt, int_run_days):

    intNumProfs = dfNodesTrips.shape[0]
    arrSOCrequirements = np.zeros((intNumProfs, 96))

    ##this looks like its only treating each car like it has a single travel period, rather than multiple?
    for ind, strId in enumerate(dfNodesTrips['TripId']):

        # find miles travelled
        fltMiles = dfTripData.loc[dfTripData['ID'] == strId,'DIST_M'].values[0]

        # get a list of locations
        lsLocations = dfTripData.loc[dfTripData['ID'] == strId].values.flatten()[3:-1].tolist()

        # so that we can find how many intervals we are away
        intAway = sum(1 if strLoc == "A" else 0 for strLoc in lsLocations)

        # and then find the average miles travelled each interval
        fltAvgMileInterval = fltMiles / intAway

        intStart = 0
        while intStart < 96 and 'A' in lsLocations[intStart:]:

            # Find the first first time we start moving
            intStart = intStart + lsLocations[intStart:].index('A')

            # from there find when we arrive at a destination
            if 'W' in lsLocations[intStart:] and 'H' in lsLocations[intStart:]:
                intEnd = min(lsLocations[intStart:].index('H'), lsLocations[intStart:].index('W'))
            elif 'W' in lsLocations[intStart:]:
                intEnd = lsLocations[intStart:].index('W')
            elif 'H' in lsLocations[intStart:]:
                intEnd = lsLocations[intStart:].index('H')
            else: # we are going to assume they are driving the rest of the time??
                break


            # intervals driving for trip is then (intStart+intEnd +1) - intStart = intEnd =1
            # then we can find miles travelled for first trip
            fltMilesTraveled = fltAvgMileInterval * (intEnd)
            fltKwhRequirement = fltMilesTraveled / fltCarEff # miles / (miles/kWh)

            # now find SOC required
            fltInitialSoc = fltKwhRequirement

            # so we need to to have the initial SOC at the interval before this time:
            arrSOCrequirements[ind,intStart-1] = fltInitialSoc

            # now let's back fill in SOC from there
            intNext = 2
            fltPreviousSOC = fltInitialSoc - fltChargeRate * fltDt
            while fltPreviousSOC > 0:

                # fill in the SOC needed in the previous time series
                arrSOCrequirements[ind, intStart - intNext] = fltPreviousSOC

                intNext += 1 # for next iteration go to the next previous time step
                fltPreviousSOC = fltPreviousSOC - fltChargeRate * fltDt


            intStart += intEnd # increment the starting point for finding the next trip

    # adjust final output for the length of the simulation
    arrSOCrequirementsFinal = arrSOCrequirements
    for ind in range(int_run_days-1):

        arrSOCrequirementsFinal = np.hstack((arrSOCrequirementsFinal, arrSOCrequirements))

    return arrSOCrequirementsFinal


def getInitialSoc(arrSOCconstraint, fltBatteryCap):

    '''
    Function to get an random initialization of the vehicle's SOC
    '''

    intNumVehicles = np.shape(arrSOCconstraint)[0]
    lsInitialSoc = np.zeros((intNumVehicles)) # holder to have the initial SOC of vehicles

    for intVehicle in range(intNumVehicles): # for each vehicle

        fltSocStart = random.random() * fltBatteryCap

        # make sure that the random SOC is > the minimum SOC required
        while fltSocStart < arrSOCconstraint[intVehicle,0]:
            fltSocStart = random.random() * fltBatteryCap

        lsInitialSoc[intVehicle] = fltSocStart

    # must START AND END at that SOC
    arrSOCconstraint[:,0] = lsInitialSoc
    arrSOCconstraint[:,-1] = lsInitialSoc

    return arrSOCconstraint


def processAsValues(dfAsRDdamPrices,dfAsRUdamPrices,dfAsRDrtmPrices,dfAsRUrtmPrices,int_run_days):

    
    #Naming Dict
    

    dfRTMru=pd.DataFrame()
    dfRTMrd=pd.DataFrame()
    dfDAMru=pd.DataFrame()
    dfDAMrd=pd.DataFrame()
    
    #get dataframe for each day
    for days in range(1,31):
        lsRuValue = []
        lsRuHolder=[]
        lsRdValue = []
        lsRdHolder=[]
        dfRuHolder=dfAsRUdamPrices.loc[(dfAsRUdamPrices['GROUP'])==days].sort_values('OPR_HR').reset_index()
        dfRdHolder=dfAsRDdamPrices.loc[(dfAsRDdamPrices['GROUP'])==days].sort_values('OPR_HR').reset_index()
        
        for intHr in range(24):

        # get hourly $/MW price -- convert to 15 minute kWh
            fltUp = dfRuHolder['MW'].iloc[intHr]/1000
            fltDown = dfRdHolder['MW'].iloc[intHr]/1000

            for intInterval in range(4):
                lsRuValue.append(fltUp)
                lsRdValue.append(fltDown)

    # adjust for simulation run period
        for ind in range(int_run_days):

            lsRdHolder+=lsRdValue
            lsRuHolder+=lsRuValue
            
        dfDAMru.insert(days-1,str(days),lsRuHolder,True)
        dfDAMrd.insert(days-1,str(days),lsRdHolder,True)
       # dctRdDAM[days]=lsRdValue
       # dctRuDAM[days]=lsRuValue
    
    for days in range(1,31):
        lsRuValue = []
        lsRuHolder=[]
        lsRdValue = []
        lsRdHolder=[]
        dfRuHolder=dfAsRUrtmPrices.loc[(dfAsRUrtmPrices['GROUP'])==days].sort_values('OPR_TM').reset_index()
        dfRdHolder=dfAsRDrtmPrices.loc[(dfAsRDrtmPrices['GROUP'])==days].sort_values('OPR_TM').reset_index()
        
        for intHr in range(96):

        # get hourly $/MW price -- convert to 15 minute kWh
            fltUp = dfRuHolder['MW'].iloc[intHr]/1000
            fltDown = dfRuHolder['MW'].iloc[intHr]/1000
            
            lsRuValue.append(fltUp)
            lsRdValue.append(fltDown)

        for ind in range(int_run_days):

            lsRdHolder+=lsRdValue
            lsRuHolder+=lsRuValue
            
        dfRTMru.insert(days-1,str(days),lsRuHolder,True)
        dfRTMrd.insert(days-1,str(days),lsRdHolder,True)    
        
    RTMruseries=dfRTMru.sum()
    RTMruMax=dfRTMru.iloc[:,int(RTMruseries.idxmax())-1].values.tolist()  
    RTMrdseries=dfRTMrd.sum()
    RTMrdMax=dfRTMrd.iloc[:,int(RTMrdseries.idxmax())-1].values.tolist()    
    DAMruseries=dfDAMru.sum()
    lsDAMruMax=dfDAMru.iloc[:,int(DAMruseries.idxmax())-1].values.tolist()    
    DAMrdseries=dfDAMrd.sum()
    lsDAMrdMax=dfDAMrd.iloc[:,int(DAMrdseries.idxmax())-1].values.tolist() 
    
        
    lsNetru=np.array(RTMruMax)-np.array(lsDAMruMax)
    lsNetru[lsNetru<0]=0
    lsNetrd=np.array(RTMrdMax)-np.array(lsDAMrdMax)
    lsNetrd[lsNetrd<0]=0
    
    
   
    dctASallprices={'RTMru':dfRTMru,"RTMrd":dfRTMrd,"DAMru":dfDAMru,"DAMrd":dfDAMrd}
      
    return dctASallprices,lsDAMrdMax,lsDAMruMax,lsNetru,lsNetrd



def BatteryDegradationModel ():
    # dummy values -- MUST UPDATE
    #velvet->debugged, but wasnt sure intent on some of these, should this actually be part of a function that is called in the objective function?

    # battery constants
    fltAlphaSei = 5.75 * 10**-2
    fltBetaSei = 121
    fltKdelta1 = 1.40 * 10**5
    fltKdelta2 = -5.01 * 10**-1
    fltKdelta3= 1.23*10**5
    fltKsigma = 1.04
    fltSigmaRef = 0.5
    fltKtime = 4.14* 10**-10 # 1/s
    fltKtemp = 6.93 * 10**-2
    fltTref = 25
    fltCost = 7000 # cost of Tesla battery replacement

    fltTemp=25
    fltDepthofDischarge=0.6
    fltStateofCharge=0.5
    fltTime=3600*int_run_hours # time in seconds


    fltStressTemp=math.exp(fltKtemp*(fltTemp-fltTref))*fltTref/fltTemp
    fltStressSOC=math.exp(fltKsigma*(fltStateofCharge-fltSigmaRef))
    #fltStressTime=fltKtime*fltTime  #not clear what t is supposed to be
    fltStressDischarge=(fltKdelta1*fltDepthofDischarge**fltKdelta2+fltKdelta3)**(-1)

    fltDegradation=(fltStressDischarge)*fltStressSOC*fltStressTemp

    #varBatteryLife=fltAlphaSei*math.exp((-varNumberOfCycles*fltBetaSei*fltDegradation))+(1-fltAlphaSei)*math.exp((-varNumberOfCycles*fltDegradation))
    #need to double check the inputs

    return fltAlphaSei,fltBetaSei,fltDegradation




def dispatches(int_run_days, int_run_hours, int_run_time_interval, fltPctDispatch):

    lsRuPart = []
    lsRdPart = []

    # let's dispatch 10% of the time
    intIntervals = int(int_run_days*int_run_hours*int_run_time_interval)
    intDispatches = int(intIntervals*fltPctDispatch)
    
    if fltPctDispatch==0:
        lsRuIdentity = np.zeros(intIntervals)
        lsRdIdentity = np.zeros(intIntervals)
    elif fltPctDispatch<=.5:  
         for ind in range(intDispatches):
            intRuPart = random.randint(0,intIntervals-1)
            while intRuPart in lsRuPart: # check if it is not in the array already
                intRuPart = random.randint(0,intIntervals-1)
            lsRuPart.append(intRuPart)

            intRdPart = random.randint(0,intIntervals-1)
            while intRdPart in lsRuPart or intRdPart in lsRdPart: # keep pulling until we don't have an interval assigned to up
                intRdPart = random.randint(0,intIntervals-1)
            lsRdPart.append(intRdPart)

    # now make those into a binary array
            lsRuIdentity = [1 if ind in lsRuPart else 0 for ind in range(intIntervals)]
            lsRdIdentity = [1 if ind in lsRdPart else 0 for ind in range(intIntervals)]

    else:  
         for ind in range(intDispatches):
            intRuPart = random.randint(0,intIntervals-1)
            while intRuPart in lsRuPart: # check if it is not in the array already
                intRuPart = random.randint(0,intIntervals-1)
            lsRuPart.append(intRuPart)
            
    # now make those into a binary array
            lsRuIdentity = [1 if ind in lsRuPart else 0 for ind in range(intIntervals)]
            lsRdIdentity = [0 if ind in lsRuPart else 1 for ind in range(intIntervals)]

    return lsRuIdentity, lsRdIdentity

def BatteryDegradationModel ():

    # battery constants
    fltAlphaSei = 5.75 * 10**-2
    fltBetaSei = 121
    fltKdelta1 = 1.40 * 10**5
    fltKdelta2 = -5.01 * 10**-1
    fltKdelta3= 1.23*10**5
    fltKsigma = 1.04
    fltSigmaRef = 0.5
    #fltKtime = 4.14* 10**-10 # 1/s
    #fltKtemp = 6.93 * 10**-2
    #fltTref = 25 
    fltCost = 7000 # cost of Tesla battery replacement

    #fltTemp=25
    #fltDepthofDischarge=0.6
    #fltStateofCharge=0.5
    #fltTime=3600*int_run_hours # time in seconds


    #fltStressTemp=math.exp(fltKtemp*(fltTemp-fltTref))*fltTref/fltTemp
    #fltStressSOC=math.exp(fltKsigma*(fltStateofCharge-fltSigmaRef))
    #fltStressTime=fltKtime*fltTime  #not clear what t is supposed to be 
    #fltStressDischarge=(fltKdelta1*fltDepthofDischarge**fltKdelta2+fltKdelta3)**(-1)

    #fltDegradation=(fltStressDischarge)*fltStressSOC*fltStressTemp

    #varBatteryLife=fltAlphaSei*math.exp((-varNumberOfCycles*fltBetaSei*fltDegradation))+(1-fltAlphaSei)*math.exp((-varNumberOfCycles*fltDegradation))
    #need to double check the inputs

        
    return fltAlphaSei,fltBetaSei,fltKdelta1,fltKdelta2,fltKdelta3,fltKsigma,fltSigmaRef,fltCost