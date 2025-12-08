### 
### This code is developed by Christian P. Janssen of Utrecht University
### It is intended for students from the Master's course Cognitive Modeling
### Large parts are based on the following research papers:
### Janssen, C. P., & Brumby, D. P. (2010). Strategic adaptation to performance objectives in a dual‐task setting. Cognitive science, 34(8), 1548-1560. https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01124.x
### Janssen, C. P., Brumby, D. P., & Garnett, R. (2012). Natural break points: The influence of priorities and cognitive and motor cues on dual-task interleaving. Journal of Cognitive Engineering and Decision Making, 6(1), 5-29. https://journals.sagepub.com/doi/abs/10.1177/1555343411432339
###
### If you want to use this code for anything outside of its intended purposes (training of AI students at Utrecht University), please contact the author:
### c.p.janssen@uu.nl



### 
### import packages
###

import numpy 
import matplotlib.pyplot as plt


###
###
### Global parameters. These can be called within functions to change (Python: make sure to call GLOBAL)
###
###


###
### Car / driving related parameters
###
steeringUpdateTime = 250    #in ms ## How long does one steering update take? (250 ms consistent with Salvucci 2005 Cognitive Science)
timeStepPerDriftUpdate = 50 ### msec: what is the time interval between two updates of lateral position?
startingPositionInLane = 0.27 			#assume that car starts already slightly away from lane centre (in meters) (cf. Janssen & Brumby, 2010)


#parameters for deviations in car drift due the simulator environment: See Janssen & Brumby (2010) page 1555
gaussDeviateMean = 0
gaussDeviateSD = 0.13 ##in meter/sec



### The car is controlled using a steering wheel that has a maximum angle. Therefore, there is also a maximum to the lateral velocity coming from a steering update
maxLateralVelocity = 1.7	# in m/s: maximum lateral velocity: what is the maximum that you can steer?
minLateralVelocity = -1* maxLateralVelocity

startvelocity = 0 	#a global parameter used to store the lateral velocity of the car


###
### Switch related parameters
###
retrievalTimeWord = 200   #ms. ## How long does it take to think of the next word when interleaving after a word (time not spent driving, but drifting)
retrievalTimeSentence = 300 #ms. ## how long does it take to retrieve a sentence from memory (time not spent driving, but drifting)



###
### parameters for typing task
###
timePerWord = 0  ### ms ## How much time does one word take
wordsPerMinuteMean = 39.33   # parameters that control typing speed: when typing two fingers, on average you type this many words per minute. From Jiang et al. (2020; CHI)
wordsPerMinuteSD = 10.3 ## this si standard deviation (Jiang et al, 2020)


## Function to reset all parameters. Call this function at the start of each simulated trial. Make sure to reset GLOBAL parameters.
def resetParameters():
    global timePerWord
    global retrievalTimeWord
    global retrievalTimeSentence 
    global steeringUpdateTime 
    global startingPositionInLane 
    global gaussDeviateMean
    global gaussDeviateSD 
    global gaussDriveNoiseMean 
    global gaussDriveNoiseSD 
    global timeStepPerDriftUpdate 
    global maxLateralVelocity 
    global minLateralVelocity 
    global startvelocity
    global wordsPerMinuteMean
    global wordsPerMinuteSD
    
    timePerWord = 0  ### ms

    retrievalTimeWord = 200   #ms
    retrievalTimeSentence = 300 #ms
	
    steeringUpdateTime = 250    #in ms
    startingPositionInLane = 0.27 			#assume that car starts already away from lane centre (in meters)
	

    gaussDeviateMean = 0
    gaussDeviateSD = 0.13 ##in meter/sec
    gaussDriveNoiseMean = 0
    gaussDriveNoiseSD = 0.1	#in meter/sec
    timeStepPerDriftUpdate = 50 ### msec: what is the time interval between two updates of lateral position?
    maxLateralVelocity = 1.7	# in m/s: maximum lateral velocity: what is the maximum that you can steer?
    minLateralVelocity = -1* maxLateralVelocity
    startvelocity = 0 	#a global parameter used to store the lateral velocity of the car
    wordsPerMinuteMean = 39.33
    wordsPerMinuteSD = 10.3

	



##calculates if the car is not accelerating (m/s) more than it should (maxLateralVelocity) or less than it should (minLateralVelocity)  (done for a vector of numbers)
def velocityCheckForVectors(velocityVectors):
    global maxLateralVelocity
    global minLateralVelocity

    velocityVectorsLoc = velocityVectors

    if (type(velocityVectorsLoc) is list):
            ### this can be done faster with for example numpy functions
        velocityVectorsLoc = velocityVectors
        for i in range(len(velocityVectorsLoc)):
            if(velocityVectorsLoc[i]>1.7):
                velocityVectorsLoc[i] = 1.7
            elif (velocityVectorsLoc[i] < -1.7):
                velocityVectorsLoc[i] = -1.7
    else:
        if(velocityVectorsLoc > 1.7):
            velocityVectorsLoc = 1.7
        elif (velocityVectorsLoc < -1.7):
            velocityVectorsLoc = -1.7

    return velocityVectorsLoc  ### in m/s
	




## Function to determine lateral velocity (controlled with steering wheel) based on where car is currently positioned. See Janssen & Brumby (2010) for more detailed explanation.
## Lateral velocity update depends on current position in lane. Intuition behind function: the further away you are, the stronger the correction will be that a human makes
def vehicleUpdateActiveSteering(LD):
    """
    LD: lane deviation
    how much you steer when actively steering
    """
    latVel = 0.2617 * LD*LD + 0.0233 * LD - 0.022
    returnValue = velocityCheckForVectors(latVel)
    
    if LD > 0: ## Copy-paste Jotan
        returnValue *= -1
    
    return returnValue ### in m/s
	



### function to update lateral deviation in cases where the driver is NOT steering actively (when they are distracted by typing for example). Draw a value from a random distribution. This can be added to the position where the car is already.
def vehicleUpdateNotSteering():
    
    global gaussDeviateMean
    global gaussDeviateSD 

    

    vals = numpy.random.normal(loc=gaussDeviateMean, scale=gaussDeviateSD,size=1)[0]
    returnValue = velocityCheckForVectors(vals)
    return returnValue   ### in m/s





### Function to run a trial. Needs to be defined by students (section 2 and 3 of assignment)

def runTrial(nrWordsPerSentence =5,nrSentences=3,nrSteeringMovementsWhenSteering=2, interleaving="word"): 
    
    resetParameters()
    locPos = []
    currentPos = startingPositionInLane
    trialTime = 0
    locColor = []
    vals = numpy.random.normal(loc=wordsPerMinuteMean, scale=wordsPerMinuteSD,size=1)[0]
    timePerWord = 60000/vals
    print(timePerWord)

    if interleaving == "word":
        locPos.append(currentPos)
        locColor.append("blue")

        for i in range(nrSentences):
            for j in range(nrWordsPerSentence):
                if j == 0:
                    typingTime = retrievalTimeSentence + timePerWord + retrievalTimeWord ## Different than Jotan
                else:
                    typingTime = timePerWord + retrievalTimeWord

                trialTime += typingTime

                numOfUpdates = int(numpy.floor(typingTime/timeStepPerDriftUpdate)) ## Different floor than Jotan

                for k in range(numOfUpdates):
                    drift = vehicleUpdateNotSteering()/20
                    currentPos += drift
                    locPos.append(currentPos)
                    locColor.append("red")

                if not (i == nrSentences - 1 and j == nrWordsPerSentence - 1): ## Different than Jotan
                    for l in range(nrSteeringMovementsWhenSteering):
                        steer = vehicleUpdateActiveSteering(currentPos)
                        for m in range(5): 
                            currentPos += steer/20
                            locPos.append(currentPos)
                            locColor.append("blue")
                        trialTime += steeringUpdateTime

        

        locTime = [] ## Copy-paste Jotan
        time = 0
        for pos in locPos:
            locTime.append(time)
            time += 50


        absPos = [abs(i) for i in locPos]
        return trialTime, numpy.mean(absPos), max(absPos)


    elif interleaving == "sentence":
        locPos.append(currentPos)
        locColor.append("blue")

        for i in range(nrSentences):
            typingTime = retrievalTimeSentence + timePerWord*nrWordsPerSentence

            trialTime += typingTime

            numOfUpdates = int(numpy.floor(typingTime/timeStepPerDriftUpdate))

            for k in range(numOfUpdates):
                drift = vehicleUpdateNotSteering()/20
                currentPos += drift
                locPos.append(currentPos)
                locColor.append("red")

            if not (i == nrSentences - 1): ## Different than Jotan
                for l in range(nrSteeringMovementsWhenSteering):
                    steer = vehicleUpdateActiveSteering(currentPos)
                    for m in range(5): 
                        currentPos += steer/20
                        locPos.append(currentPos)
                        locColor.append("blue")
                    trialTime += steeringUpdateTime

        

        locTime = []
        time = 0
        for pos in locPos:
            locTime.append(time)
            time += 50


        absPos = [abs(i) for i in locPos]
        return trialTime, numpy.mean(absPos), max(absPos)


    elif interleaving == "drivingOnly":
        locPos.append(currentPos)
        locColor.append("blue")
        
        for i in range(nrSentences):
            for j in range(nrWordsPerSentence):
                if j == 0:
                    trialTime += retrievalTimeSentence + timePerWord
                else:
                    trialTime += timePerWord

        for cycle in range(int(trialTime/50)):
            steeringVelocity = vehicleUpdateActiveSteering(currentPos)
            for _ in range(5):                                  # The position needs to be updated every 50 ms
                currentPos += steeringVelocity/20               # /20 bc the function returns lateral speed in m/s, so for 50 ms its that /20
                locPos.append(currentPos)
                locColor.append("blue")

        

        locTime = [] ## Copy-paste Jotan
        time = 0
        for pos in locPos:
            locTime.append(time)
            time += 50
            
        absPos = [abs(i) for i in locPos]
        return trialTime, numpy.mean(absPos), max(absPos)

    
    elif interleaving == "none":
        locPos.append(currentPos)
        locColor.append("red")

        for i in range(nrSentences):
            for j in range(nrWordsPerSentence):
                if j == 0:
                    typingTime = retrievalTimeSentence + timePerWord ## Different than Jotan
                else:
                    typingTime = timePerWord

                trialTime += typingTime

                numOfUpdates = int(numpy.floor(typingTime/timeStepPerDriftUpdate)) ## Different floor than Jotan

                for k in range(numOfUpdates):
                    drift = vehicleUpdateNotSteering()/20
                    currentPos += drift
                    locPos.append(currentPos)
                    locColor.append("red")
        

        locTime = [] ## Copy-paste Jotan
        time = 0
        for pos in locPos:
            locTime.append(time)
            time += 50


        absPos = [abs(i) for i in locPos]
        return trialTime, numpy.mean(absPos), max(absPos)


    elif interleaving == "batch":
        locPos.append(currentPos)
        locColor.append("blue")

        batchFull = 3
        retrievalTimeBatch = 250       # Halfway between retrievalTimeWord and retrievalTimeSentence 

        for sentence in range(nrSentences):
            batch = 0
            for word in range(nrWordsPerSentence):   
                batch += 1
                
                # Calculate how long we take to text
                if word == 0:
                    addtrialTime: float = retrievalTimeSentence + retrievalTimeBatch + timePerWord
                elif batch == 1:
                    addtrialTime: float = retrievalTimeBatch + timePerWord
                else:
                    addtrialTime: float = timePerWord   
                trialTime += addtrialTime

                # For every rounded 50 ms we do one step of drift
                numOfUpdates = int(numpy.floor(addtrialTime/timeStepPerDriftUpdate))
                for _ in range(numOfUpdates):
                    currentPos += vehicleUpdateNotSteering()/20            
                    locPos.append(currentPos)
                    locColor.append("red")
                         
                if batch == batchFull or word == nrWordsPerSentence - 1:
                    if not (sentence == nrSentences - 1 and word == nrWordsPerSentence - 1):
                        for _ in range(nrSteeringMovementsWhenSteering):        
                            steeringVelocity = vehicleUpdateActiveSteering(currentPos)
                            for _ in range(5):                               
                                currentPos += steeringVelocity/20               
                                locPos.append(currentPos)
                                locColor.append("blue")
                                trialTime += 50
                        batch = 0 
        timeList = []
        time = 0
        for i in locPos:
            timeList.append(time)
            time += 50
        absPos = [abs(i) for i in locPos]

        return trialTime, numpy.mean(absPos), max(absPos)


    else:
        return 0

	


### function to run multiple simulations. Needs to be defined by students (section 3 of assignment)
def runSimulations(nrSims = 100):
    totalTime = []
    meanDeviation = []
    maxDeviation = []
    Condition = []
    markers = []
    trackerWord = 0
    trackerSent = 0
    trackerDrive = 0
    trackerNone = 0
    trackerBatch = 0
    timeWord = 0
    timeSent = 0
    timeDrive = 0
    timeNone = 0
    timeBatch = 0
    listWord = []
    listSent = []
    listDrive = []
    listNone = []
    listBatch = []
    listWordTime = []
    listSentTime = []
    listDriveTime = []
    listNoneTime = []
    listBatchTime = []
    nrSentences = 10
    nrSteeringMovementsWhenSteering = 4
    for i in range(nrSims):
        for cond, mark in [("word", "s"), ("sentence", "^"), ("drivingOnly", "p"), ("none", "o"), ("batch", "x")]:
            nrWordsPerSentence = numpy.random.randint(15, 21)
            trialTime, meanDev, maxDev = runTrial(nrWordsPerSentence, nrSentences, nrSteeringMovementsWhenSteering, cond)
            totalTime.append(trialTime)
            meanDeviation.append(meanDev)
            maxDeviation.append(maxDev)
            Condition.append(cond)
            markers.append(mark)
            
    
    for j in range(len(totalTime)):
        plt.scatter(totalTime[j], maxDeviation[j], marker = markers[j], c ="gray")
        if markers[j] == "s":
            trackerWord += maxDeviation[j]
            timeWord += totalTime[j]
            listWord.append(maxDeviation[j])
            listWordTime.append(totalTime[j])
        elif markers[j] == "^":
            trackerSent += maxDeviation[j]
            timeSent += totalTime[j]
            listSent.append(maxDeviation[j])
            listSentTime.append(totalTime[j])
        elif markers[j] == "p":
            trackerDrive += maxDeviation[j]
            timeDrive += totalTime[j]
            listDrive.append(maxDeviation[j])
            listDriveTime.append(totalTime[j])
        elif markers[j] == "o":
            trackerNone += maxDeviation[j]
            timeNone += totalTime[j]
            listNone.append(maxDeviation[j])
            listNoneTime.append(totalTime[j])
        elif markers[j] == "x":
            trackerBatch += maxDeviation[j]
            timeBatch += totalTime[j]
            listBatch.append(maxDeviation[j])
            listBatchTime.append(totalTime[j])
    

    listOfMeans = [trackerWord/nrSims, trackerSent/nrSims, trackerDrive/nrSims, trackerNone/nrSims, trackerBatch/nrSims]
    listOfTimes = [timeWord/nrSims, timeSent/nrSims, timeDrive/nrSims, timeNone/nrSims, timeBatch/nrSims]
    listOfStdDevs = [numpy.std(listWord), numpy.std(listSent), numpy.std(listDrive), numpy.std(listNone), numpy.std(listBatch)]
    listOfStdTimes = [numpy.std(listWordTime), numpy.std(listSentTime), numpy.std(listDriveTime), numpy.std(listNoneTime), numpy.std(listBatchTime)]
   
    condition_colors = ["blue", "red", "green", "black", "yellow"]
    condition_labels = ["word", "sentence", "drivingOnly", "none", "batch"]
    condition_markers = ["s", "^", "p", "o", "x"]
    
    for k in range(len(listOfMeans)):
        plt.scatter(listOfTimes[k], listOfMeans[k], marker=condition_markers[k], c=condition_colors[k], s=200, label=condition_labels[k])
        plt.errorbar(listOfTimes[k], listOfMeans[k], xerr = listOfStdTimes[k], yerr = listOfStdDevs[k], c=condition_colors[k])
        
    plt.ylabel("Maximum Lateral Deviation (m)")
    plt.xlabel("Trial Time (ms)")
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.legend()
    plt.show()

        
runTrial()


	




