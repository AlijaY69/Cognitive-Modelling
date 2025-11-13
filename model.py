from matplotlib import pyplot as plt

def start():
    return 0

def perceptualstep(type = "middle"):
    if type == "slow":
        return 200
    elif type == "fast":
        return 50
    else:
        return 100

def cognitivestep(type = "middle"):
    if type == "slow":
        return 170
    elif type == "fast":
        return 25
    else:
        return 70

def motorstep(type = "middle"):
    if type == "slow":
        return 100
    elif type == "fast":
        return 30
    else:
        return 70


def example1():
    return start() + perceptualstep() + cognitivestep() + motorstep()


def example2(completeness = "extremes"):
    answer = []
    if completeness == "extremes":
        fast: int = start() + perceptualstep("fast") + cognitivestep("fast") + motorstep("fast")
        middle: int =  start() + perceptualstep() + cognitivestep() + motorstep()
        slow: int =  start() + perceptualstep("slow") + cognitivestep("slow") + motorstep("slow")

        return fast, middle, slow
    
    elif completeness == "all":
        for typep in ["fast", "middle", "slow"]:
            for typec in ["fast", "middle", "slow"]:
                for typem in ["fast", "middle", "slow"]:
                    time = start() + perceptualstep(typep) + cognitivestep(typec) + motorstep(typem)
                    print(start() + perceptualstep(typep) + cognitivestep(typec) + motorstep(typem))
                    answer.append(time)
        plt.boxplot(answer)
        plt.ylabel("time (ms)")
        plt.ylim(0, max(answer) + 100)
        plt.show()



##example2("all")


def example3():
    print(start() + perceptualstep("fast") + perceptualstep("fast") + cognitivestep("middle") + cognitivestep("middle") + motorstep("slow"))




def example4():
    slowest: int = 0
    for time in [40, 80, 110, 150, 210, 240]:
        for typep in ["fast", "middle", "slow"]:
            for typec in ["fast", "middle", "slow"]:
                for typem in ["fast", "middle", "slow"]:
                    if time > perceptualstep(typep):
                        ans = start() + time + perceptualstep(typep) + cognitivestep(typec) + cognitivestep(typec) + motorstep(typem)
                        if ans > slowest:
                            slowest = ans
                    else:
                        ans = start() + perceptualstep(typep) + perceptualstep(typep) + cognitivestep(typec) + cognitivestep(typec) + motorstep(typem)
                        if ans > slowest:
                            slowest = ans
    print(slowest)

example4()

                    

def example5():
    times = []
    errors = []
    error: int = 0.01
    for typep in ["fast", "middle", "slow"]:
        for typec in ["fast", "middle", "slow"]:
            for typem in ["fast", "middle", "slow"]:
                if typep == "fast" or typec == "fast":
                    error *= 3
                
                
                ans = start() + perceptualstep(typep) + perceptualstep(typep) + cognitivestep(typec) + cognitivestep(typec) + motorstep(typem)
                errors.append(error)
                times.append(ans)
    
    plt.scatter(times, errors)
    plt.show()
    print(errors)

example5()