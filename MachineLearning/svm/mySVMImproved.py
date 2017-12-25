import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
style.use('ggplot')

class Support_Vector_Machine:

    def __init__(self, visualization=True):

        self.currentChoice = 0
        self.currentIsOne = True
        self.stepIntervalRotation = 30.0
        self.constantSteps = 30.0

        self.visualization = visualization
        self.colors = {1:'r', -1:'b', 2:'r', -2:'b', 666:'r', -666:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
            #self.ax = self.fig.add_subplot(111, projection='3d')

    def fit(self, data):
        self.data = data
        opt_dict = {}

        numTransforms = 0
        binTransForms = []
        for i in range(2**len(self.data[next(iter(self.data))][0])):
            bin = "{0:b}".format(numTransforms)
            if len(bin) < len(self.data[next(iter(self.data))][0]):
                for i in range( len(self.data[next(iter(self.data))][0]) - len(bin)):
                    bin = '0'+bin
            binTransForms.append(bin)
            numTransforms += 1

        transformationSets = []
        for i in binTransForms:
            transformSet = []
            for n in i:
                if n == '0':
                    transformSet.append(-1)
                else:
                    transformSet.append(1)
            transformationSets.append(transformSet)

        transforms = transformationSets

        all_data = []
        # for all class in data
        for yi in self.data:
            # for all data in our classes
            for featureset in self.data[yi]:
                # for all features in each data
                for feature in featureset:
                    # all features are added to all data list
                    all_data.append(feature)

        print('ALL DATA:',all_data)

        # get the highest and lowest value from all features
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01]

        # extremely expensive
        b_range_multiple = 5
        # we dont need to take as small steps with b as we do with w
        b_multiple = 5

        lastest_optimum = self.max_feature_value*10.0

        firstStep = True

        # for each step sizes
        for step in step_sizes:


            customDimen = []
            for i in self.data[next(iter(self.data))][0]:
                customDimen.append(lastest_optimum)

            # range in which to search. will turn into -80, 80  to -3.2, 3.2 to -0.8, 0.8 etc
            w = np.array(customDimen)
            print('starting w to increment from', w, 'with steps of',step)

            if firstStep:
                firstStep = False
                checking_list = np.arange(-1 * (lastest_optimum/4.0 * b_range_multiple),
                                          lastest_optimum/4.0 * b_range_multiple,
                                          step * b_multiple)
            else:
                checking_list = np.arange(-1 * (lastest_optimum * 2.0),
                                          lastest_optimum * 2.0,
                                          step * b_multiple)

            # this is where I set the stepping intervals (how vector rotates)
            self.stepIntervalRotation = int(len(checking_list)/4.0)
            self.constantSteps = self.stepIntervalRotation

            # print('checking list', checking_list)
            print('checking list length', len(checking_list))

            # we can do this because convex
            optimized = False
            while not optimized:

                # for each interval
                for b in checking_list:

                    # for each direction
                    for transformation in transforms:

                        # get dot product of width range (80,80) to each transform (1,1), (-1,-1) outputs (-80,80)
                        w_t = w*transformation

                        found_option = True

                        # for all class in data
                        for i in self.data:
                            # for all feature sets in each class

                            for xi in self.data[i]:
                                # yi = current class
                                yi = i

                                if not (yi*(np.dot(w_t, xi)+b))-(abs(yi)) >= 0:

                                    found_option = False

                                    break

                            if found_option == False:

                                break

                        if found_option:

                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]


                notOp = True

                for i in w:
                    if i < 0:
                        optimized = True
                        notOp = False

                if notOp:

                    # step down the ladder algorithm
                    wChoice = []
                    for i in w:
                        wChoice.append(i)

                    wChoice[self.currentChoice] -= step
                    self.stepIntervalRotation = self.stepIntervalRotation - 1
                    if self.stepIntervalRotation == 0:
                        self.stepIntervalRotation = self.constantSteps
                        wChoice[self.currentChoice] = round(wChoice[self.currentChoice-1] - step * int(self.constantSteps / 4.0), 3)
                        lastOne = self.currentChoice
                        while self.currentChoice == lastOne:
                            self.currentChoice = random.randint(0, len(w)-1)

                    w = np.array(wChoice)


            # checks which optimized values was lowests then apply those as the new global values
            norms = sorted( [n for n in opt_dict])
            # print(norms)
            # { ||w||: [w,b] }
            opt_choice = opt_dict[norms[0]]
            print('current best option(svm decision line):', opt_choice)
            print('')
            self.w = opt_choice[0]
            self.b = opt_choice[1]

            # this sets the lastest optimized range value to new optimized value plus a couple steps for safety measures
            lastest_optimum = abs(max(opt_choice[0], key=abs)) + step*2.0

        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, ":", yi * (np.dot(self.w, xi) + self.b))



    def predict(self,features):

        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])

        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplace = x.w+b
        # v = x.w+b
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        #datarange = (self.min_feature_value*0.9, self.max_feature_value * 1.1)
        datarange = (self.min_feature_value, self.max_feature_value)

        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max],[nsv1,nsv2], 'r')

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max],[psv1,psv2], 'g')

        # (w.x+b) = 0
        # divider bound
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max],[db1,db2], 'y--')

        plt.show()


data_dict = {1:np.array([[2,0],[2,2],[2,4],[2,-2]]),
             -1:np.array([[7,3],[4,2],[8,4]])}


svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],[1,3],[3,4],[3,5],[5,5],[5,6],[6,-5],[5,8],[4,-2],[2,0],[0,0],[-1,-5],[8,7],[8,8],[3,-6],[3,0]]
for p in predict_us:
    svm.predict(p)

svm.visualize()

print(svm.w)
print(svm.b)












