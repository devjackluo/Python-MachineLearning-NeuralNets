import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import random
style.use('ggplot')

class Support_Vector_Machine:

    def __init__(self, visualization=True):

        self.currentIsOne = True
        self.stepIntervalRotation = 30
        self.constantSteps = 30

        self.visualization = visualization
        self.colors = {1:'r', -1:'b', 2:'r', -2:'b', 666:'r', -666:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        # each direction vector can go.
        # transforms = [[1,1],
        #               [-1,1],
        #               [-1,-1],
        #               [1,-1]]

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
        all_data = None

        # support vectors yi(xi.w+b) = 1


        # more expensive each level
        # step_sizes = [self.max_feature_value * 0.1,
        #               self.max_feature_value * 0.01,
        #               self.max_feature_value * 0.001]
        # the one im making doesn't need tier 3
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01]

        # extremely expensive
        b_range_multiple = 5
        # we dont need to take as small steps with b as we do with w
        b_multiple = 5

        lastest_optimum = self.max_feature_value*10

        firstStep = True

        # for each step sizes
        for step in step_sizes:


            customDimen = []
            for i in self.data[next(iter(self.data))][0]:
                customDimen.append(lastest_optimum)

            # range in which to search. will turn into -80, 80  to -3.2, 3.2 to -0.8, 0.8 etc
            w = np.array(customDimen)
            print('starting w to increment from', w, 'with steps of',step)


            # create a array from max feature * b range of -negative to +positive
            # intervals on step * b range
            # because the previous check's best magnitude is here, we'll constant near it to search
            if firstStep:
                firstStep = False
                checking_list = np.arange(-1 * (lastest_optimum/4 * b_range_multiple),
                                          lastest_optimum/4 * b_range_multiple,
                                          step * b_multiple)
            else:
                checking_list = np.arange(-1 * (lastest_optimum * b_range_multiple),
                                          lastest_optimum * b_range_multiple,
                                          step * b_multiple)


            # this is where I set the stepping intervals (how vector rotates)
            self.stepIntervalRotation = int(len(checking_list)/4)
            self.constantSteps = self.stepIntervalRotation


            # checking_list = np.arange(-1 * (self.max_feature_value * b_range_multiple),
            #                           self.max_feature_value * b_range_multiple,
            #                           step * b_multiple)


            # print('checking list', checking_list)
            print('checking list length', len(checking_list))




            # we can do this because convex
            optimized = False
            while not optimized:


                # moved b check list creation after step iteration ^^^


                # for each interval
                for b in checking_list:


                    # for each direction
                    for transformation in transforms:


                        # get dot product of width range (80,80) to each transform (1,1), (-1,-1) outputs (-80,80)
                        w_t = w*transformation


                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        #
                        # ## add a break here later..


                        # for all class in data
                        for i in self.data:
                            # for all feature sets in each class

                            for xi in self.data[i]:
                                # yi = current class
                                yi = i


                                # if class * dotproduct of magnituded vector with featureset + (check list y-intercept)  not >= 1
                                # why is it 1? I know it has to do with normalizing and wtf but where in code..
                                # >=1 or >=2 has to do with the class name some how...
                                # can change >=1 to anything as long as the data set class is changed also

                                # !!! this only tells me that it works as a support vector by checking all spots
                                # !!! because -1 or 1 will always positive due to yi
                                # !!! therefore if it isn't positive then it was on wrong side and thus doesn't work
                                # !!! old : yi*(np.dot(w_t, xi)+b)) >= 1

                                if not (yi*(np.dot(w_t, xi)+b))-(abs(yi)) >= 0:
                                    # print('origin to (w_t):',w_t)
                                    # print(yi*(np.dot(w_t,xi)+b))
                                    # print('xi', xi)
                                    # print(np.dot(w_t, xi) + b)
                                    # print(yi)
                                    # print("#####")
                                    # then this was not a value SVM

                                    found_option = False

                                    break

                                # elif 1 >= (yi*(np.dot(w_t, xi)+b))-1 >= 0:
                                #     print((yi*(np.dot(w_t, xi)+b))-1)
                                #     print('afasfa',np.linalg.norm(w_t))

                                # elif 2 >= yi*(np.dot(w_t,xi)+b) >= 1:
                                #     print(b)
                                #     print(w_t)
                                #     print(xi)
                                #     print(np.dot(w_t, xi))
                                #     print(np.dot(w_t, xi)+ b)
                                #     print(yi*(np.dot(w_t,xi)+b))
                                #     print('888888888')
                                #print(xi,":" ,yi*(np.dot(w_t,xi)+b))



                            if found_option == False:

                                break


                        # however, if this was a valid SVM(possible) then
                        # store the vector direction magnitude with the y-intercept as the magnitude key(euclid distance)
                        # !!! this is where it will eventually tell me which is the best one (lowest magnitude)
                        # !!! but why lowest?
                        # !!! because something to do with the equation for width = 2/||w|| therefore, we minimized this
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]


                # until we can't optimized anymore (step goes below zero)
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step because w is less than 0 at', w[0])
                elif w[1] < 0:
                    optimized = True
                else:
                    # before code
                    # w = [5,5]
                    # steps = 1
                    # w - step = [4,4]???
                    # #w = w - step


                    #step down the ladder algorithm
                    wOne = w[0]
                    wTwo = w[1]

                    if self.currentIsOne:
                        wOne = wOne - step
                        self.stepIntervalRotation = self.stepIntervalRotation - 1
                        if self.stepIntervalRotation == 0:
                            self.stepIntervalRotation = self.constantSteps
                            self.currentIsOne = False

                            # tax heavy if you lower multiple's value
                            wOne = wTwo - step*int(self.constantSteps/4)

                    else:
                        wTwo = wTwo - step
                        self.stepIntervalRotation = self.stepIntervalRotation - 1
                        if self.stepIntervalRotation == 0:
                            self.stepIntervalRotation = self.constantSteps
                            self.currentIsOne = True
                            wTwo = wOne - step*int(self.constantSteps/4)

                    w = np.array([wOne, wTwo])








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
            lastest_optimum = abs(max(opt_choice[0], key=abs)) + step*2

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
        # psv = 1
        # nsv = -1
        # dec = 0
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



# data_dict = {-2:np.array([[1,7],[2,8],[3,8],[3,4]]),
#              2:np.array([[5,1],[6,-1],[7,3],[4,2]])}

# data_dict = {-666:np.array([[1,7],[2,8],[3,8],[3,4]]),
#              666:np.array([[5,1],[6,-1],[7,3],[4,2]])}

# data_dict = {1:np.array([[1,17],[2,18],[3,18],[3,14],[4,14]]),
#              -1:np.array([[5,7],[6,-11],[7,9],[4,8],[0,10]])}

# data_dict = {1:np.array([[1,7],[2,8],[3,8],[3,4],[6,5],[0,2],[0,1],[0,0],[0,-1],[0,-2],[0,-3],[0,-4]]),
#              -1:np.array([[5,1],[6,-1],[7,3],[4,2],[8,4]])}
# data_dict = {1:np.array([[1,7],[2,8],[3,8],[3,4],[6,5],[0,2],[0,1],[0,0],[0,-1],[0,-2]]),
#              -1:np.array([[5,1],[6,-1],[7,3],[4,2],[8,4]])}


data_dict = {1:np.array([[6,5],[0,-1],[0,-2]]),
             -1:np.array([[7,3],[4,2],[8,4]])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

# predict_us = [[0,10],[1,3],[3,4],[3,5],[5,5],[5,6],[6,-5],[5,8],[4,-2],[2,0],[0,0],[-1,-5],[8,7],[8,8],[3,-6],[3,0]]
# for p in predict_us:
#     svm.predict(p)

svm.visualize()

print(svm.w)
print(svm.b)












