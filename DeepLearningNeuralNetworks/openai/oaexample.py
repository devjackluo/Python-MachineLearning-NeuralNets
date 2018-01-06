import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 0.001  # 1e-3
# get the game i want to train
env = gym.make('CartPole-v0')
env.reset()
# max number of steps i want to take
goal_steps = 500
# initial data score requirement
score_requirement = 50
# how many random games to create init data
initial_games = 10000


# function to show how games are played
def some_random_games_first():
    # five games
    for episode in range(5):
        # reset game
        env.reset()
        # how max number of steps
        for t in range(goal_steps):
            # render out gui
            env.render()
            # get a random action (1 or 0)
            action = env.action_space.sample()
            # perform action, returns tuple of
            # observation is an evaluation of game state
            # reward for cartpole is +1 each frame
            # done is game has ended
            # info is something... just placeholder for this one.. info for other games (previous board etc)
            observation, reward, done, info = env.step(action)
            # if game ended, then stop looping steps
            if done:
                break

# some_random_games_first()



def initial_population():

    # training data to be populated
    training_data = []
    # all scores of games?
    scores = []
    # all scores above accepted score requirement?
    accepted_scores = []

    # for number of games to get data from
    for _ in range(initial_games):
        # init score
        score = 0
        # how the game turned out
        game_memory = []
        # ...
        prev_observation = []
        # for max num steps
        for _ in range(goal_steps):
            # random action of (0 or 1)
            action = random.randrange(0, 2)
            # get game data from step
            observation, reward, done, info = env.step(action)

            # if we have any previous observation
            if len(prev_observation) > 0:
                # then we append the game memory with previous observation with the action we took
                game_memory.append([prev_observation, action])

            # set previous observation as current observation
            prev_observation = observation

            # score = how many frames we survived
            score += reward
            # if game ended, end loop
            if done:
                break

        # if this game scored higher than requirement
        if score >= score_requirement:
            # we append the score to accepted scores
            accepted_scores.append(score)
            # for all data in the game that just ended
            for data in game_memory:
                # if the action was 1(right?) or 0(left?)
                # then set output to [0,1] or [1,0] respectively
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                # then link the previous observation to respected output
                # (this is because since it is better than required scored, must be good moves)
                training_data.append([data[0], output])

        # reset game
        env.reset()
        # append all scores (not needed?)
        scores.append(scores)

    # convert to numpy array
    # training data is a set of all previous observations with (predicted/expected move(1 or 0))
    training_data_save = np.array(training_data)
    # save training data as file
    np.save('saved.np', training_data_save)

    # stuff
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data_save





def neural_network_model(input_size):

    # pass in input data size
    # input data size is how many beginning nodes it has
    network = input_data(shape=[None, input_size, 1], name='input')

    # create hidden layers of varies lengths and dropping 20% of each.
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    # output layer returns only two options (1 or 0)
    network = fully_connected(network, 2, activation='softmax')

    # this is the learning method
    # also the loss adjustment method of cate_cross
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    # then officially create model with tflearn.DNN
    model = tflearn.DNN(network, tensorboard_dir='log')

    #return the model
    return model


def train_model(training_data, model=False):

    # for all training data's previous observation
    # previous observations contains 4 value array [1,1,1,1]
    # reshape previous observation in array containing 4 arrays [[1],[1],[1],[1]]
    X = np.array([i[0] for i in training_data])
    X = X.reshape(-1, len(training_data[0][0]), 1)

    # y is all the actions [0,1] or [1,0]
    y = [i[1] for i in training_data]

    # if we didn't pass in model
    if not model:
        # create new NN model
        model = neural_network_model(input_size=len(X[0]))

    # model uses data of observations to match its expected action [0,1] or [1,0]
    model.fit({'input': X}, {'targets':y}, n_epoch=1, snapshot_step=500, show_metric=True, run_id='openaistuff')

    # return trained model
    return model



# get random training data
# not very useful for long term goal
training_data = initial_population()

print(len(training_data))
# train model based of score requirement obtained data
model = train_model(training_data)


def scoreModel():

    scores = []
    choices = []

    # for 100 games
    for each_game in range(100):

        score = 0
        game_memory = []
        prev_obs = []
        env.reset()

        # for max number of steps
        for _ in range(goal_steps):
            #env.render()

            # for first step, we make random move
            if len(prev_obs) == 0:
                action = random.randrange(0,2)
            else:
                # if not first step
                # use model to predict what action from previous observation
                # below returns a multiple array so we take only one the inside one
                # print(model.predict(prev_obs.reshape(-1, len(prev_obs), 1)))
                # argmax returns which index of [0.2412412, 0.53535] is the max (this case 1)
                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

            # append choice the action
            choices.append(action)

            # get new data from performing action
            new_observation, reward, done, info = env.step(action)
            # set previous observation as new observation
            prev_obs = new_observation

            # append the new observation and action to game memory
            game_memory.append([new_observation, action])

            # score increase each step you can go
            score += reward
            # end game if done (done is if pool drops 15%?)
            if done:
                break

        # append the score to scores
        scores.append(score)

    # self explain
    averageScore = sum(scores)/len(scores)
    print('Average Score', averageScore)

    # # how many of each choice did we do (right and lefts?)
    # print('Choice 1: {}, Choice 2: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))
    #
    #
    # userinput = input('To Save Model? (Y/N) : ').lower()
    # if userinput == 'y':
    #     nameInput = input('Model Name: ')
    #     model.save(nameInput + ".model")


    # model.load('first.model')

    return averageScore





def createNextGenerationData(requirement, manyGames):

    new_data = []

    for each_game in range(manyGames):

        current_score = 0
        game_data = []
        prev_obser = []
        env.reset()

        for _ in range(goal_steps):

            if len(prev_obser) == 0:
                action = random.randrange(0, 2)
            else:
                action = np.argmax(model.predict(prev_obser.reshape(-1, len(prev_obser), 1))[0])

            #choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obser = new_observation

            game_data.append([new_observation, action])

            current_score += reward
            if done:
                break

        if current_score >= requirement:
            #accepted_scores.append(score)
            for data in game_data:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                new_data.append([data[0], output])

        #scores.append(score)

    new_data_save = np.array(new_data)
    np.save('saved.np', new_data_save)


    return new_data_save














