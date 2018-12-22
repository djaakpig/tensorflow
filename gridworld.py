import numpy as np
import random
import itertools

from skimage.transform import resize

class GameOb():
    def __init__(self, coords, size, intensity, channel, reward, name):
        self.x = coords[0]
        self.y = coords[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

class GameEnv():
    def __init__(self, partial, size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        a = self.reset()

    def reset(self):
        self.objects = []

        hero = GameOb(self.newPos(), 1, 1, 2, None, 'hero')
        self.objects.append(hero)
        bug = GameOb(self.newPos(), 1, 1, 1, 1, 'goal')
        self.objects.append(bug)
        hole = GameOb(self.newPos(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole)
        bug2 = GameOb(self.newPos(), 1, 1, 1, 1, 'goal')
        self.objects.append(bug2)
        hole2 = GameOb(self.newPos(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole2)
        bug3 = GameOb(self.newPos(), 1, 1, 1, 1, 'goal')
        self.objects.append(bug3)
        bug4 = GameOb(self.newPos(), 1, 1, 1, 1, 'goal')
        self.objects.append(bug4)

        state = self.renderEnv()
        self.state = state
        return state

    def moveChar(self, direction):
        # 0 - up, 1 = down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY - 2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX - 2:
            hero.x += 1
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize

    def newPos(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for o in self.objects:
            if (o.x, o.y) not in currentPositions:
                currentPositions.append((o.x, o.y))
        for p in currentPositions:
            points.remove(p)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def checkGoal(self):
        others = []
        for o in self.objects:
            if o.name == 'hero':
                hero = o
            else:
                others.append(o)
        ended = False
        for o in others:
            if hero.x == o.x and hero.y == o.y:
                self.objects.remove(o)
                if o.reward == 1:
                    self.objects.append(GameOb(self.newPos(), 1, 1, 1, 1, 'goal'))
                else:
                    self.objects.append(GameOb(self.newPos(), 1, 1, 0, -1, 'fire'))
                return o.reward, False
        if ended == False:
            return 0.0, False

    def renderEnv(self):
        a = np.zeros([self.sizeY+2,self.sizeX+2,3])
        a[1:-1,1:-1,:] = 0
        hero = None
        for o in self.objects:
            a[o.y+1:o.y+o.size+1,
              o.x+1:o.x+o.size+1,
              o.channel] = o.intensity
            if o.name == 'hero':
                hero = o
        if self.partial == True:
            a = a[hero.y:hero.y+3,hero.x:hero.x+3,:]
        b = resize(a[:,:,0], [84,84,1], mode='reflect', anti_aliasing=True)
        c = resize(a[:,:,1], [84,84,1], mode='reflect', anti_aliasing=True)
        d = resize(a[:,:,2], [84,84,1], mode='reflect', anti_aliasing=True)
        a = np.stack([b,c,d], axis=2)
        return a

    def step(self, action):
        penalty = self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
        return state, (reward + penalty), done
