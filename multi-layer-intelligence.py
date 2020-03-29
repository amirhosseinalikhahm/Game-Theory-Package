#Authors: Khashayar Shamsolketabi & Amir Hossein Alikhah Mishamandani
#Date & Time: 27/March/2020
#Description: First Library for Game Theory
#-----------------------------------------------------------------------------

#Required Libraries
from prettytable import PrettyTable
import numpy as np
from itertools import chain

#Global Variables
TARGET_MIN = 0;
TARGET_MAX = 1;

#Linear transformation \unit tested
def linearTransformation(sourceMin, sourceMax, targetMin, targetMax, inputNumber):
    if (sourceMin == sourceMax):
        return (targetMin + targetMax)/2;
    else:
        return (targetMin + ((inputNumber - sourceMin)/(sourceMax - sourceMin))*(targetMax - targetMin));

#Linear normalization \unit tested
def linearNormalization(sourceMin, sourceMax, inputNumber):
    return linearTransformation(sourceMin, sourceMax, TARGET_MIN, TARGET_MAX, inputNumber);

#Linear normalization of vectors \unit tested
def linearNormalizationOfVector(sourceMin, sourceMax, inputVector):
    return list(map(lambda x: linearNormalization(sourceMin, sourceMax, x), inputVector));

#Linear normalization of matrices \unit tested
def linearNormalizationOfMatrix(sourceMin, sourceMax, inputMatrix):
    return list(map(lambda X: linearNormalizationOfVector(sourceMin, sourceMax, X), inputMatrix));

#Linear normalization of pay off matrices \unit tested
def linearNormalizationOfPayOff(firstPlayerPayOffs, secondPlayerPayOffs):
    sourceMin = np.min([np.min(firstPlayerPayOffs), np.min(secondPlayerPayOffs)]);
    sourceMax = np.max([np.max(firstPlayerPayOffs), np.max(secondPlayerPayOffs)]);
    return [linearNormalizationOfMatrix(sourceMin, sourceMax, firstPlayerPayOffs), 
            linearNormalizationOfMatrix(sourceMin, sourceMax, secondPlayerPayOffs)]; 

#Weighting average of a vector \unit tested
def vectorWeightException(vector, weights, probabilities):
    return sum(list(map(lambda x: x[1]*weights[x[0]]*probabilities[x[0]], enumerate(vector))));

#Weighting average of a matrix \unit tested
def matrixWeightException(matrix, weights, probabilities):
    nonNormalizedReuslt = list(map(lambda X: vectorWeightException(X,weights, probabilities), matrix));
    totalSum = sum(nonNormalizedReuslt);
    return [x/totalSum for x in nonNormalizedReuslt];

#Calculation of decision distribution of first player \unit tested
def firstPlayerDecisionWithMixStrategyIteration(firstPlayerPayOffs, secondPlayerDecision, secondPlayerMixStrategyDistribution):
    return matrixWeightException(firstPlayerPayOffs, secondPlayerDecision, secondPlayerMixStrategyDistribution);

#Calculation of decision distribution of second player \unit tested
def secondPlayerDecisionWithMixStrategyIteration(secondPlayerPayOffs, firstPlayerDecision, firstPlayerMixStrategyDistribution):
    return matrixWeightException(np.array(secondPlayerPayOffs).T.tolist(), firstPlayerDecision, firstPlayerMixStrategyDistribution);

#Iteration over pay offs with mix strategy \unit tested
def playersDecisionWithMixStrategyIteration(
    firstPlayerPayOffs, 
    secondPlayerPayOffs, 
    firstPlayerInitialDecision, 
    secondPlayerInitialDecision, 
    firstPlayerMixStrategyDistribution, 
    secondPlayerMixStrategyDistribution):
    return [firstPlayerDecisionWithMixStrategyIteration(firstPlayerPayOffs, secondPlayerInitialDecision, secondPlayerMixStrategyDistribution),
            secondPlayerDecisionWithMixStrategyIteration(secondPlayerPayOffs, firstPlayerInitialDecision, firstPlayerMixStrategyDistribution)];

#Iteration over pay offs without mix strategy \unit tested
def playersDecisionWithoutMixStrategyIteration(
    firstPlayerPayOffs, 
    secondPlayerPayOffs, 
    firstPlayerInitialDecision, 
    secondPlayerInitialDecision):
    m = len(firstPlayerPayOffs);
    n = len(secondPlayerPayOffs[0]);
    return [firstPlayerDecisionWithMixStrategyIteration(firstPlayerPayOffs, secondPlayerInitialDecision, np.ones(n)/n),
            secondPlayerDecisionWithMixStrategyIteration(secondPlayerPayOffs, firstPlayerInitialDecision, np.ones(m)/m)];

#Multiple iteration over pay offs with multiple times using mix strategy
def playersDecisionWithMultiTimesMixStrategyIterations(
    firstPlayerPayOffs, 
    secondPlayerPayOffs, 
    firstPlayerMixStrategyDistribution, 
    secondPlayerMixStrategyDistribution,
    numberOfIterations):
    m = len(firstPlayerPayOffs);
    n = len(secondPlayerPayOffs[0]);
    firstPlayerInitialDecision = np.ones(m)/m;
    secondPlayerInitialDecision = np.ones(n)/n;
    printPlayersIterationDecision(0, firstPlayerInitialDecision, secondPlayerInitialDecision);
    if (numberOfIterations > 0):
        for i in range(0, numberOfIterations):
            [firstPlayerInitialDecision, secondPlayerInitialDecision] = playersDecisionWithMixStrategyIteration(
                firstPlayerPayOffs, 
                secondPlayerPayOffs, 
                firstPlayerInitialDecision, 
                secondPlayerInitialDecision, 
                firstPlayerMixStrategyDistribution, 
                secondPlayerMixStrategyDistribution);
            printPlayersIterationDecision(i + 1, firstPlayerInitialDecision, secondPlayerInitialDecision);
    return [firstPlayerInitialDecision, secondPlayerInitialDecision];
        
#Multiple iteration over pay offs with one time using mix strategy
def playersDecisionWithOneTimeMixStrategyIterations(
    firstPlayerPayOffs, 
    secondPlayerPayOffs, 
    firstPlayerMixStrategyDistribution, 
    secondPlayerMixStrategyDistribution,
    numberOfIterations):
    m = len(firstPlayerPayOffs);
    n = len(secondPlayerPayOffs[0]);
    firstPlayerInitialDecision = np.ones(m)/m;
    secondPlayerInitialDecision = np.ones(n)/n;
    printPlayersIterationDecision(0, firstPlayerInitialDecision, secondPlayerInitialDecision);
    if (numberOfIterations > 0):
        for i in range(0, numberOfIterations):
            if (i == 0):
                [firstPlayerInitialDecision, secondPlayerInitialDecision] = playersDecisionWithMixStrategyIteration(
                    firstPlayerPayOffs, 
                    secondPlayerPayOffs, 
                    firstPlayerInitialDecision, 
                    secondPlayerInitialDecision, 
                    firstPlayerMixStrategyDistribution, 
                    secondPlayerMixStrategyDistribution);
                printPlayersIterationDecision(1, firstPlayerInitialDecision, secondPlayerInitialDecision);
            else:
                [firstPlayerInitialDecision, secondPlayerInitialDecision] = playersDecisionWithoutMixStrategyIteration(
                    firstPlayerPayOffs, 
                    secondPlayerPayOffs, 
                    firstPlayerInitialDecision, 
                    secondPlayerInitialDecision);
                printPlayersIterationDecision(i + 1, firstPlayerInitialDecision, secondPlayerInitialDecision);
    return [firstPlayerInitialDecision, secondPlayerInitialDecision]; 

#Multiple iteration over pay offs without mix strategy
def playersDecisionWithoutMixStrategyIterations(
    firstPlayerPayOffs, 
    secondPlayerPayOffs,
    numberOfIterations):
    m = len(firstPlayerPayOffs);
    n = len(secondPlayerPayOffs[0]);
    firstPlayerInitialDecision = np.ones(m)/m;
    secondPlayerInitialDecision = np.ones(n)/n;
    printPlayersIterationDecision(0, firstPlayerInitialDecision, secondPlayerInitialDecision);
    if (numberOfIterations > 0):
        for i in range(0, numberOfIterations):
            [firstPlayerInitialDecision, secondPlayerInitialDecision] = playersDecisionWithoutMixStrategyIteration(
                firstPlayerPayOffs, 
                secondPlayerPayOffs, 
                firstPlayerInitialDecision, 
                secondPlayerInitialDecision);
            printPlayersIterationDecision(i + 1, firstPlayerInitialDecision, secondPlayerInitialDecision);
    return [firstPlayerInitialDecision, secondPlayerInitialDecision];

#Print players decision for iteration
def printPlayersIterationDecision(iteration, firstPlayerInitialDecision, secondPlayerInitialDecision):
    print('Iteration', iteration, ': P1 =', firstPlayerInitialDecision, '& P2 =', secondPlayerInitialDecision);


print('transformation unit test: ', 
      linearTransformation(-4,-1, 1, 4, -2) );
print('normalization unit test: ', 
      linearNormalization(-4,-1, -2) );
print('normalization of vector unit test: ', 
      linearNormalizationOfVector(-4,-1, [-2, -1]));
print('normalization of matrix unit test: ', 
      linearNormalizationOfMatrix(-4,-1, [[-2, -3], [-1, 0]]));
print('normalization of pay offs unit test: ', 
      linearNormalizationOfPayOff([[-2, -3], [-1, 0]], 
                                  [[-2, -4], [-1, 0]]));
print('weigting exception of vectors unit test: ', 
      vectorWeightException([2, 3, 1, 0],
                            [.2, .3, .2, .3], 
                            [.2, .4, .1, .3]));
print('weigting exception of matrix unit test: ', 
      matrixWeightException([[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
                            [.2, .3, .2, .3], 
                            [.2, .4, .1, .3]));
print('calculate decision for first player with mix strategy unit test: ', 
      firstPlayerDecisionWithMixStrategyIteration([[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
                            [.2, .3, .2, .3], 
                            [.2, .4, .1, .3])); 
print('calculate decision for second player with mix strategy unit test: ', 
      secondPlayerDecisionWithMixStrategyIteration(
          [[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
          [.3, .4, .3], 
          [.3, .5, .2])); 
print('update decision for players with mix strategy unit test: ', 
      playersDecisionWithMixStrategyIteration(
          [[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
          [[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
          [.3, .4, .3],
          [.2, .3, .2, .3],
          [.3, .5, .2],
          [.2, .4, .1, .3]));
print('update decision for players without mix strategy unit test: ', 
      playersDecisionWithoutMixStrategyIteration(
          [[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
          [[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
          [.3, .4, .3],
          [.2, .3, .2, .3]));
print('Iterate decision for players with multipe mix strategy unit test: ', 
      playersDecisionWithMultiTimesMixStrategyIterations(
          [[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
          [[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
          [.3, .5, .2],
          [.2, .4, .1, .3],
          3));
print('Iterate decision for players with one time mix strategy unit test: ', 
      playersDecisionWithOneTimeMixStrategyIterations(
          [[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
          [[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
          [.3, .5, .2],
          [.2, .4, .1, .3],
          3));
print('Iterate decision for players without mix strategy unit test: ', 
      playersDecisionWithoutMixStrategyIterations(
          [[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
          [[2, 3, 1, 0], [2, 3, 1, 0], [2, 3, 1, 0]],
          3));
