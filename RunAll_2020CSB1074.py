# -*- coding: utf-8 -*-
"""
Arshdeep Singh
2020CSB1074
CS518 Computer Vision
TASK : Create a Bag of words-based matching/categorization solutions on the MNIST-fashion database.
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray
# from skimage.feature import match_descriptors, plot_matches, SIFT
import cv2 as cv #only used for SIFT feature extraction

#other imports
from statistics import mode


# labeling corresponding names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# feature extraction using SIFT
def feature_using_SIFT(data_img,labels):
  descriptor_extractor = cv.SIFT_create()
  descriptors = []
  keypoints = []
  new_labels = []
  j = 0
  for i, img in enumerate(data_img):
    # using sift of ski-image instead of cv2
    keypoint,descriptor = descriptor_extractor.detectAndCompute(img,None)
    if descriptor is not None:
      new_labels.append(labels[j])
      keypoints.append(keypoint)
      descriptors.append(descriptor)
    j = j + 1
  return descriptors,keypoints,new_labels

# for finding most matching cluster in BOW
def most_matching_cluster(descriptor,BOW):
  return np.argmin(np.sqrt(np.sum((descriptor-BOW)**2,axis = 1)))

#for computing the histogram for given image
def computeOneHistogram(descriptor,BOW):
  histogram = [0 for i in range(len(BOW))]
  for i, desc in enumerate(descriptor):
    matching_index = most_matching_cluster(desc,BOW)
    histogram[matching_index] +=1
  return histogram

# just returs the most matching image as 
def most_matching_image(matching_images):
  return mode(matching_images)

# for matching a given histogram to all the available train histograms
def matchOneHistogram(img_histogram,ALLhistograms,labels):
  # match using cosine similarity
  similarity = []
  for i,hist in enumerate(ALLhistograms):
    cosine = np.abs(np.dot(img_histogram,hist)/(np.linalg.norm(img_histogram)*np.linalg.norm(hist)))
    similarity.append(cosine)

  data = list(zip(similarity, labels))
  data.sort(reverse = True)

  data_points = 50
  data = data[:data_points]
  label_match = [0 for _ in range(10)]
  for i in range(data_points):
    label_match[data[i][1]]+=data[i][0]
  match = np.argmax(label_match)
  return match

# for matching all test histograms to all the test histograms
# class matchoneHistogram function
def matchHistograms(test_histograms,train_histograms,train_labels,test_labels):
  matchings = []
  for i,img_histogram in enumerate(test_histograms):
    match = matchOneHistogram(img_histogram,train_histograms,train_labels)
    matchings.append(match)
  return matchings

 
#--------------------------------------------------------------------------------------------------------------
# K means Clustering implementation in python (scratch)
class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters # cluster number
        self.max_iterations = 300 # max iteration. don't want to run inf time
        self.num_examples = len(X)
        self.num_features = 128 # num of examples, num of features
        self.plot_figure = True # plot figure
        
    # randomly initialize centroids
    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features)) # row , column full with zero 
        for k in range(self.K): # iterations of 
            centroid = X[np.random.choice(range(self.num_examples))] # random centroids
            centroids[k] = centroid
        #print(centroids)
        return centroids # return random centroids
    
    # create cluster Function
    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.K)]
        for point_idx, point in enumerate(X):
            closest_centroid = most_matching_cluster(point,centroids)
            # closest centroid using euclidian distance
            clusters[closest_centroid].append(point_idx)
        return clusters 
        
    # calculation of new centroids from the given cluster
    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.num_features)) # row , column full with zero
        for idx,clt in enumerate(clusters):
          clt_points = np.array(clt)
          if len(clt_points)==0:
            continue
          elements = np.array(X[clt_points])
          centroids[idx] = np.mean(elements,axis = 0)
        return centroids
    
    # prediction
    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples) # row1 fillup with zero
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred
        
    # fit data
    def fit(self, X):
        centroids = self.initialize_random_centroids(X) # initialize random centroids
        for _ in range(self.max_iterations):
            clusters = self.create_cluster(X, centroids) # create cluster
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X) # calculate new centroids
            diff = centroids - previous_centroids # calculate difference
            if not diff.any():
                break
        return centroids

#------------------------------------------------------------------------------------------------------------


def createVisualDictionary(img_descriptors,img_keypoints):
  desc = []
  for i,d in enumerate(img_descriptors):
    for j in range(len(d)):
      desc.append(d[j])

  ALLdesc = np.array(desc)
  # return k clusters representing BOW (bag of visual words)
  numClusters = 50
  km = KMeansClustering(ALLdesc,numClusters)
  BOF = km.fit(ALLdesc)
  return BOF

# for computing histogram
# calls computeOneHistogram - which returns histogram for a single image
def computeHistograms(ALLdescriptors,BOW):
  ALLhistograms = []
  for i,img_descriptor in enumerate(ALLdescriptors):
    histogram = computeOneHistogram(img_descriptor,BOW)
    ALLhistograms.append(histogram)
  return ALLhistograms





from sklearn.metrics import classification_report, accuracy_score, recall_score

def classAccuracy(predicted,actual):
  classBin = [0 for _ in range(10)]
  classCount = [0 for _ in range(10)]

  for i in range(len(predicted)):
    if predicted[i] == actual[i]:
      classBin[predicted[i]] +=1
    classCount[actual[i]] +=1
  
  for i in range(10):
    classAccuracy = classBin[i]
    test_cases = classCount[i]
    if test_cases>0:
      classBin[i] = classAccuracy/test_cases
  return classBin


def statisticalAnalysis(pred_labels,test_labels):
  recall = recall_score(test_labels,pred_labels,average = None,zero_division=0)
  accuracy = accuracy_score(test_labels,pred_labels)
  print("Recall: ",recall)
  print("Overall Accuracy: ",accuracy*100)
  class_wise_accuracy = classAccuracy(pred_labels,test_labels)
  print("Class Wise Accuracy: ")
  for i in range(10):
    print(f"{i:4d} {class_names[i]:30s} {class_wise_accuracy[i]:5.5f}")
  # Writing to the report file
  file = open('result.txt','w')
  for i,predict in enumerate(pred_labels):
    file.write(f"{1+i:4d} {'Predicted: ':15s} {class_names[pred_labels[i]]:15s} {'Actual: ':10s} {class_names[test_labels[i]]:15s}")



#---------------------------------------------------Main--------------------------------------------------------
def main():
  num_train = int(input("Enter number of Train images (min 100 max 60000): "))
  num_test = int(input("Enter number of Test images (max 10000): "))
  # Step 1:
  #Collecting data from dataset
  fashion_mnist = tf.keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
  train_images = train_images[:num_train]
  train_labels = train_labels[:num_train]
  test_images = test_images[:num_test]
  test_labels = test_labels[:num_test]
  print("Step1: Data collection")
  print('Train: X=%s, y=%s' % (train_images.shape, train_labels.shape))
  print('Test: X=%s, y=%s' % (test_images.shape, test_labels.shape))
  #Step 2:
  #feature extraction of all images
  print("Step2: Feature Extraction Using SIFT")
  train_desc, train_key, train_labels = feature_using_SIFT(train_images,train_labels)
  test_desc, test_key, test_labels = feature_using_SIFT(test_images,test_labels)

  # Step 3:
  # creating a Bag of Visual Words (features)
  print("Step3: Creating Bag of Fearures")
  BOW = createVisualDictionary(train_desc,train_key)
  print("BOW shape: ",BOW.shape)

  #Step 4:
  #computing histograms of all the images
  print("Step4: Computing all Histograms")
  ALLhistograms_train = computeHistograms(train_desc,BOW)
  ALLhistograms_test = computeHistograms(test_desc,BOW)

  #Step 5:
  # matching the histograms
  print("Step5: Matching Histograms")
  results = matchHistograms(ALLhistograms_test,ALLhistograms_train,train_labels,test_labels)

  #Step 6:
  #Stats
  print("Step6: Analysis")
  statisticalAnalysis(results,test_labels)


if __name__ == '__main__':
  main()