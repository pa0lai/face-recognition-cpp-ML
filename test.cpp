// test.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//
//
#define NOMINMAX//add

#include <opencv2/core/core.hpp>//opencv
#include <opencv2/highgui/highgui.hpp>//opencv 
#include <opencv2/imgproc/imgproc.hpp>//opencv 
#include <opencv2/opencv.hpp>//opencv 
#include <stdio.h>
#include <iostream>
#include <Windows.h>
#include <ctype.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <fstream>  
#include <time.h>

using namespace std; 
using namespace cv;//opencv

vector<Mat> images;
char filename[100];
int image0 = 20;
int image1 = 20;
int image2 = 20;
int image3 = 20;
int image4 = 20;
int num = 0;
Mat src;

int main()
{
 int choice;
 cout << "Enter 1 for Function 1, 2 for Function 2: ";
 cin >> choice;

 if (choice == 1) {
  Mat values(image0 + image1 + image2 + image3 + image4, 1, CV_32SC1);
  for (int i = 1; i <= image0; i++)
  {
   sprintf(filename, "image0//%d.jpg", i);
   values.at<int>(i - 1, 0) = 0;//face:1
   src = cvLoadImage(filename, 0);
   images.push_back(src);
   num++;
  }

  for (int i = 1; i <= image1; i++)
  {
   sprintf(filename, "image1//%d.jpg", i);
   values.at<int>(num, 0) = 1;//not face:0
   src = cvLoadImage(filename, 0);
   images.push_back(src);
   num++;
  }

  for (int i = 1; i <= image2; i++)
  {
   sprintf(filename, "image2//%d.jpg", i);
   values.at<int>(num, 0) = 2;//not face:0
   src = cvLoadImage(filename, 0);
   images.push_back(src);
   num++;
  }

  for (int i = 1; i <= image3; i++)
  {
   sprintf(filename, "image3//%d.jpg", i);
   values.at<int>(num, 0) = 3;//not face:0
   src = cvLoadImage(filename, 0);
   images.push_back(src);
   num++;
  }

  for (int i = 1; i <= image4; i++)
  {
   sprintf(filename, "image4//%d.jpg", i);
   values.at<int>(num, 0) = 4;//not face:0
   src = cvLoadImage(filename, 0);
   images.push_back(src);
   num++;
  }

  int nEigens = images.size() - 1;//number of Eigen vectors, max components to preserve

  //Load the images into a Matrix 
  Mat desc_mat(images.size(), images[0].rows * images[0].cols, CV_8UC1);

  for (int i = 0; i < images.size(); i++)
  {
   desc_mat.row(i) = images[i].reshape(1, 1) + 0;

  }

  Mat average;
  PCA pca(desc_mat, average, CV_PCA_DATA_AS_ROW, nEigens);//PCA structure
  Mat data(desc_mat.rows, nEigens, CV_32FC1);//This Mat will contain all the Eigenfaces that will be used later with SVM for detection

  //Project the images onto the PCA subspace, desc_mat become smaller
  for (int i = 0; i < images.size(); i++)
  {
   Mat projectedMat(1, nEigens, CV_32FC1);
   pca.project(desc_mat.row(i), projectedMat);
   data.row(i) = projectedMat.row(0) + 0;
  }

  printf("after input\n");

  CvSVMParams param;//SVM machine learning

  CvTermCriteria criteria;
  criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
  param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);

  CvMat d1 = (CvMat)data;//traning set
  CvMat d2 = (CvMat)values;//mark set

  //SVM traning         
  CvSVM svm;//auto train svm
  svm.train(&d1, &d2, NULL, NULL, param);
  svm.save("svmdata.xml");

  //test data
  Mat cs;

  int counts[5] = { 0 };

  for (int i = 1; i <= image0; i++)
  {
   sprintf(filename, "image0//%d.jpg", i);

   src = cvLoadImage(filename, 0);
   if (!src.data)
   {
    printf("bad frame\n");
   }

   cs = src.reshape(1, 1) + 0;
   Mat projectedMat(1, nEigens, CV_32FC1);
   pca.project(cs, projectedMat);
   CvMat d3 = (CvMat)projectedMat;


   int ret = svm.predict(&d3);//SVM predict
   imshow("userFace", src);
   printf("%d\n", ret);

   int prediction = svm.predict(&d3); // SVM 預測
   counts[prediction]++; // 更新計數器
  }

  for (int i = 1; i <= 4; i++) // 從1開始，因為0代表您自己
  {
   float similarity = (float)counts[i] / image0 * 100.0;
   printf("classmate %d: similar rate %.2f%%\n", i, similarity);
  }
  waitKey(100);
  system("pause");
  return 0;
 }
 else if (choice == 2) {
 Mat values(image1 + image2 + image3 + image4, 1, CV_32SC1);

 for (int i = 1; i <= image1; i++)
 {
  sprintf(filename, "image1//%d.jpg", i);
  values.at<int>(num, 0) = 1;//not face:0
  src = cvLoadImage(filename, 0);
  images.push_back(src);
  num++;
 }

 for (int i = 1; i <= image2; i++)
 {
  sprintf(filename, "image2//%d.jpg", i);
  values.at<int>(num, 0) = 2;//not face:0
  src = cvLoadImage(filename, 0);
  images.push_back(src);
  num++;
 }

 for (int i = 1; i <= image3; i++)
 {
  sprintf(filename, "image3//%d.jpg", i);
  values.at<int>(num, 0) = 3;//not face:0
  src = cvLoadImage(filename, 0);
  images.push_back(src);
  num++;
 }

 for (int i = 1; i <= image4; i++)
 {
  sprintf(filename, "image4//%d.jpg", i);
  values.at<int>(num, 0) = 4;//not face:0
  src = cvLoadImage(filename, 0);
  images.push_back(src);
  num++;
 }

 int nEigens = images.size() - 1;//number of Eigen vectors, max components to preserve

 //Load the images into a Matrix 
 Mat desc_mat(images.size(), images[0].rows * images[0].cols, CV_8UC1);

 for (int i = 0; i < images.size(); i++)
 {
  desc_mat.row(i) = images[i].reshape(1, 1) + 0;

 }

 Mat average;
 PCA pca(desc_mat, average, CV_PCA_DATA_AS_ROW, nEigens);//PCA structure
 Mat data(desc_mat.rows, nEigens, CV_32FC1);//This Mat will contain all the Eigenfaces that will be used later with SVM for detection

 //Project the images onto the PCA subspace, desc_mat become smaller
 for (int i = 0; i < images.size(); i++)
 {
  Mat projectedMat(1, nEigens, CV_32FC1);
  pca.project(desc_mat.row(i), projectedMat);
  data.row(i) = projectedMat.row(0) + 0;
 }

 printf("after input\n");

 CvSVMParams param;//SVM machine learning

 CvTermCriteria criteria;
 criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
 param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);

 CvMat d1 = (CvMat)data;//traning set
 CvMat d2 = (CvMat)values;//mark set

 //SVM traning         
 CvSVM svm;//auto train svm
 svm.train(&d1, &d2, NULL, NULL, param);
 svm.save("svmdata.xml");

 //test data
 Mat cs;

 int counts[5] = { 0 };

 for (int i = 1; i <= image0; i++)
 {
  sprintf(filename, "image0/%d.jpg", i);

  src = cvLoadImage(filename, 0);
  if (!src.data)
  {
   printf("bad frame\n");
  }

  cs = src.reshape(1, 1) + 0;
  Mat projectedMat(1, nEigens, CV_32FC1);
  pca.project(cs, projectedMat);
  CvMat d3 = (CvMat)projectedMat;


  int ret = svm.predict(&d3);//SVM predict
  imshow("userFace", src);
  printf("%d\n", ret);

  int prediction = svm.predict(&d3); // SVM 預測
  counts[prediction]++; // 更新計數器
 }

 for (int i = 1; i <= 4; i++) // 從1開始，因為0代表您自己
 {
  float similarity = (float)counts[i] / image0 * 100.0;
  printf("classmate %d: similar rate %.2f%%\n", i, similarity);
 }

 }
 else {
  cout << "Invalid choice" << endl;
 }
 waitKey(100);
 system("pause");
 return 0;
 
}