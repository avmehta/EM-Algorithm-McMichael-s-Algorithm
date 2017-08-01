% The purpose of this script is to replicate the EM algorithm described in
% Little and Rubin and tested in Application of a Gaussian, Missing-Data
% Model to Product Recommendation (Roberts)

close all
clear all

%%Importing data
%Y is a data set inclusive of all observed values of n users. X is a data
%set inclusive of all missing values of n users. In this form, the data is
%seperated with column 1 being different ratings, column 2 being different
%movies, and column 3 being different users. The data here is sorted in
%order of user. 
Y=csvread('Y.csv');  
X=csvread('P.csv');

%%Creating the appropriate matrices
%First we'll get our Z matrix (from the data in Y). This puts what Y was
%previously represented as in the form of n users (n columns) and k ratings
%(k rows) 
Z=getZmat(Y);
        
    
    
    


