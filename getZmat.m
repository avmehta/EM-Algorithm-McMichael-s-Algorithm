%This function returns the Z matrix. This is a matrix with all observed and
%missing user ratings. We will use this matrix to make predictions based on
%the EM algorithm (and later on McMichael's algorithm). 

function [matrx] = getZmat(dta);
%Return the total number of users without assuming every user in range
%1:max(Y(:,3)) provided a rating (although this is a safe assumption for
%the given data). 
numUsers=size(unique(dta(:,3)),1); 
%Total number of movies rated (This data is k=100)
ttlMovies=max(dta(:,2)); 
matrx=zeros(ttlMovies,numUsers);
LiD=1; %Last user ID
for uIdx = 1:size(dta,1);
    iD=dta(uIdx,3); %Current user ID
    movie=dta(uIdx,2); %Movie rated by user
    rating=dta(uIdx,1); %Rating given by user
    matrx(movie,iD)=rating; %Assign the givn users rating to the tmp vct
end