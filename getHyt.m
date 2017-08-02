%This function returns a subset of the identity matrix after unobserved
%rows have been removed. 
%Input: vect = Z(sub t); ie vector of observed and missing values of the
%t-th user. id: Full identity matrix of Z(sub t) 
%Output: H  = subset of identity matrix with missing value rows removed. 
function [H]=getHyt(vect,id);
idxs = find(vect==0); %Indexes of all the missing values 
H=id;
H(idxs,:)=[]; %Remove all rows that 0s were found. 
end
