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
P=csvread('P.csv');

%%Creating the appropriate matrices
%First we'll get our Z matrix (from the data in Y). This puts what Y was
%previously represented as in the form of n users (n columns) and k ratings
%(k rows). Missing data represented as 0 to preserve matrix shape.
%A matrix where missing data is represented as NaN is also created to allow
%for functions such as nancov and nanmean to be used. 
Z=getZmat(Y);
ZN=getZmatNaN(Y);
rtings=size(Z,1);
users=size(Z,2);

%Creating matrix to check RMSE/MMSE (These are essentially target values)
P=getZmat(P);


%mu is the mean vector corresponding to all user ratings in Z(sub t) It is
%transposed to keep the dimensions consistent (after transposing Z2). 
%As is defined in the paper, mu_yt are simply subsets of this mean vector
%as defined by the H_yt subset of the identity k x k identity matrix (H_yt
%being the subset of the identity matrix removed of unobserved rows). This
%arithmetic mean of the observed values is noted to be a good initial
%estimate for mu (beginning of section C initialization).
mu=(sum(Z')./sum(Z'~=0))'; 
%R=nancov(ZN','pairwise'); %Covariance excluding unknown values.
id_mat=eye(rtings); %Full identity matrix


%%Here we can start the EM algorithm. 

%Initializing R for the first user. FOUR ways to initialize R are described
%in Robert's paper. The simplest way is to select R to be the identity
%matrix. 
%R=id_mat;  

%Initialization of R using method j-4
%Equation 11:
%This is the value described in equation 11 that relates to initialization
%of R for j=2,3,and 4
S=0;  
N=zeros(rtings,rtings);
for usrIdx = 1:size(Z,2)
    usr=Z(:,usrIdx);
    %Subset of identity matrix corresponding to Y(sub t)
    H_yt=getHyt(usr,id_mat);
    %Observed ratings of user
    yt=H_yt*(usr);
    lN=H_yt'*H_yt;
    s=H_yt'*(yt-H_yt*mu)*(yt-H_yt*mu)'*H_yt;
    S=S+s;
    N=lN+N;
end
R=(N^-.5)*S*(N^-.5);
    




%Number of iterations to perform EM algorithm. Note that in practice, the
%algorithm will continue iterations until some convergence condition is
%met. 
iter=100; %27 is the number of iters EM took to converge in Robert's paper
lastIter=0; 
%A vector of RMSEs from each iteration
RMSEs=zeros(1,1);
%Initialize RMSE value (for convergence criterion).
Last_RMSE=5;
%We will loop every iteration, updating mu, then subsequently updating R.
for ii = 1:iter
    %Updating mu 
    sumTerm1=zeros(rtings,rtings); %Initializing sumTerm1 and sumTerm2
    sumTerm2=zeros(rtings,1);
    for usrIdx = 1:size(Z,2)
        usr=Z(:,usrIdx);
        %Subset of identity matrix corresponding to Y(sub t)
        H_yt=getHyt(usr,id_mat);
        %Observed ratings of user
        yt=H_yt*(usr);
        %Observed values covariance matrix 
        R_yt=H_yt*R*H_yt';
        %Algorithm described in section B equation 2
        %Note that terms will be combined outside of this loop
        term1=H_yt'*(inv(R_yt))*H_yt; 
        term2=H_yt'*(inv(R_yt))*yt;
        sumTerm1=term1+sumTerm1;
        sumTerm2=term2+sumTerm2;
    end
    mu_hat=inv(sumTerm1)*sumTerm2;

    
    %Updating R and calculating error
    sumTerm3=zeros(rtings,rtings); %0 the summation terms. 
    sumTerm4=zeros(rtings,rtings);
    totalLen=0;
    sum_t_terms=0;
    for usrIdx=1:size(Z,2)
        usr=Z(:,usrIdx);
        usrT=P(:,usrIdx); %Targets of user
        H_yt=getHyt(usr,id_mat);
        %H_xt is a subset of the identity matrix corresponding to only
        %unobserved values. 
        H_xt=getHxt(usr,id_mat);
        yt=H_yt*usr;
        %First we need to compute equation 6 (as stated in section B) to
        %get X_hat(sub t) (ie predicted values for unobserved movies).
        %Getting required values
        R_yt=H_yt*R*H_yt';
        R_xt=H_xt*R*H_xt';
        R_xtyt=H_xt*R*H_yt';
        mu_xt=H_xt*mu_hat; %Note that we're using the updated mu term 
        mu_yt=H_yt*mu_hat;
        %Equation 6:
        X_hat_t=(R_xtyt*(inv(R_yt))*(yt-mu_yt))+(mu_xt); %Predicted values
        %Equation 8: (Actually computing updated covariance matrix)
        Z_hat_t=(H_yt'*yt)+(H_xt'*X_hat_t);
        term1=(Z_hat_t-mu)*((Z_hat_t-mu)');
        term2=H_xt'*(R_xt-(R_xtyt*(inv(R_yt))*((R_xtyt)')))*H_xt;
        sumTerm3=term1+sumTerm3;
        sumTerm4=term2+sumTerm4;
        %%Gather RMSE/MMSE
        %Below replicates what is described in Section 3 subsection B Eq 12
        H_ytT=getHyt(usrT,id_mat); %Observed TARGET values*
        specP=H_ytT*Z_hat_t; %Specific values corresponding to target vals
        yt_T=H_ytT*usrT; %Target vals of this user. Cmpare agnst pred vals
        lt=size(H_ytT,1); %Number of classifications made this iteration
        t_terms=yt_T-specP; %Top terms (Equation 12)
        totalLen=lt+totalLen;
        sum_t_terms=sum_t_terms+((t_terms)'*(t_terms));
    end
    %Calculating this iteration's error
    errSq=sum_t_terms./totalLen;
    RMSE=sqrt(errSq)
    %Calculating next iteration's covariance
    R=(sumTerm3+sumTerm4)./(size(Z,2)); %Where size(Z,2) is n
    
    %Check if convergence criterion was met. If it was, stop iterating.
    %Note that this convergence criterion is not exactly as described in
    %equation 13, however it is similar and accomplishes the same purpose.
    RMSEs=vertcat(RMSEs,RMSE);
    if Last_RMSE-RMSE<=.0001 %Convergence condition
        MMSE=RMSE; %Lowest RMSE
        break; 
    end
    Last_RMSE=RMSE;
end
RMSEs(1,:)=[]; %Clear zero row;
plot(RMSEs);
ylabel('RMSE');
xlabel('Iteration');
title('RMSE vs Iteration of EM Algorithm');





%%TODO:
%Clipping predicted values over 5 to 5 and under 1 to 1
%Convergence criterion based on PDF rather than change in RMSE






