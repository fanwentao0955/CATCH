function [BxTrain,ByTrain,BxTest,ByTest] = CATCH(X, Y, L, param, XTest, YTest)

G= NormalizeFea(L,1);
X=X'; Y=Y'; L=L'; G=G';

c = size(L,1);
n = size(L,2);
beta = param.beta;
gamma = param.gamma;
theta = param.theta;
r = param.nbits;

sel_sample = Y(:,randsample(n, 2000),:);
[pcaW, ~] = eigs(cov(sel_sample'), r);
V = pcaW'*Y;
B = sign(V);
B(B==0) = -1;

C = NormalizeFea(L',0)'*NormalizeFea(L',0);
A = rand(r,c);
D = A;


for iter = 1: 20 
    
    A = (D*D'+beta*eye(size(D,1)))\(D*C+beta*D);
    D = (A*A'+beta*eye(size(A,1)))\(A*C+beta*A); 
    
end


for iter = 1:8 
    
    B = sign(r*theta*V*G'*G);
    
    K = A*L + theta*r*B*G'*G;
    K = K';
    Temp = K'*K-1/n*(K'*ones(n,1)*(ones(1,n)*K));
    [~,Lmd,OO] = svd(Temp); clear Temp
    idx = (diag(Lmd)>1e-4);
    O = OO(:,idx); O_ = orth(OO(:,~idx));
    Pt = (K-1/n*ones(n,1)*(ones(1,n)*K)) *  (O / (sqrt(Lmd(idx,idx))));
    P_ = orth(randn(n,r-length(find(idx==1))));
    V = sqrt(n)*[Pt P_]*[O O_]';
    V = V';  
    
end


M1 = rand(size(B));
M2 = rand(size(B));
Q1 = B;
Q2 = B;

for iter = 1:20 
   
    T1 = B + Q1.*M1;
    T2 = B + Q2.*M2;

    H1 = T1*X'/(X*X'+ gamma*eye(size(X,1)));
    H2 = T2*Y'/(Y*Y'+ gamma*eye(size(Y,1)));

    R1 = H1*X-B;
    R2 = H2*Y-B;

    M1 = max(R1.*Q1,0);
    M2 = max(R2.*Q2,0);

end


clear X Y

BxTest = double(XTest*H1'>0);
ByTest = double(YTest*H2'>0);
BxTrain = double(B > 0)';
ByTrain = double(B > 0)';
    
end