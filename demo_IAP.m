clear all
load('IAPRTC_12.mat')
%% parameter set
param.beta = 1e-2;
param.theta = 1e-3;
param.gamma = 1e-2;
nbitset     = [16,32,64,128];
eva_info    = cell(1,length(nbitset));
%% centralization
XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));
%% kernelization
[XKTrain,XKTest] = Kernelize(XTrain, XTest, 1500); [YKTrain,YKTest]=Kernelize(YTrain,YTest, 1500);
XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1)); XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1)); YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));
%% SC-CMH
for kk= 1:length(nbitset)
    
param.nbits = nbitset(kk);

[BxTrain, ByTrain, BxTest, ByTest] = CATCH(XKTrain, YKTrain, LTrain, param, XKTest, YKTest);

DHamm = pdist2(BxTest, ByTrain,'hamming');
[~, orderH] = sort(DHamm, 2);
eva_info_.Image_to_Text_MAP = mAP(orderH', LTrain, LTest);
 
DHamm = pdist2(ByTest, BxTrain,'hamming');
[~, orderH] = sort(DHamm, 2);
eva_info_.Text_to_Image_MAP = mAP(orderH', LTrain, LTest);

eva_info{1,kk} = eva_info_;
Image_to_Text_MAP = eva_info_.Image_to_Text_MAP;
Text_to_Image_MAP = eva_info_.Text_to_Image_MAP;

fprintf('CATCH %d bits -- Image_to_Text_MAP: %.4f ; Text_to_Image_MAP: %.4f ; \n',nbitset(kk),Image_to_Text_MAP,Text_to_Image_MAP);

end