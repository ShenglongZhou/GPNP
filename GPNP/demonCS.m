% demon compressed sensing problems 
clc; clear; close all;
addpath(genpath(pwd));
 
n          = 30000; 
m          = ceil(n/4);
s          = ceil(0.05*n); 
data       = CSdata('CS',m,n,s) ;
out        = GPNPcs(n,s,data.b,data.A,data.A'); 
  
fprintf(' Sample size:          %dx%d\n',   m,n);
fprintf(' Recovery time:        %.3fsec\n', out.time);
fprintf(' Objective value:      %5.2e\n',   out.obj);
fprintf(' True objective value: %5.2e\n',   data.obj);
RecoveryShow(data.xtrue,out.x,[1000, 450, 500 250],1)

 
