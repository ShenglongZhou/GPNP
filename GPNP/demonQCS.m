% demon compressed sensing problems 
clc; clear; close all;
addpath(genpath(pwd));
 
n          = 1000; 
m          = ceil(0.8*n);
s          = ceil(0.01*n); 
data       = CSdata('QCS',m,n,s) ;
pars.disp  = 1;
out        = GPNPqcs(m,n,s,data.b,data.A,pars); 

if  norm(out.x-data.xtrue) > norm(out.x+data.xtrue) 
    out.x  = - out.x;
end   

fprintf(' Sample size:          %dx%d\n',   m,n);
fprintf(' Recovery time:        %.3fsec\n', out.time);
fprintf(' Objective value:      %5.2e\n',   out.obj);
fprintf(' True objective value: %5.2e\n',   data.obj);
RecoveryShow(data.xtrue,out.x,[1000, 454, 400 200],1)

 
