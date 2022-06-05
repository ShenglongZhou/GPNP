% This code demonstrates the success recovery rate for 
% compressed sensing and quadratic compressed sensing
clc; clear; close all; 
addpath(genpath(pwd));

test    = 1; 
noS     = 100;
problem = {'CS','QCS'};
switch  test
case 1; n = 256; m = 64; sm = 5:2:35;
case 2; n = 120; m = 80; sm = 3:15;
end    
    
SucRate      = [];
pars.disp    = 0;
pars.draw    = 0;
for j        = 1:length(sm)
    rate     = 0; 
    s        = sm(j);
 
    for S    = 1:noS         
        data = CSdata(problem{test},m,n,s); 
        switch  test 
            case 1;  out = GPNPcs(n,s,data.b,data.A,data.A',pars);  
            case 2;  out = GPNPqcs(m,n,s,data.b,data.A,pars);
                     if  norm(out.x-data.xtrue) > norm(out.x+data.xtrue) 
                         out.x = - out.x;
                     end   
        end
        clc; SucRate     
        rate = rate + (norm(out.x-data.xtrue)/norm(data.xtrue)<1e-2); 
        
    end
    clc; SucRate  = [SucRate  rate]  
    
    figure(1)
    set(gcf, 'Position', [1000, 200, 400 350]);
    xlab = {'s','m/n'};
    plot(sm(1:j),SucRate/noS,'r*-','LineWidth',1),hold on
    xlabel(xlab{test}), ylabel('Success Rate') 
    axis([min(sm) max(sm) 0 1]); grid on; 
    legend('GPNP','Location','NorthEast'); hold on, pause(0.1)
    
end


