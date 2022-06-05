function out = GPNPcs(n,s,b,A,At,pars)
% A solver for sparsity constrained compressive sensing:
%
%      min ||Ax-b||^2,  s.t. ||x||_0<=s,
%
% where A in R^{m-by-n}, b in R^{m} and s<<n.
% =========================================================================  
% Inputs: 
%     n       : Dimension of the solution x              (Required)
%     s       : Sparsity level of x, an integer in (0,n) (Required)
%     A       : The measurement matrix in R^{m-by-n}  
%               or the function handle A=@(x)A(x)        (Required)
%     At      : The transpose of A, (i.e., A=A')      
%               or the function handle At=@(x)At(x)      (Required)
%     b       : The observation vector in R^m            (Required) 
%     pars:     Parameters are all OPTIONAL
%               pars.disp  --  Display or not results at each step (default 1) 
%               pars.maxit --  Maximum nonumber of iteration  (default 10000) 
%               pars.tol   --  Tolerance of stopping criteria (default 1e-6) 
%               pars.obj   --  The provided objective (default 1e-20)
%                              Useful for noisy case.
% Outputs:
%     out.x:             The sparse solution x 
%     out.obj:           ||Ax-b||^2
%     out.time           CPU time
%     out.iter:          Number of iterations
% =========================================================================  
% This code is programmed based on the algorithm proposed in 
% S. Zhou,  2022, Applied and Computational Harmonic Analysis,
% Gradient projection newton pursuit for sparsity constrained optimization
%%%%%%%    Send your comments and suggestions to                     %%%%%%
%%%%%%%    slzhou2021@163.com                                        %%%%%% 
%%%%%%%    Warning: Accuracy may not be guaranteed!!!!!              %%%%%%
% =========================================================================  

warning off;

if nargin<5; fprintf(' Error!!!\n Inputs are not enough!!!\n'); return; end
if nargin<6; pars=[]; end

[sigma,J,flag,m,alpha0,gamma,thd,disp,tol,tolF,maxit]...
          = set_parameters(s,n,A,b,pars);
t0        = tic;
x         = zeros(n,1);
xo        = zeros(n,1);
bt        = b';
Fnorm     = @(var)norm(var)^2;
funhd     = isa(A,'function_handle');
if  funhd
    ATu   = A;
    gx    = At(b);
    Atb   = gx;
    OBJ   = zeros(3,1);
    suppz = @(z,t)supp(z,t);
    sub   = @(z,t)z(t,:);
    subH  = @(z,T)(sub(At(A(suppz(z,T))),T));     
else
    gx    = At*b; 
    Atb   = gx;
    OBJ   = zeros(5,1);
end
fx        = Fnorm(b);
[~,Tx]    = maxk(gx,s,'ComparisonMethod','abs');
Tx        = sort(Tx);  
minobj    = zeros(maxit,1);
minobj(1) = fx;

% main body
if  disp 
    fprintf(' Start to run the solver -- GPNP \n');
    fprintf(' --------------------------------------\n');
    fprintf(' Iter         ||Ax-b||           Time \n'); 
    fprintf(' --------------------------------------\n');
end
 
for iter = 1:maxit     
     
    % Line search for setp size alpha
 
    alpha  = alpha0;  
    for j  = 1:J
        [subu,Tu] = maxk(x-alpha*gx,s,'ComparisonMethod','abs');
        u         = xo; 
        u(Tu)     = subu;  
        if ~funhd  
            ATu   = A(:,Tu);
        end     
        Aub       = Axb(ATu,subu,u);   
        fu        = Fnorm(Aub);  
        if fu     < fx - sigma*Fnorm(u-x); break; end
        alpha     = alpha*gamma;        
    end
   
    gx      = AtAxb(Aub);
    normg   = Fnorm(gx);
    x       = u;
    fx      = fu; 
    
    % Newton step
    sT   = sort(Tu); 
    mark = nnz(sT-Tx)==0;
    Tx   = sT;
    eps  = 1e-4;
    if  mark || normg < 1e-4 || (alpha0==1 && ~funhd)
        v     = xo; 
        if funhd  
           cgit     = min(20,5*iter);   
           subv     = my_cg(@(var)subH(var,Tu),Atb(Tu),1e-10*n,cgit,zeros(s,1));
        else 
           if  s   <  2000 && m <= 2e4
               subv = (ATu'*ATu)\(bt*ATu)'; 
               eps  = 1e-10;
           else
               cgit = min(20,2*iter);  
               subv = my_cg(@(var)((ATu*var)'*ATu)',Atb(Tu),1e-30,cgit,zeros(s,1)); 
           end           
        end 
        v(Tu)       = subv; 
        Avb         = Axb(ATu,subv,v); 
        fv          = Fnorm(Avb);  
        if fv      <= fu  - sigma * Fnorm(subu-subv)
           x        = v;  
           fx       = fv;
           subu     = subv;  
           gx       = AtAxb(Avb); 
           normg    = Fnorm(gx); 
        end   
    end
    
    % Stop criteria  
	error     = Fnorm(gx(Tu)); 
    obj       = sqrt(fx);
    OBJ       = [OBJ(2:end); obj];
    if disp 
       fprintf('%4d         %9.4e      %7.3fsec\n',iter,fx,toc(t0)); 
    end

    maxg      = max(abs(gx));
    minx      = min(abs(subu));
    J         = 8;
    if error  < tol*1e3 && normg>1e-2 && iter < maxit-10
       J      = min(8,max(1,ceil(maxg/minx)-1));     
    end  

    if isfield(pars,'obj')  && obj <=tolF && flag
        maxit  = iter + 100*s/n; 
        flag   = 0; 
    end
 
    minobj(iter+1) = min(minobj(iter),fx);  
    if fx    < minobj(iter) 
        xmin = x;  
        fmin = fx;  
    end
    
    if iter  > thd 
       count = std(minobj(iter-thd:iter+1) )<1e-10;
    else
       count = 0; 
    end

    if  normg<tol || fx < tolF  || ...
        count  || (std(OBJ)<eps*(1+obj)) 
        if count && fmin < fx; x=xmin; fx=fmin; end
        break; 
    end  
    
end

out.x     = x;
out.obj   = fx;
out.iter  = iter;
out.time  = toc(t0);

if  disp
    fprintf('---------------------------------------\n');
end
if  fx<1e-10 && disp
    fprintf(' A global optimal solution may be found\n');
    fprintf(' because of ||Ax-b|| = %5.3e!\n',sqrt(fx)); 
    fprintf('---------------------------------------\n');
end

%--------------------------------------------------------------------------
function diff = Axb(AT,xT,x)
     if  funhd
         diff = A(x)-b; 
     else
         diff = AT*xT-b;
     end
end

%--------------------------------------------------------------------------
function grad = AtAxb(Axb)
     if  funhd
         grad = At(Axb);
     else
         grad = At*Axb;
     end
end

%-------------------------------------------------------------------------- 
function z = supp(x,T)
      z    = zeros(n,1);
      z(T) = x;
end
end


% set parameters-------------------------------------------------------
function [sigma,J,flag,m,alpha0,gamma,thd,disp,tol,tolF,maxit]=set_parameters(s,n,A,b,pars)
    sigma     = 1e-4; 
    J         = 1;    
    m         = length(b);
    flag      = 1;
    alpha0    = 5;
    gamma     = 0.5;
    funhd     = isa(A,'function_handle');  
    if m/n   >= 1/6 && s/n <= 0.05 && n >= 1e4 && ~funhd
       alpha0 = 1; 
       gamma  = 0.1; 
    elseif funhd
       gamma  = 0.1;  
    end
    if s/n   <= 0.05
       thd    = ceil(log2(2+s)*50); 
    else
        if  n    > 1e3 
            thd  = 100;
        elseif n > 500
            thd  = 500;
        else
            thd  = ceil(log2(2+s)*750);
        end
    end   
    
    if isfield(pars,'disp');  disp   = pars.disp;   else; disp  = 1;     end
    if isfield(pars,'tol');   tol    = pars.tol;    else; tol   = 1e-10; end  
    if isfield(pars,'obj');   tolF   = pars.obj;    else; tolF  = 1e-20; end 
    if isfield(pars,'maxit')
        maxit = pars.maxit;  
    else 
        maxit = (n<=1e3)*1e4 + (n>1e3)*5e3; 
    end   
end

% conjugate gradient-------------------------------------------------------
function x = my_cg(fx,rhs,cgtol,cgit,x)
    r = rhs;
    e = sum(r.*r);
    t = e;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        if  i == 1  
            p = r;
        else
            p = r + (e/e0)*p;
        end  
        w  = fx(p);
        a  = e/sum(p.*w);
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = sum(r.*r);
    end       
    
end
