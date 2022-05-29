function out = GPNPqcs(m,n,s,b,A,pars)
% A solver for sparsity constrained quadratic compressive sensing:
%
%      min ||(Ax).^2-b||^2/4/m,  s.t. ||x||_0<=s,
%
% where A in R^{m-by-n}, b in R^{m} and s<<n.
% =========================================================================  
% Inputs: 
%     n       : Dimension of the solution x              (Required)
%     s       : Sparsity level of x, an integer in (0,n) (Required)
%     A       : The measurement matrix in R^{m-by-n}  
%               or the function handle A=@(x)A(x)        (Required)
%     b       : The observation vector in R^m            (Required) 
%     pars:     Parameters are all OPTIONAL
%               pars.disp  --  Display or not results at each step (default 1) 
%               pars.maxit --  Maximum nonumber of iteration  (default 10000) 
%               pars.tol   --  Tolerance of stopping criteria (default 1e-6) 
%               pars.obj   --  The provided objective (default 1e-20) 
%                              Useful for noisy case.
% Outputs:
%     out.x:             The sparse solution x 
%     out.obj:           ||(Ax).^2-b||^2/4/m
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

[sigma,J,alpha0,alpha00,gamma,thd,disp,tol,tolF,maxit]...
          = set_parameters(s,n,pars);
      
t0        = tic;      
funcs     = @(x,key,T)QCS(x,key,T);
xo        = zeros(n,1);
Atb       = b'*A;
[~,Tx]    = maxk(Atb,s,'ComparisonMethod','abs');
x         = xo;
x(Tx)     = ones(s,1)/s;

Fnorm     = @(var)norm(var)^2;
gx        = funcs(x(Tx),'grad',Tx); 
OBJ       = zeros(5,1);  
fx        = funcs(x(Tx),'obj',Tx);  
[~,Tx]    = maxk(gx,s,'ComparisonMethod','abs');
Tx        = sort(Tx);

minobj    = zeros(maxit,1);
minobj(1) = fx;
% main body
if  disp 
    fprintf(' Start to run the solver -- GPNP \n');
    fprintf(' --------------------------------------\n');
    fprintf(' Iter        objective           Time \n'); 
    fprintf(' --------------------------------------\n');
end
 
for iter = 1:maxit     
     
    % Line search for setp size alpha
 
    alpha  = alpha0;  
    for j  = 1:J
        xg        = x-alpha*gx;
        [subu,Tu] = maxk(xg,s,'ComparisonMethod','abs');
        u         = xo; 
        u(Tu)     = subu;  
        fu        = funcs( subu,'obj',Tu);      
        if fu     < fx - sigma*Fnorm(u-x); break; end
        alpha     = alpha*gamma;        
    end
   
    gx      = funcs( subu,'grad',Tu);
    normg   = Fnorm(gx); 
    x       = u;
    fx      = fu; 
    
    % Newton step
    sT   = sort(Tu); 
    mark = nnz(sT-Tx)==0; 
    
    Tx   = sT;
    if ( mark || normg < 1e-4  ) && iter>1
        v        = xo; 
        subH     = funcs(subu,'hess',Tu);
        diff     = my_cg(subH,-gx(Tu),1e-20*n,20,zeros(s,1));
        subv     = diff + subu;  
        v(Tu)    = subv; 
        fv       = funcs(subv,'obj',Tu); 
        if fv   <= fu  - sigma * Fnorm(diff)  
           x     = v;  
           fx    = fv; 
           subu  = subv;  
           gx    = funcs(subv,'grad',Tu);
           normg = Fnorm(gx); 
        end   
    end
    
    % Stop criteria  
	error     = Fnorm(gx(Tu)); 
    obj       = fx;
    OBJ       = [OBJ(2:end); obj];
  
    if disp 
       fprintf('%4d         %9.4e      %7.3fsec\n',iter,obj, toc(t0)); 
    end
    
    maxg      = maxk(abs(gx),s); 
    maxg      = maxg(s);
    minx      = min(abs(subu));
    J         = 10;
    if error  < tol*1e3 && normg>1e-3 && iter < maxit-10
       J      = min(10,max(1,ceil(maxg/minx)-1));    
       alpha0 = minx/maxg/gamma^J;  
    else
       alpha0 = alpha00;
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
        count  || (std(OBJ)<1e-18*(1+obj)) 
        if count && fmin < fx; x=xmin; fx=fmin; end
        break; 
    end  
    
end

if isfield(pars,'xtrue') 
    x     = bestMatch(x,pars.xtrue); 
    if  norm(x-pars.xtrue) > norm(x+pars.xtrue) 
        x = -x;
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
    fprintf(' A global optimal solution is found\n');
    fprintf(' because of objective = %5.3e!\n', fx); 
    fprintf('---------------------------------------\n');
end

function func = QCS(x,key,T)
    AT     = A(:,T);
    Ax     = AT*x;
    Ax2    = (Ax).^2;  
    tmp    = Ax2-b; 
    switch key
        case 'obj'
              func  = Fnorm(tmp)/m/4;  
        case 'grad'
              func  = ((Ax.*tmp)'*A)'/m;
        case 'hess'    
              func  = @(z)( ((3*Ax2-b).*(AT*z))'*AT )'/m;        
    end
end

end

% set parameters-------------------------------------------------------
function [sigma,J,alpha0,alpha00,gamma,thd,disp,tol,tolF,maxit]=set_parameters(s,n,pars)
    sigma   = 1e-4;
    J       = 10;
    alpha0  = 2;
    alpha00 = alpha0;
    gamma   = 0.25;
    if s/n <= 0.03
       thd  = ceil(log2(2+s)*50);
    else
       thd  = ceil(log2(2+s)*750)*(n<=1e3)+500*(n>1e3 );
    end   
    if isfield(pars,'tol');   tol  = pars.tol;  else; tol  = 1e-10; end  
    if isfield(pars,'obj');   tolF = pars.obj;  else; tolF = 1e-20; end  
    if isfield(pars,'disp');  disp = pars.disp; else; disp = 1;     end
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
        a  = e/(sum(p.*w)+1e-20);
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = sum(r.*r);
    end       
   
end

function xBest=bestMatch(x1,x2)
% bestMatch finds the permutation of x1 that matches x2 the best in
% terms of the following ambiguities: circular shift, sign, flipping
[n,~]=size(x1);
minErr=inf;
for kk=1:n
    for signInd=1:2
        for flip=0:1
            if (flip)
                x1shift=flipud(circshift(x1,kk)*(-1)^signInd);
            else
                x1shift=circshift(x1,kk)*(-1)^signInd;
            end
            err=norm(x2-x1shift);
            if err<minErr
                xBest=x1shift;
                minErr=err;
            end
        end
    end
end
end