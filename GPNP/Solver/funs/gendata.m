function [X,Xt,y,ye,xe,A,out]=gendata(n,p,K,sigma,ratio,seednum,matrixtype)
%========================================================================%
% INPUTS:                                                                %
%         n   ---- number of samples                                     %
%         p   ---- signal length                                         %
%         K   ---- number of nonzero elements in the signal              %
%      ratio  ---- range of value in the signal (= 10^ratio)             %
%      sigma  ---- noise variance                                        %
%     seednum ---- seed number                                           %
% matrixtype  ---- type of sample matirx                                 %
%          'gauss',                                                      %
%          'bernoulli'                                                   %
%          'partdct'                                                     %
%          'real1d'                                                      %
%          'real2d'                                                      %
% OUTPUTS:                                                               %
%       X ---- nomalized sample matrix                                   %
%      Xt ---- transpose of X                                            %
%     y   ---- data vector with noise                                    %
%    ye   ---- data vector without noise                                 %
%    xe   ---- true signal                                               %
%    A    ---- the support of xe                                         %
%   out   ---- a structure                                               %
% out.W   ---- the transpose of Wavelet transform                        %
% out.I   ---- the original imgae                                        %
%out.Ibar ---- the mean of Image                                         %
%------------------------------------------------------------------------%
% (c) By Yuling Jiao (yulingjiaomath@whu.edu.cn)                         %
%    and Bangti Jin  (bangti.jin@gmail.com)                              %
% Created on Oct 17, 2013                                                %
%========================================================================%
disp('Data is generating...')
rand('seed',seednum);   % fix seed
randn('seed', seednum); % fix seed
out = [];
% generate signal
xe = zeros(p,1);     % true signal
q = randperm(p);
A = q(1:K);
if ratio ~= 0
    vx = ratio*rand(K,1);
    vx = vx-min(vx);
    vx = vx/max(vx)*ratio;
    xe(A) = 10.^vx.*sign(randn(K,1));
else
    xe(A) = sign(randn(K,1));
end

% generate matrix X
switch matrixtype
    case 'gauss'
        X = randn(n,p);
        X = normalize(X);
        Xt = X';
        % generate right hand side
        ye  = X*xe;
        y   = ye + sigma*randn(n,1);
        A = find(xe);
    case 'bernoulli'
        X = sign(randn(n,p));
        X = normalize(X);
        Xt = X';
        % generate right hand side
        ye  = X*xe;
        y   = ye + sigma*randn(n,1);
        A = find(xe);
    case 'partdct'
        Idx = randperm(p);
        Idx = Idx(1:n);
        S = @(x) x(Idx,:);
        St = @(x) upsam(x,Idx,p);
        XF = @(x) S(dct(x));
        XFt = @(x) idct(St(x));
        X = XF;
        Xt = XFt;
        % generate observation
        ye  = X(xe);
        y   = ye + sigma*randn(n,1);
        A = find(xe);
    case 'real1d'
        imgsize = 32;
        nsam = 28;
        [X,Xt,y,ye,xe,A,out] = getrealdata(imgsize,nsam,sigma,matrixtype);
    case  'real2d'
        imgsize = 256;
        nsam = 60; %  50 for (256*256) is ok!
        [X,Xt,y,ye,xe,A,out] = getrealdata(imgsize,nsam,sigma,matrixtype);
    otherwise
        disp('Undefined matrix type')
end
disp('Generating  is done')
end

% subfunctions
function [sX,d] = normalize(X)
%------------------------------------------------------------------------%
% Copyright(c) by Yuling Jiao(yulingjiaomath@whu.edu.cn)                 %
% Created on Oct 17, 2013                                                %
%------------------------------------------------------------------------%
[n,p] = size(X);
d = zeros(p,1);
sX = X;
for k =1:p
    Xk = X(:,k);
    dk = 1/norm(Xk);
    d(k) = dk;
    sX(:,k) = Xk/norm(Xk);
end
end

function upz = upsam(z,idx,nn)
%------------------------------------------------------------------------%
% Copyright(c) by Yuling Jiao(yulingjiaomath@whu.edu.cn)                 %
% Created on Oct 17, 2013                                                %
%------------------------------------------------------------------------%
upz = zeros(nn,1);
upz(idx) = z;
end

function [X,Xt,y,ye,xe,A,out] = getrealdata(imgsize,nsam,sigma,matrixtype)
switch matrixtype
    case 'real1d'
        mx = imgsize;
        my = mx;
        N = mx^2;
        h = 1/N;
        mesh = (h/2:h:1-h/2)';
        out.mesh = mesh;
        Img  = (.75*(.2<mesh&mesh<.35)) + 10*(.25*(.5<mesh&mesh<.6)) + ((.75<mesh&mesh<.85).*sin(20*pi*mesh).^2);
        out.I = Img;
        Img = reshape(Img,mx,my);
        dW_L = 2;                    % levels of wavelet transform
    otherwise
        Img = phantom(imgsize);
        dW_L = 4 ;
        out.I = Img;
end
[my,mx] = size(Img);
Imgbar = mean(Img(:));
Imgcntr = Img -Imgbar;
%  Sampling operator
[M,Mh,mh,mhi] = LineMask(nsam,mx);
OMEGA = mhi;
Phi = @(z) A_fhp(z,OMEGA);      % z is same size of original image, out put is a vector.
Phit = @(z) At_fhp(z,OMEGA,mx); % z is a vector, out put is same size of original image.
% taking measurements
ye = Phi(Imgcntr);
M = length(ye(:));
% Effective sensing operator
wav = 'db1';                 % type of wavelet
[lo,hi,rlo,rhi] = wfilters(wav);
X = @(z) HDWIdW(Phi,Phit,z,lo,hi,rlo,rhi,1,1,dW_L,0);
Xt = @(z) HDWIdW(Phi,Phit,z,lo,hi,rlo,rhi,1,1,dW_L,1);
W = @(z) WaveDecRec(z,dW_L,lo,hi,rlo,rhi,1,1,0);  % Rec
Wt = @(z) WaveDecRec(z,dW_L,lo,hi,rlo,rhi,1,1,1); % Dec
xe =  Wt(Imgcntr);
xe = xe(:);
A = find(xe);
randn('state',0)
noise = sigma*randn(M,1);
y = ye + noise;
test=Xt(zeros(M,1)); clear test; 

out.Ibar = Imgbar;
out.W = W;
end

function [M,Mh,mi,mhi] = LineMask(L,N)
% Returns the indicator of the domain in 2D fourier space for the
% specified line geometry.
% Usage :  [M,Mh,mi,mhi] = LineMask(L,N)
%
% Written by : Justin Romberg
% Created : 1/26/2004
% Revised : 12/2/2004
thc = linspace(0, pi-pi/L, L);
%thc = linspace(pi/(2*L), pi-pi/(2*L), L);
M = zeros(N);
% full mask
for ll = 1:L
    
    if ((thc(ll) <= pi/4) || (thc(ll) > 3*pi/4))
        yr = round(tan(thc(ll))*(-N/2+1:N/2-1))+N/2+1;
        for nn = 1:N-1
            M(yr(nn),nn+1) = 1;
        end
    else
        xc = round(cot(thc(ll))*(-N/2+1:N/2-1))+N/2+1;
        for nn = 1:N-1
            M(nn+1,xc(nn)) = 1;
        end
    end
    
end
% upper half plane mask (not including origin)
Mh = M;
Mh(N/2+2:N,:) = 0;
Mh(N/2+1,N/2+1:N) = 0;
M = ifftshift(M);
mi = find(M);
Mh = ifftshift(Mh);
mhi = find(Mh);
end

function y = A_fhp(x, OMEGA)
% Takes measurements in the upper half-plane of the 2D Fourier transform.
% x - N vector
% y - K vector = [mean; real part(OMEGA); imag part(OMEGA)]
% OMEGA - K/2-1 vector denoting which Fourier coefficients to use
%         (the real and imag parts of each freq are kept).
% Written by: Justin Romberg, Caltech and Modified by Yuling Jiao
[s1,s2] = size(x);
n = round(sqrt(s1*s2));
yc = 1/n*fft2(x);
y = [yc(1,1); sqrt(2)*real(yc(OMEGA)); sqrt(2)*imag(yc(OMEGA))];
end

function x = At_fhp(y, OMEGA, n)
% Adjoint of At_fhp (2D Fourier half plane measurements).
% y - K vector = [mean; real part(OMEGA); imag part(OMEGA)]
% OMEGA - K/2-1 vector denoting which Fourier coefficients to use
%         (the real and imag parts of each freq are kept).
% n - Image is nxn pixels
% x - N vector
% Written by: Justin Romberg, Caltech and modified by Yuling Jiao

K = length(y);
fx = zeros(n,n);
fx(1,1) = y(1);
fx(OMEGA) = sqrt(2)*(y(2:(K+1)/2) + 1i*y((K+3)/2:K));
x = real(n*ifft2(fx));
end

function [Y,strc] = WaveDecRec(X,level,lo,hi,rlo,rhi,emd,ds,isdec)
persistent s;
if isdec == 1
    % Phi' * X
    [Y,s] = mwavedec2(X,level,lo,hi,emd,ds);
    strc = s;
    Y = Y';
else
    Y  = mwaverec2(X(:),s,rlo,rhi,emd,0);
    strc = s;
end
end

function Y = HDWIdW(A,At,X,lo,hi,rlo,rhi,emd,ds,L,isdec)
% X: sparse vector which is taken from Wavelet transform of an image
% A: random projection matrix M x N
% rlo,rhi: scaling filter
% L: level of decomposition
% m, n: size of image
% Return Y: vector M x 1
% Written by Kun Qiu, ISU, April 2009 and modified by Yuling Jiao
persistent s
if isdec == 1 % Ht
    % converting measurements into samples
    if ~isa(At, 'function_handle')
        Y=At*X;
    else
        Y=At(X);
    end
    % converting samples into wavelet coefficients (sparse representation)
    [Y,s]= mwavedec2(Y,L,lo,hi,emd,ds);
    Y = Y';
else
    Y  = mwaverec2(X(:),s,rlo,rhi,emd,0);
    if ~isa(A, 'function_handle')
        Y = A*Y;
    else
        Y = A(Y);
    end
end
end