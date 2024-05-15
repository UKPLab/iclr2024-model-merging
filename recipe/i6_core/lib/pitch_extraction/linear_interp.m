function [y] = spline_interp(x,n)
% Interpolate the 1-D time series with piecewise cubic spline to length n
% Usage: [y] = spline_interp(x,n)
% In this case, the inputs and outputs are:
%   x -- input raw pitch values extracted by get_f0 
%   n -- total number of frames of MFCC/PLP features
%   y -- output spline interpolated pitch values
%
%  By Xin Lei, 12/22/2004

ind = find(x);      % find all the non-zero elements in x

% if all elements of x are 0, then return 0's
if length(ind) == 0
    y = rand(n, 1); return
end

%y1 = [x(ind(1)); x(ind); x(ind(end))];
%ind1 = [1; ind; n];
y1 = x(ind);

y = interp1(ind,y1,ind(1):ind(end),'linear');
y = [ones(1,ind(1)-1)*x(ind(1))+rand(1,ind(1)-1) y ...
    ones(1,n-ind(end))*x(ind(end))+rand(1,n-ind(end))];

y = y';

