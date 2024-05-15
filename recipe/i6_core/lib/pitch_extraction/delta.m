function [d] = delta(x)
% Generate the HTK style delta coefficients using a delta window of 5
%  Usage: [d] = delta(x)
%  Input and output:
%   x -- input vector
%   d -- output delta vector

n = length(x);

if n == 1,
  d = 0;
elseif n == 2,
  d(1) = ( (x(2) - x(1)) * 2 + ( x(2) - x(1) ) ) / 10;
  d(2) = ( (x(2) - x(1)) * 2 + ( x(2) - x(1) ) ) / 10;
elseif n == 3,
  d(1) = ( (x(3) - x(1)) * 2 + ( x(2) - x(1) ) ) / 10;
  d(2) = ( (x(3) - x(1)) * 2 + ( x(3) - x(1) ) ) / 10;
  d(3) = ( (x(3) - x(1)) * 2 + ( x(3) - x(2) ) ) / 10;  
else
  % for the first and last 2 frames, treat with padding
  d(1) = ( (x(3) - x(1)) * 2 + ( x(2) - x(1) ) ) / 10;
  d(2) = ( (x(4) - x(1)) * 2 + ( x(3) - x(1) ) ) / 10;
  d(n-1) = ( 3 * x(n) - x(n-2) - 2 * x(n-3) ) / 10;
  d(n) = ( 3 * x(n) - x(n-1) - 2 * x(n-2) ) / 10;
  
  % for other elements, compute the delta according to HTK book equation
  % (5.16) in page 65, using \Theta = 2
  if n ~= 4,
    for i = 3:n-2,
      d(i) = ( (x(i+2) - x(i-2)) * 2 + (x(i+1) - x(i+1)) ) / 10;
    end
  end
  
end

d = d';
