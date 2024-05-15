#!/usr/bin/octave -q

%%%% generate pitch features based on the raw pitch for input file

my_dir = fileparts(mfilename('fullpath'));
addpath(my_dir);

%%%% check number of paramters
if (length(argv) != 1)
	disp('usage: genpitch.m <raw-pitch-file>')
	exit
end

%%%% global values, here interpolation-window-length
lpn_window = 200;

%%%% read the raw pitch feature and
rawPitchFile = argv(){1};

%%%% load the first row of raw pitch feature, perform spline interpolation
x = load(rawPitchFile);
x = x(:,1);
n = length(x);

%%%% make computation deterministic by seeding the RNG with the number of 
%%%% frames for each waveform
rand("seed", n);
    
%%%% perform spline interpolation (use interpolation-window-length)
y = spline_interp(x, n);
y = log(y);

% long term pitch normalization (LPN)
if (lpn_window > 0)
	y = smooth(y) - smooth(y, lpn_window);
end
    
%%%% write the pitch features to stdout
printf('%10.6f\n', y);
