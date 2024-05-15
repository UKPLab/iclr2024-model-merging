#!/usr/bin/octave -q
%%%% generate the delta and deltadelta of a given feature file

%%%% add a path to find the other scripts
CN_SETUP_PATH = getenv("CN_SETUP_PATH");
if (isempty(CN_SETUP_PATH))
	addpath('/u/plahl/environments/setup/common/bin/training/pitch.octave')
else
	addpath(fullfile(CN_SETUP_PATH, 'bin/training/pitch.octave'))
end

%%%% check number of paramters
if (length(argv) != 1)
        % printf ("%s", program_name);
        disp('usage: calcDelta.m <feature-file>')
        exit
end

%%%% read the raw pitch feature and
%rawFile = argv{1};
rawFile = argv(){1};
%%smoothedFile = argv{2};

%%%% load the raw pitch feature
x = load(rawFile);

%%%% compute the delta and double delta of the pitch features
dx = delta(x);
ddx = delta(dx);
    
%%%% write the pitch features to stdout or file
%if 
	for j = 1:length(x),
		printf('%10.6f %10.6f %10.6f\n', x(j), dx(j), ddx(j));
	end
%else
%	fid = fopen(smoothedFile,'w');
%	fprintf(fid, '%10.6f\n', y);
%	fclose(fid);
%end
