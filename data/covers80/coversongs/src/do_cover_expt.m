% do_cover_expt.m
%
% Run the top-level steps for a cover song experiment.
%
% 2007-06-15 Dan Ellis dpwe@ee.columbia.edu
% $Header: $

list1 = '../covers32k/list1.list';
list2 = '../covers32k/list2.list';
coverdir = '../covers32k/';
ftrsdir = '../ftrs32k/';

needftrs = 1;
if needftrs
  l1 = calclistftrs(list1,coverdir,'.mp3',ftrsdir,'.mat');
  l2 = calclistftrs(list2,coverdir,'.mp3',ftrsdir,'.mat');
end

tic; [R,S,T,C] = coverTestLists(l1,l2); toc
[vv,xx] = max(R');
ncovers = length(xx);
subplot(121)
imgsc(R);
gcolor
hold on;
plot(xx,[1:ncovers],'.r');
hold off;

ncorr = sum(xx==1:ncovers);

disp(['Got ',num2str(ncorr),' of ',num2str(ncovers),' = ', ...
      sprintf('%.1f',100*ncorr/ncovers),'% correct']);

% Elapsed time is 332.622730 seconds.
% Got 33 of 80 = 41.2% correct

