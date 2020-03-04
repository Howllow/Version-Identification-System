function [score, data] = testtempo(tmean, tsd, data)
% [score, data] = testtempo(tmean, tsd, data)
%   Run tempo estimation over all the tempo contest test data
%   to allow tuning of parameters.
% 2006-09-06 dpwe@ee.columbia.edu  mirex06 tempo

ntest = 20;

testpath = '../mirex06train';

if nargin < 3 
  % need to recalc data matrix
  for i = 1:ntest
    % Load the audio
    [d,sr] = wavread(fullfile(testpath, ['train',num2str(i), '.wav']));
    [t,rxc,D,fmm] = tempo(d,sr,tmean,tsd);
    data.rxc{i} = rxc;
    data.fmm{i} = fmm;
    data.t{i} = t;
  end
  % and reload the ground truth
  for i = 1:ntest
    data.tempo{i} = textread(fullfile(testpath, ['train',num2str(i), '-tempo.txt']));
  end
else
  for i = 1:ntest
    [t,v] = tempo([],0,tmean,tsd,data.fmm{i});
    data.t{i} = t;
  end
end

% do scoring
score = 0;

for i = 1:ntest
  
  gt = data.tempo{i};
  
  pd1 = gt(1);
  pd2 = gt(2);
  st1 = gt(3);
  st2 = 1 - st1;
  
  % Did we get either/both periods?
  so = data.t{i};
  sst1 = so(3);
  stt = so([1 2]);
  tt1 = min(abs(pd1 - stt)/pd1) < 0.08;
  tt2 = min(abs(pd2 - stt)/pd2) < 0.08;
  
  pscore = st1*tt1 + st2*tt2;
  score = score + pscore;

  disp(['tr',num2str(i,'%02d'),' ref=',num2str(gt,'%.2f '),...
        ' sys=',num2str(so,'%.2f '),...
        ' pscore=', num2str(pscore,2)]);
  
end

score = score/ntest;

disp(['total score = ', num2str(score, 3)]);

% >> [sc, da] = testtempo(120,1.4,da); 
% >> sc
% sc =
%     0.769
