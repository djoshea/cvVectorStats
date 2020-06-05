function [ CI, CIDist ] = cvCISequential( fullDataStatistic, dataFun, dataCell, mode, alpha, nResamples )
    %Internal function used to switch between CI methods depending on user
    %input; this is called in each cv function.  
    if strcmp(mode, 'bootCentered') || strcmp(mode,'bootPercentile')
        [ CI, CIDist ] = cvBootCISequential( fullDataStatistic, dataFun, dataCell, mode, alpha, nResamples );
    elseif strcmp(mode, 'jackknife')
        [ CI, CIDist ] = cvJackknifeCISequential( fullDataStatistic, dataFun, dataCell, alpha );
    else
        error('Invalid CI Mode, should be bootCentered, bootPercentile, or jackknife');
    end
end

