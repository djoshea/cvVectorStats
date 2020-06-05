function [ CI, bootStats ] = cvBootCISequential( fullDataStatistic, dataFun, dataCell, mode, alpha, nResamples )
    %This function uses the bootstrap to compute a confidence interval. 
    
    %fullDataStatistic is the statistic computed on the full, non-resampled
    %data.
    
    %dataFun is a handle to the function that computes the statistic of
    %interest.
    
    % dataCell is nClasses { nDims { nObs x T } } cell that is input into dataFun. Data points are
    %resampled with replacement (row-wise) within each element of dataCell separately.
    %This ensures the same number of trials for each class of data. 
    
    nClasses = numel(dataCell);
    nDims = numel(dataCell{1});
    
    bootStats = zeros(nResamples,size(fullDataStatistic, 2), size(fullDataStatistic, 3));
    for n=1:nResamples
        resampledCell = dataCell;
        for c=1:nClasses
            for d = 1:nDims
                resampleIdx = randi(size(dataCell{c}{d},1), size(dataCell{c}{d},1), 1);
                resampledCell{c}{d} = dataCell{c}{d}(resampleIdx,:,:);
            end
        end
        
        bootStats(n,:,:) = dataFun(resampledCell{:});
    end
    
    if strcmp(mode, 'bootCentered')
        cenStats = bootStats - mean(bootStats);
        cenStats = cenStats + fullDataStatistic;
        CI = prctile(cenStats, 100*[alpha/2, 1-alpha/2], 1);
    elseif strcmp(mode, 'bootPercentile')
        CI = prctile(bootStats, 100*[alpha/2, 1-alpha/2], 1);
    else
        error('Invalid mode, should be bootCentered or bootPercentile');
    end
end

