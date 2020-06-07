function [ CI, jackS ] = cvJackknifeCISequential( fullDataStatistic, dataFun, dataCell, alpha )
    %This function uses the jackknife to compute a confidence interval. 
    
    %fullDataStatistic is the statistic computed on all of the data.
    
    %dataFun is a handle to the function that computes the statistic of
    %interest
    
    %dataCell is nClasses { nDims { nObs x T } } cell that is input into dataFun. 
    %Matching data points are removed row-wise from dataCell to compute leave-one-out statistics for
    %the jackknife. 
    %
    % in sequential mode each entry of dataCell{.} is a cell array of size nDims x 1
    %
    % CI is 2 x nOuts (x time)
            
    nClasses = numel(dataCell);
    nDims = numel(dataCell{1});
    
    obsMat = cellfun(@(x) size(x, 1), cat(2, dataCell{:})'); % nClasses x nDims
    minObs = min(obsMat, [], 'all');
    assert(minObs >= 3, 'Minimum of 3 observations per dimension required for jackknife');
    
    [~, smallFoldIndices, nFolds] = getSequentialFoldIndicatorMatrices(obsMat, 'strategy', 'min', 'smallAsMatrices', false); % C x D 

    jackS = zeros(nFolds,size(fullDataStatistic, 1), size(fullDataStatistic, 2));
    for f=1:nFolds
        deleteCell = dataCell;
        for c=1:nClasses
            for d = 1:nDims
                if iscell(smallFoldIndices{c, d})
                    idx_del = smallFoldIndices{c, d}{f};
                else
                    idx_del = smallFoldIndices{c, d}(f);
                end
                deleteCell{c}{d}(idx_del,:,:) = [];
            end
        end
        jackS(f,:,:) = shiftdim(dataFun( deleteCell{:} ), -1); % nStats Time --> 1 x nStats x Time     
    end

    ps = nFolds*shiftdim(fullDataStatistic, -1) - (nFolds-1)*jackS; % folds x nStats x time
    v = var(ps, [], 1); % 1 x nStats x time
    
    multiplier = norminv((1-alpha/2), 0, 1);
    CI = [(shiftdim(fullDataStatistic, -1) - multiplier*sqrt(v/nFolds)); (shiftdim(fullDataStatistic, -1) + multiplier*sqrt(v/nFolds))];
end

