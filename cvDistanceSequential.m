function [ euclideanDistance, squaredDistance, CI, CIDistribution ] = cvDistanceSequential( class1, class2, subtractMean, CIMode, CIAlpha, CIResamples )
    %This function estimates the distance between the means of two
    %distributions.

    %inputs:
    
    %class1 and class2 are D { N_i x T } , where D is the number of
    %dimensions and N_i is the number of samples for dimension i

    %If subtractMean is true, this will center each vector
    %before computing the size of the difference (default is off).
    
    %CIMode can be none, bootCentered, bootPercentile, or jackknife
    
    %CIAlpha sets the coverage of the confidence interval to
    %100*(1-CIAlpha) percent
    
    %CIResamples sets the number of bootstrap resamples, if using bootstrap
    %mode (as opposed to jackknife)
    
    %CIDistribution is the distribution of bootstrap statistics or
    %jackknife leave-one-out statistics
    
    %outputs:
    
    %The first column of CI corresponds to euclidean distance, the second
    %column corresponds to squared distance. 
    
    %CIDistribution is the bootstrap distribution or leave-one-out
    %jackknife estimates

    if nargin<3
        subtractMean = false;
    end
    if nargin<4
        CIMode = 'none';
    end
    if nargin<5
        CIAlpha = 0.05;
    end
    if nargin<6
        CIResamples = 10000;
    end
    
    assert(iscell(class1) && isvector(class1));
    assert(iscell(class2) && isvector(class2));
    class1 = makecol(class1);
    class2 = makecol(class2);
    nDims = numel(class1);
    nTime = size(class1{1}, 2);
    
    obsMat = cellfun(@(x) size(x, 1), [class1'; class2']); % 2 x D
    smallAsMatrices = true;
    [bigFoldMatrices, smallFold, nFolds] = getSequentialFoldIndicatorMatrices(obsMat, 'normalizeForAveraging', true, 'strategy', 'min', 'smallAsMatrices', smallAsMatrices); % C x D { nFolds x obsMat(c, d) } logical indicator matrices
    
    % if we multiply foldMatrices{1, d} * class1{d}, we have sum of observations included in each fold as nFolds x T
    % if instead of using foldMatrices directly, we normalize by the number of included rows to average appropriately
    % want to assmelbe these set-wise means into F x T x D matrices, which can then be reshaped and multiplied
    
    [bigMeans1, smallMeans1, bigMeans2, smallMeans2] = deal(nan(nFolds, nTime, nDims));
    for d = 1:nDims
        bigMeans1(:, :, d) = bigFoldMatrices{1, d} * class1{d}; % (F x nObs) * (nObs x T) --> F x T
        
        bigMeans2(:, :, d) = bigFoldMatrices{2, d} * class2{d}; % (F x nObs) * (nObs x T) --> F x T
        
        if smallAsMatrices
            smallMeans1(:, :, d) = smallFold{1, d} * class1{d}; % (F x nObs) * (nObs x T) --> F x T
            smallMeans2(:, :, d) = smallFold{2, d} * class2{d}; % (F x nObs) * (nObs x T) --> F x T
        else
            smallMeans1(:, :, d) = class1{d}(smallFold{1, d}, :); % nObs x T --> F x T
            smallMeans2(:, :, d) = class2{d}(smallFold{2, d}, :); % (F x nObs) * (nObs x T) --> F x T
        end 
    end
    
    meanDiff_bigSet = bigMeans1 - bigMeans2; % F x T x D 
    meanDiff_smallSet = smallMeans1 - smallMeans2; % F x T x D
        
    if subtractMean
        meanDiff_bigSet = meanDiff_bigSet - mean(meanDiff_bigSet, 3);
        meanDiff_smallSet = meanDiff_smallSet - mean(meanDiff_smallSet, 3);
    end
    
    squaredDistEstimates = sum(meanDiff_bigSet .* meanDiff_smallSet, 3); % F x T x D --> F x T
    
    squaredDistance = mean(squaredDistEstimates, 1); % F x T --> 1 x T
    euclideanDistance = sign(squaredDistance).*sqrt(abs(squaredDistance)); % 1 x T
    
    %compute confidence interval if requensted
    if ~strcmp(CIMode, 'none')
        wrapperFun = @(x,y)(ciWrapper(x,y,subtractMean));
        [CI, CIDistribution] = cvCISequential([euclideanDistance; squaredDistance], wrapperFun, {class1, class2}, CIMode, CIAlpha, CIResamples);
    else
        CI = [];
        CIDistribution = [];
    end
end

function output = ciWrapper(class1, class2, subtractMean)
    [ euclideanDistance, squaredDistance ] = cvDistanceSequential( class1, class2, subtractMean );
    output = [euclideanDistance; squaredDistance];
end

function vec = makecol( vec )

    % transpose if it's currently a row vector (unless its 0 x 1, keep as is)
    if (size(vec,2) > size(vec, 1) && isvector(vec)) && ~(size(vec, 1) == 0 && size(vec, 2) == 1)
        vec = vec';
    end

    if size(vec, 1) == 1 && size(vec, 2) == 0
        vec = vec';
    end

end



