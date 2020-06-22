function [ euclideanDistance, squaredDistance, CI, CIDistribution ] = cvDistanceProjectedSequential( class1, class2, projectionKbyD, subtractMean, CIMode, CIAlpha, CIResamples)
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

    if nargin<4
        subtractMean = false;
    end
    if nargin<5
        CIMode = 'none';
    end
    if nargin<6
        CIAlpha = 0.05;
    end
    if nargin<7
        CIResamples = 10000;
    end
    
    assert(iscell(class1) && isvector(class1));
    assert(iscell(class2) && isvector(class2));
    class1 = makecol(class1);
    class2 = makecol(class2);
    nDims = numel(class1);
    nTime = size(class1{1}, 2);
    
    assert(size(projectionKbyD, 2) == nDims);
    W = projectionKbyD'; % K by D --> D x K
    
    obsMat = cellfun(@(x) size(x, 1), [class1'; class2']); % 2 x D
    [bigFoldMatrices, smallFoldMatrices, ~, nObsBig, nObsSmall] = getSequentialFoldIndicatorMatrices(obsMat); % C x D { nFolds x obsMat(c, d) } logical indicator matrices
    
    % compute the big (Delta) and small (delta) fold differences for each fold (for each dimension)
    % if we multiply foldMatrices{1, d} * class1{d}, we have sum of observations included in each fold as nFolds x T
    % if instead of using foldMatrices directly, we normalize by the number of included rows to average appropriately
    % want to assmelbe these set-wise means into F x T x D matrices, which can then be reshaped and multiplied
    [mean_Delta, mean_delta] = deal(nan(nDims, nTime)); 
    A = nan(nDims, nDims, nTime); % D x D x T
    for d = 1:nDims
        bigMeans1 = bigFoldMatrices{1, d} * class1{d} ./ nObsBig{1,d}; % (F x nObs) * (nObs x T) --> F x T
        bigMeans2 = bigFoldMatrices{2, d} * class2{d} ./ nObsBig{2,d}; % (F x nObs) * (nObs x T) --> F x T
        smallMeans1 = smallFoldMatrices{1, d} * class1{d}./ nObsSmall{1,d}; % (F x nObs) * (nObs x T) --> F x T
        smallMeans2 = smallFoldMatrices{2, d} * class2{d} ./ nObsSmall{2,d}; % (F x nObs) * (nObs x T) --> F x T
        
        mean_Delta(d, :) = mean(bigMeans1 - bigMeans2, 1); % F x T --> 1 x T
        mean_delta(d, :) = mean(smallMeans1 - smallMeans2, 1); % F x T --> 1 x T
        
        % and fill in the diagonal term of A
        A(d, d, :) = mean( (bigMeans1 - bigMeans2) .* (smallMeans1 - smallMeans2), 1); % F x T --> 1 x T
    end

    % now loop over each pair of dimensions to fill in the off-diagonal terms
    for d = 1:nDims
        for e = 1:nDims
            if d == e, continue, end
            A(d, e, :) = mean_Delta(d, :) .* mean_delta(e, :); % 1 x 1 x T
        end
    end

    % Now we have A (D x D x T) and want to compute
    squaredDistance = nan(1, nTime);
    for t = 1:nTime
        squaredDistance(t) = trace(W' * A(:, :, t) * W);
    end
   euclideanDistance = sign(squaredDistance).*sqrt(abs(squaredDistance)); % 1 x T
    
    %compute confidence interval if requensted
    if ~strcmp(CIMode, 'none')
        wrapperFun = @(x,y)(ciWrapper(x,y, projectionKbyD, subtractMean));
        [CI, CIDistribution] = cvCISequential([euclideanDistance; squaredDistance], wrapperFun, {class1, class2}, CIMode, CIAlpha, CIResamples);
    else
        CI = [];
        CIDistribution = [];
    end
end

function output = ciWrapper(class1, class2, projectionKbyD, subtractMean)
    [ euclideanDistance, squaredDistance ] = cvDistanceProjectedSequential( class1, class2, projectionKbyD, subtractMean );
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



